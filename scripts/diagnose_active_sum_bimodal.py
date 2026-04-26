"""Diagnose multi-peak active_sum distributions for Timepix event features.

The script intentionally avoids matplotlib so it can run in lightweight
environments. It uses numpy plus Pillow for PNG outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


RGB = tuple[int, int, int]


@dataclass(frozen=True)
class FeatureRow:
    file: str
    active_count: float
    active_sum: float
    active_mean: float
    active_var: float
    active_min: float
    active_max: float
    bbox_aspect_ratio: float


def parse_float(value: object) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_features(path: Path) -> list[FeatureRow]:
    rows: list[FeatureRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = FeatureRow(
                file=str(raw.get("file", "")),
                active_count=parse_float(raw.get("active_count")),
                active_sum=parse_float(raw.get("active_sum")),
                active_mean=parse_float(raw.get("active_mean")),
                active_var=parse_float(raw.get("active_var")),
                active_min=parse_float(raw.get("active_min")),
                active_max=parse_float(raw.get("active_max")),
                bbox_aspect_ratio=parse_float(raw.get("bbox_aspect_ratio")),
            )
            if not math.isnan(row.active_sum):
                rows.append(row)
    return rows


def percentile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return math.nan
    return float(np.percentile(values, q))


def describe(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if len(arr) == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "q25": math.nan,
            "median": math.nan,
            "q75": math.nan,
            "max": math.nan,
        }
    return {
        "n": float(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "q25": percentile(arr, 25),
        "median": percentile(arr, 50),
        "q75": percentile(arr, 75),
        "max": float(np.max(arr)),
    }


def find_peaks(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, int]]]:
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    kernel = np.asarray([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    smooth = np.convolve(counts, kernel, mode="same")
    peaks: list[tuple[float, float, int]] = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]:
            peaks.append((float(smooth[i]), float(centers[i]), int(counts[i])))
    peaks.sort(reverse=True)
    return counts, edges, peaks


def valley_between_top_peaks(
    counts: np.ndarray,
    edges: np.ndarray,
    peaks: list[tuple[float, float, int]],
    min_separation: float | None = None,
) -> float:
    if len(peaks) < 2:
        return float(np.median((edges[:-1] + edges[1:]) / 2))
    centers = (edges[:-1] + edges[1:]) / 2
    kernel = np.asarray([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    smooth = np.convolve(counts, kernel, mode="same")
    if min_separation is None:
        min_separation = max(1200.0, 0.15 * float(edges[-1] - edges[0]))

    peak_pair: tuple[tuple[float, float, int], tuple[float, float, int]] | None = None
    for i, peak_a in enumerate(peaks):
        for peak_b in peaks[i + 1 :]:
            if abs(peak_a[1] - peak_b[1]) >= min_separation:
                peak_pair = (peak_a, peak_b)
                break
        if peak_pair is not None:
            break
    if peak_pair is None:
        peak_pair = (peaks[0], peaks[1])

    left, right = sorted([peak_pair[0][1], peak_pair[1][1]])
    mask = np.where((centers > left) & (centers < right))[0]
    if len(mask) == 0:
        return float((left + right) / 2)
    valley_idx = int(mask[np.argmin(smooth[mask])])
    return float(centers[valley_idx])


def safe_font(size: int = 14) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    x_label: str,
    y_label: str,
    title: str,
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = box
    grid = (230, 230, 230)
    axis = (35, 35, 35)
    for i in range(6):
        x = int(left + (right - left) * i / 5)
        y = int(bottom - (bottom - top) * i / 5)
        draw.line([(x, top), (x, bottom)], fill=grid)
        draw.line([(left, y), (right, y)], fill=grid)
    draw.rectangle(box, outline=axis, width=2)
    draw.text((left, 12), title, fill=axis, font=font)
    draw.text(((left + right) // 2 - 70, bottom + 34), x_label, fill=axis, font=font)
    draw.text((10, (top + bottom) // 2 - 10), y_label, fill=axis, font=font)


def plot_histogram(
    values: np.ndarray,
    threshold: float,
    out_path: Path,
    bins: int,
    title: str,
) -> None:
    width, height = 1200, 720
    margin = (95, 65, 55, 105)
    left, top = margin[0], margin[1]
    right, bottom = width - margin[2], height - margin[3]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = safe_font(22)
    small = safe_font(14)
    counts, edges = np.histogram(values, bins=bins)
    max_count = max(int(np.max(counts)), 1)
    for count, x0, x1 in zip(counts, edges[:-1], edges[1:]):
        px0 = int(left + (x0 - edges[0]) / (edges[-1] - edges[0]) * (right - left))
        px1 = int(left + (x1 - edges[0]) / (edges[-1] - edges[0]) * (right - left))
        py = int(bottom - count / max_count * (bottom - top))
        color: RGB = (45, 128, 184) if (x0 + x1) / 2 < threshold else (202, 96, 69)
        draw.rectangle((px0, py, max(px0 + 1, px1 - 1), bottom), fill=color)
    tx = int(left + (threshold - edges[0]) / (edges[-1] - edges[0]) * (right - left))
    draw.line((tx, top, tx, bottom), fill=(30, 30, 30), width=3)
    draw.text((tx + 8, top + 8), f"valley={threshold:.1f}", fill=(30, 30, 30), font=small)
    draw_axes(draw, (left, top, right, bottom), "active_sum", "count", title, font)
    image.save(out_path)


def plot_scatter(
    rows: list[FeatureRow],
    threshold: float,
    out_path: Path,
    title: str,
    max_points: int = 40000,
) -> None:
    width, height = 1200, 720
    margin = (95, 65, 55, 105)
    left, top = margin[0], margin[1]
    right, bottom = width - margin[2], height - margin[3]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = safe_font(22)
    rows_plot = rows
    if len(rows_plot) > max_points:
        rows_plot = random.Random(42).sample(rows_plot, max_points)
    xs = np.asarray([r.active_count for r in rows if not math.isnan(r.active_count)])
    ys = np.asarray([r.active_mean for r in rows if not math.isnan(r.active_mean)])
    x_min, x_max = float(np.percentile(xs, 0.5)), float(np.percentile(xs, 99.5))
    y_min, y_max = float(np.percentile(ys, 0.5)), float(np.percentile(ys, 99.5))
    x_min = min(0.0, x_min)
    for r in rows_plot:
        if math.isnan(r.active_count) or math.isnan(r.active_mean):
            continue
        x = max(x_min, min(x_max, r.active_count))
        y = max(y_min, min(y_max, r.active_mean))
        px = int(left + (x - x_min) / (x_max - x_min) * (right - left))
        py = int(bottom - (y - y_min) / (y_max - y_min) * (bottom - top))
        color: RGB = (38, 118, 177) if r.active_sum < threshold else (202, 76, 62)
        draw.point((px, py), fill=color)
    draw_axes(draw, (left, top, right, bottom), "active_count", "active_mean", title, font)
    image.save(out_path)


def plot_heatmap(rows: list[FeatureRow], out_path: Path, title: str, bins: int) -> None:
    width, height = 1200, 720
    margin = (95, 65, 55, 105)
    left, top = margin[0], margin[1]
    right, bottom = width - margin[2], height - margin[3]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = safe_font(22)
    counts = np.asarray([r.active_count for r in rows if not math.isnan(r.active_count)])
    sums = np.asarray([r.active_sum for r in rows if not math.isnan(r.active_sum)])
    x_min, x_max = 0, int(np.percentile(counts, 99.5)) + 1
    y_min, y_max = float(np.percentile(sums, 0.5)), float(np.percentile(sums, 99.5))
    heat = np.zeros((bins, x_max - x_min + 1), dtype=int)
    for r in rows:
        if math.isnan(r.active_count) or math.isnan(r.active_sum):
            continue
        x = int(round(r.active_count))
        if x < x_min or x > x_max:
            continue
        y = max(y_min, min(y_max, r.active_sum))
        yi = int((y - y_min) / (y_max - y_min) * (bins - 1))
        heat[bins - 1 - yi, x - x_min] += 1
    vmax = max(int(heat.max()), 1)
    cell_w = (right - left) / heat.shape[1]
    cell_h = (bottom - top) / heat.shape[0]
    for yi in range(heat.shape[0]):
        for xi in range(heat.shape[1]):
            value = heat[yi, xi]
            if value == 0:
                continue
            t = math.log1p(value) / math.log1p(vmax)
            color = (int(255 - 210 * t), int(245 - 150 * t), int(230 - 175 * t))
            x0 = int(left + xi * cell_w)
            y0 = int(top + yi * cell_h)
            x1 = int(left + (xi + 1) * cell_w)
            y1 = int(top + (yi + 1) * cell_h)
            draw.rectangle((x0, y0, x1, y1), fill=color)
    draw_axes(draw, (left, top, right, bottom), "active_count", "active_sum", title, font)
    image.save(out_path)


def read_matrix(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append([float(item) for item in stripped.split()])
    return np.asarray(rows, dtype=float)


def crop_active_region(matrix: np.ndarray, pad: int = 3) -> np.ndarray:
    coords = np.argwhere(matrix > 0)
    if coords.size == 0:
        return matrix
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    y0 = max(0, int(y0) - pad)
    x0 = max(0, int(x0) - pad)
    y1 = min(matrix.shape[0], int(y1) + pad)
    x1 = min(matrix.shape[1], int(x1) + pad)
    cropped = matrix[y0:y1, x0:x1]
    side = max(cropped.shape)
    canvas = np.zeros((side, side), dtype=matrix.dtype)
    oy = (side - cropped.shape[0]) // 2
    ox = (side - cropped.shape[1]) // 2
    canvas[oy : oy + cropped.shape[0], ox : ox + cropped.shape[1]] = cropped
    return canvas


def matrix_tile(matrix: np.ndarray, size: int = 160, crop: bool = False) -> Image.Image:
    if crop:
        matrix = crop_active_region(matrix)
    if matrix.size == 0:
        return Image.new("RGB", (size, size), "white")
    vmax = float(np.percentile(matrix[matrix > 0], 99)) if np.any(matrix > 0) else 1.0
    vmax = max(vmax, 1.0)
    norm = np.clip(matrix / vmax, 0, 1)
    rgb = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = np.asarray(255 * norm, dtype=np.uint8)
    rgb[..., 1] = np.asarray(220 * np.sqrt(norm), dtype=np.uint8)
    rgb[..., 2] = np.asarray(80 * (1 - norm), dtype=np.uint8)
    rgb[matrix <= 0] = 245
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((size, size), resample=Image.Resampling.NEAREST)


def select_samples(rows: list[FeatureRow], threshold: float, n: int) -> list[tuple[str, FeatureRow]]:
    low_peak = np.median([r.active_sum for r in rows if r.active_sum < threshold])
    high_peak = np.median([r.active_sum for r in rows if r.active_sum >= threshold])
    low = sorted((r for r in rows if r.active_sum < threshold), key=lambda r: abs(r.active_sum - low_peak))[:n]
    high = sorted((r for r in rows if r.active_sum >= threshold), key=lambda r: abs(r.active_sum - high_peak))[:n]
    return [("low", r) for r in low] + [("high", r) for r in high]


def make_montage(
    samples: list[tuple[str, FeatureRow]],
    raw_dir: Path,
    out_path: Path,
    tile_size: int = 150,
    crop: bool = False,
) -> None:
    cols = 8
    rows_n = math.ceil(len(samples) / cols)
    label_h = 46
    gap = 10
    width = cols * tile_size + (cols + 1) * gap
    height = rows_n * (tile_size + label_h) + (rows_n + 1) * gap
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = safe_font(12)
    for i, (group, row) in enumerate(samples):
        x = gap + (i % cols) * (tile_size + gap)
        y = gap + (i // cols) * (tile_size + label_h + gap)
        matrix_path = raw_dir / row.file
        if matrix_path.exists():
            tile = matrix_tile(read_matrix(matrix_path), size=tile_size, crop=crop)
        else:
            tile = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
            draw.text((x + 8, y + 8), "missing", fill=(180, 0, 0), font=font)
        image.paste(tile, (x, y))
        color: RGB = (38, 118, 177) if group == "low" else (202, 76, 62)
        draw.rectangle((x, y, x + tile_size, y + tile_size), outline=color, width=3)
        label = f"{group} sum={row.active_sum:.0f} cnt={row.active_count:.0f}"
        draw.text((x, y + tile_size + 4), label, fill=(30, 30, 30), font=font)
        draw.text((x, y + tile_size + 22), row.file[:24], fill=(80, 80, 80), font=font)
    image.save(out_path)


def corr(xs: list[float], ys: list[float]) -> float:
    arr_x = np.asarray(xs, dtype=float)
    arr_y = np.asarray(ys, dtype=float)
    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    if int(mask.sum()) < 2:
        return math.nan
    return float(np.corrcoef(arr_x[mask], arr_y[mask])[0, 1])


def write_outputs(
    rows: list[FeatureRow],
    counts: np.ndarray,
    edges: np.ndarray,
    peaks: list[tuple[float, float, int]],
    threshold: float,
    out_dir: Path,
) -> None:
    low = [r for r in rows if r.active_sum < threshold]
    high = [r for r in rows if r.active_sum >= threshold]

    columns = [
        "active_sum",
        "active_count",
        "active_mean",
        "active_var",
        "active_min",
        "active_max",
        "bbox_aspect_ratio",
    ]
    with (out_dir / "group_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "feature", "n", "mean", "std", "min", "q25", "median", "q75", "max"])
        for group_name, group_rows in [("low", low), ("high", high), ("all", rows)]:
            for column in columns:
                stats = describe(getattr(r, column) for r in group_rows)
                writer.writerow(
                    [
                        group_name,
                        column,
                        int(stats["n"]),
                        stats["mean"],
                        stats["std"],
                        stats["min"],
                        stats["q25"],
                        stats["median"],
                        stats["q75"],
                        stats["max"],
                    ]
                )

    with (out_dir / "peaks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "smoothed_count", "center", "raw_count"])
        for rank, (smooth, center, raw_count) in enumerate(peaks[:10], start=1):
            writer.writerow([rank, smooth, center, raw_count])

    with (out_dir / "event_groups.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "group", "active_sum", "active_count", "active_mean", "active_min", "active_max"])
        for row in rows:
            writer.writerow(
                [
                    row.file,
                    "low" if row.active_sum < threshold else "high",
                    row.active_sum,
                    row.active_count,
                    row.active_mean,
                    row.active_min,
                    row.active_max,
                ]
            )

    active_sum = [r.active_sum for r in rows]
    summary_lines = [
        f"n_events: {len(rows)}",
        f"active_sum_valley: {threshold:.6f}",
        f"low_group_n: {len(low)} ({len(low) / len(rows):.4%})",
        f"high_group_n: {len(high)} ({len(high) / len(rows):.4%})",
        "",
        "top_peaks:",
    ]
    for rank, (smooth, center, raw_count) in enumerate(peaks[:6], start=1):
        summary_lines.append(f"  {rank}. center={center:.3f}, smoothed_count={smooth:.3f}, raw_count={raw_count}")
    summary_lines += [
        "",
        "correlations_with_active_sum:",
        f"  active_count: {corr(active_sum, [r.active_count for r in rows]):.6f}",
        f"  active_mean: {corr(active_sum, [r.active_mean for r in rows]):.6f}",
        f"  active_var: {corr(active_sum, [r.active_var for r in rows]):.6f}",
        f"  active_min: {corr(active_sum, [r.active_min for r in rows]):.6f}",
        f"  active_max: {corr(active_sum, [r.active_max for r in rows]):.6f}",
        f"  bbox_aspect_ratio: {corr(active_sum, [r.bbox_aspect_ratio for r in rows]):.6f}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--angle", type=str, default="")
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--sample-per-group", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--min-peak-separation",
        type=float,
        default=None,
        help="Minimum active_sum distance between the two peaks used to choose the valley.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_features(args.features_csv)
    if not rows:
        raise SystemExit(f"No usable rows found in {args.features_csv}")

    values = np.asarray([r.active_sum for r in rows], dtype=float)
    counts, edges, peaks = find_peaks(values, args.bins)
    threshold = (
        args.threshold
        if args.threshold is not None
        else valley_between_top_peaks(counts, edges, peaks, args.min_peak_separation)
    )
    angle = f" angle={args.angle}" if args.angle else ""

    write_outputs(rows, counts, edges, peaks, threshold, args.out_dir)
    plot_histogram(
        values,
        threshold,
        args.out_dir / "hist_active_sum_groups.png",
        args.bins,
        f"active_sum distribution{angle}",
    )
    plot_scatter(
        rows,
        threshold,
        args.out_dir / "scatter_active_count_vs_mean.png",
        f"active_count vs active_mean{angle}",
    )
    plot_heatmap(
        rows,
        args.out_dir / "heatmap_active_count_active_sum.png",
        f"active_count vs active_sum density{angle}",
        bins=70,
    )
    samples = select_samples(rows, threshold, args.sample_per_group)
    make_montage(samples, args.raw_dir, args.out_dir / "sample_montage_low_high.png")
    make_montage(samples, args.raw_dir, args.out_dir / "sample_montage_low_high_cropped.png", crop=True)
    print(f"events: {len(rows)}")
    print(f"threshold: {threshold:.3f}")
    print(f"out_dir: {args.out_dir}")


if __name__ == "__main__":
    main()
