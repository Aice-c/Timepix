#!/usr/bin/env python3
"""Inspect Co60 candidate morphology by active-pixel-count bins."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_STAGE1_ROOT = Path(r"E:\TimepixData\particle\stage1_single_particle_candidates_100x100")

ACTIVE_BINS = [
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4", 4, 4),
    ("5-6", 5, 6),
    ("7-8", 7, 8),
    ("9-12", 9, 12),
    ("13-20", 13, 20),
    ("21-40", 21, 40),
    (">40", 41, None),
]


def assign_active_bin(active_pixels: int | float) -> str:
    value = int(active_pixels)
    for label, lower, upper in ACTIVE_BINS:
        if upper is None and value >= lower:
            return label
        if upper is not None and lower <= value <= upper:
            return label
    return "out_of_range"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.8,
            "figure.dpi": 120,
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")


def load_co60_manifest(stage1_root: Path) -> pd.DataFrame:
    manifest_path = stage1_root / "manifests" / "extraction_manifest.csv"
    usecols = [
        "sample_key",
        "particle",
        "source_subdir",
        "active_pixel_count",
        "total_ToT",
        "mean_ToT_nonzero",
        "bbox_long",
        "bbox_short",
        "tot_path",
        "toa_path",
    ]
    df = pd.read_csv(manifest_path, usecols=usecols)
    df = df[df["particle"] == "Co60"].copy()
    numeric_cols = ["active_pixel_count", "total_ToT", "mean_ToT_nonzero", "bbox_long", "bbox_short"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols + ["tot_path", "toa_path"]).copy()
    df["bbox_aspect_ratio"] = df["bbox_long"] / df["bbox_short"].clip(lower=1)
    df["active_bin"] = df["active_pixel_count"].map(assign_active_bin)
    return df


def center_nonzero_to_canvas(matrix: np.ndarray, size: int = 10) -> np.ndarray:
    coords = np.argwhere(matrix > 0)
    canvas = np.zeros((size, size), dtype=np.float32)
    if coords.size == 0:
        return canvas
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    crop = matrix[r0:r1, c0:c1]
    height, width = crop.shape
    if height > size or width > size:
        crop = crop[:size, :size]
        height, width = crop.shape
    top = (size - height) // 2
    left = (size - width) // 2
    canvas[top : top + height, left : left + width] = crop
    return canvas


def sample_rows_by_bin(df: pd.DataFrame, samples_per_bin: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    for label, _, _ in ACTIVE_BINS:
        sub = df[df["active_bin"] == label].copy()
        if sub.empty:
            continue
        take = min(samples_per_bin, len(sub))
        random_state = int(rng.integers(0, 2**31 - 1))
        picked = sub.sample(n=take, random_state=random_state).copy()
        picked["active_bin_total_count"] = int(len(sub))
        rows.append(picked)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def matrix_path(stage1_root: Path, row: pd.Series, modality: str) -> Path:
    key = "tot_path" if modality == "ToT" else "toa_path"
    return stage1_root / "dataset" / str(row[key])


def plot_samples_by_bin(
    sampled: pd.DataFrame,
    stage1_root: Path,
    output_dir: Path,
    modality: str,
    samples_per_bin: int,
) -> None:
    labels = [label for label, _, _ in ACTIVE_BINS if label in set(sampled["active_bin"])]
    if not labels:
        return
    fig, axes = plt.subplots(
        len(labels),
        samples_per_bin,
        figsize=(1.35 * samples_per_bin, 1.45 * len(labels)),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(labels), samples_per_bin)

    canvases: list[tuple[pd.Series, np.ndarray]] = []
    values_for_scale: list[float] = []
    for _, row in sampled.iterrows():
        matrix = np.loadtxt(matrix_path(stage1_root, row, modality), dtype=np.float32)
        canvas = center_nonzero_to_canvas(matrix, size=10)
        canvases.append((row, canvas))
        nonzero = canvas[canvas > 0]
        if nonzero.size:
            values_for_scale.extend(np.log1p(nonzero).tolist())
    vmax = float(np.quantile(values_for_scale, 0.98)) if values_for_scale else 1.0

    for ax in axes.ravel():
        ax.axis("off")

    for row_idx, label in enumerate(labels):
        sub = [(row, canvas) for row, canvas in canvases if row["active_bin"] == label]
        for col_idx, (row, canvas) in enumerate(sub[:samples_per_bin]):
            ax = axes[row_idx, col_idx]
            ax.imshow(np.log1p(canvas), cmap="magma", interpolation="nearest", vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"pix={int(row['active_pixel_count'])}, asp={row['bbox_aspect_ratio']:.1f}\nToT={row['total_ToT']:.1f}",
                fontsize=6,
            )
            if col_idx == 0:
                ax.set_ylabel(
                    f"{label}\nn={int(row['active_bin_total_count'])}",
                    fontsize=7,
                    rotation=0,
                    labelpad=22,
                    va="center",
                )

    fig.suptitle(f"Co60 samples by active-pixel-count bin ({modality}, 10x10 centered crop)", fontsize=9)
    save_figure(fig, output_dir / f"co60_active_bin_{modality.lower()}_samples_10x10")
    plt.close(fig)


def write_bin_counts(df: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for label, _, _ in ACTIVE_BINS:
        sub = df[df["active_bin"] == label]
        rows.append(
            {
                "active_bin": label,
                "count": int(len(sub)),
                "fraction": float(len(sub) / len(df)) if len(df) else 0.0,
                "median_total_ToT": float(sub["total_ToT"].median()) if not sub.empty else np.nan,
                "median_bbox_aspect_ratio": float(sub["bbox_aspect_ratio"].median()) if not sub.empty else np.nan,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "co60_active_bin_counts.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Co60 morphology by active-pixel-count bins.")
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <stage1-root>/figures/cleaning_diagnostics/co60_active_bin_samples",
    )
    parser.add_argument("--samples-per-bin", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modalities", nargs="+", choices=["ToT", "ToA"], default=["ToT"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    output_dir = args.output_dir or (
        args.stage1_root / "figures" / "cleaning_diagnostics" / "co60_active_bin_samples"
    )
    df = load_co60_manifest(args.stage1_root)
    sampled = sample_rows_by_bin(df, args.samples_per_bin, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_bin_counts(df, output_dir)
    sampled.to_csv(output_dir / "co60_active_bin_sample_manifest.csv", index=False)
    for modality in args.modalities:
        plot_samples_by_bin(sampled, args.stage1_root, output_dir, modality, args.samples_per_bin)
    print("Co60 active-bin counts:")
    print(pd.read_csv(output_dir / "co60_active_bin_counts.csv").to_string(index=False))
    print(f"Sample manifest written to: {output_dir / 'co60_active_bin_sample_manifest.csv'}")
    print(f"Figures written to: {output_dir}")


if __name__ == "__main__":
    main()
