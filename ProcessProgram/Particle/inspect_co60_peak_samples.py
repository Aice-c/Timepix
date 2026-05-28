#!/usr/bin/env python3
"""Inspect representative Co60 candidates from active/aspect joint peaks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_STAGE1_ROOT = Path(r"E:\TimepixData\particle\stage1_single_particle_candidates_100x100")


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
    df["bbox_aspect_key"] = df["bbox_aspect_ratio"].round(6)
    return df


def top_joint_modes(df: pd.DataFrame, count: int) -> pd.DataFrame:
    modes = (
        df.groupby(["active_pixel_count", "bbox_aspect_key"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .head(count)
    )
    return modes.rename(columns={"size": "candidate_count"})


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


def sample_peak_rows(df: pd.DataFrame, modes: pd.DataFrame, samples_per_peak: int, seed: int) -> pd.DataFrame:
    sampled: list[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    for mode_index, mode in modes.reset_index(drop=True).iterrows():
        sub = df[
            (df["active_pixel_count"] == mode["active_pixel_count"])
            & (df["bbox_aspect_key"] == mode["bbox_aspect_key"])
        ].copy()
        if sub.empty:
            continue
        take = min(samples_per_peak, len(sub))
        random_state = int(rng.integers(0, 2**31 - 1))
        picked = sub.sample(n=take, random_state=random_state).copy()
        picked["mode_index"] = int(mode_index + 1)
        picked["mode_candidate_count"] = int(mode["candidate_count"])
        sampled.append(picked)
    if not sampled:
        return pd.DataFrame()
    return pd.concat(sampled, ignore_index=True)


def matrix_path(stage1_root: Path, row: pd.Series, modality: str) -> Path:
    key = "tot_path" if modality == "ToT" else "toa_path"
    return stage1_root / "dataset" / str(row[key])


def plot_samples(
    sampled: pd.DataFrame,
    stage1_root: Path,
    output_dir: Path,
    modality: str,
    samples_per_peak: int,
) -> None:
    if sampled.empty:
        return
    modes = sampled["mode_index"].drop_duplicates().tolist()
    rows_per_mode = 2
    cols = int(np.ceil(samples_per_peak / rows_per_mode))
    total_rows = rows_per_mode * len(modes)
    fig, axes = plt.subplots(total_rows, cols, figsize=(1.35 * cols, 1.45 * total_rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(total_rows, cols)

    values_for_scale = []
    canvases: list[tuple[pd.Series, np.ndarray]] = []
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

    for mode_pos, mode in enumerate(modes):
        sub = [(row, canvas) for row, canvas in canvases if int(row["mode_index"]) == int(mode)]
        for sample_idx, (row, canvas) in enumerate(sub):
            plot_row = mode_pos * rows_per_mode + sample_idx // cols
            plot_col = sample_idx % cols
            ax = axes[plot_row, plot_col]
            ax.imshow(np.log1p(canvas), cmap="magma", interpolation="nearest", vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"pix={int(row['active_pixel_count'])}, asp={row['bbox_aspect_ratio']:.1f}\nToT={row['total_ToT']:.1f}",
                fontsize=6,
            )
            if sample_idx == 0:
                ax.set_ylabel(
                    f"Peak {int(mode)}\nn={int(row['mode_candidate_count'])}",
                    fontsize=7,
                    rotation=0,
                    labelpad=24,
                    va="center",
                )

    fig.suptitle(f"Co60 top active/aspect joint-peak samples ({modality}, 10x10 centered crop)", fontsize=9)
    save_figure(fig, output_dir / f"co60_top_joint_peak_{modality.lower()}_samples_10x10")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Co60 samples from top active/aspect joint peaks.")
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <stage1-root>/figures/cleaning_diagnostics/co60_peak_samples",
    )
    parser.add_argument("--num-peaks", type=int, default=2)
    parser.add_argument("--samples-per-peak", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modalities", nargs="+", choices=["ToT", "ToA"], default=["ToT"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    output_dir = args.output_dir or (
        args.stage1_root / "figures" / "cleaning_diagnostics" / "co60_peak_samples"
    )
    df = load_co60_manifest(args.stage1_root)
    modes = top_joint_modes(df, args.num_peaks)
    sampled = sample_peak_rows(df, modes, args.samples_per_peak, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes.to_csv(output_dir / "co60_top_joint_peak_modes.csv", index=False)
    sampled.to_csv(output_dir / "co60_top_joint_peak_sample_manifest.csv", index=False)
    for modality in args.modalities:
        plot_samples(sampled, args.stage1_root, output_dir, modality, args.samples_per_peak)
    print("Selected Co60 joint modes:")
    print(modes.to_string(index=False))
    print(f"Sample manifest written to: {output_dir / 'co60_top_joint_peak_sample_manifest.csv'}")
    print(f"Figures written to: {output_dir}")


if __name__ == "__main__":
    main()
