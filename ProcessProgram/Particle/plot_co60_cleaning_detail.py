#!/usr/bin/env python3
"""Detailed Co60 cleaning diagnostics for gamma-source candidate selection."""

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
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.dpi": 120,
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")


def load_co60(stage1_root: Path) -> pd.DataFrame:
    manifest_path = stage1_root / "manifests" / "extraction_manifest.csv"
    usecols = [
        "sample_key",
        "particle",
        "source_subdir",
        "active_pixel_count",
        "total_ToT",
        "mean_ToT_nonzero",
        "max_ToT",
        "bbox_long",
        "bbox_short",
        "tot_path",
    ]
    df = pd.read_csv(manifest_path, usecols=usecols)
    df = df[df["particle"] == "Co60"].copy()
    numeric_cols = [
        "active_pixel_count",
        "total_ToT",
        "mean_ToT_nonzero",
        "max_ToT",
        "bbox_long",
        "bbox_short",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols).copy()
    df["bbox_aspect_ratio"] = df["bbox_long"] / df["bbox_short"].clip(lower=1)
    df["log_active"] = np.log10(df["active_pixel_count"].clip(lower=0) + 1.0)
    df["log_total_ToT"] = np.log10(df["total_ToT"].clip(lower=0) + 1.0)
    return df


def add_quantile_lines(ax: plt.Axes, values: pd.Series, axis: str, quantiles: tuple[float, ...]) -> None:
    for q in quantiles:
        value = float(values.quantile(q))
        if axis == "x":
            ax.axvline(value, color="black", linestyle="--", linewidth=0.65, alpha=0.55)
            ax.text(value, ax.get_ylim()[1] * 0.97, f"P{int(q * 100)}", rotation=90, va="top", ha="right", fontsize=5.5)
        else:
            ax.axhline(value, color="black", linestyle="--", linewidth=0.65, alpha=0.55)
            ax.text(ax.get_xlim()[1] * 0.98, value, f"P{int(q * 100)}", va="bottom", ha="right", fontsize=5.5)


def plot_co60_active_total_hexbin_count(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    ax = axes[0]
    x_hi = float(df["active_pixel_count"].quantile(0.995))
    y_hi = float(df["total_ToT"].quantile(0.995))
    hb = ax.hexbin(
        df["active_pixel_count"],
        df["total_ToT"],
        gridsize=70,
        extent=(0, x_hi, 0, y_hi),
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
    )
    add_quantile_lines(ax, df["active_pixel_count"], "x", (0.50, 0.90, 0.95, 0.99))
    add_quantile_lines(ax, df["total_ToT"], "y", (0.50, 0.90, 0.95, 0.99))
    ax.set_title("Co60 active pixels vs total ToT\nlinear scale, 99.5% view")
    ax.set_xlabel("Active pixels")
    ax.set_ylabel("Total ToT")
    fig.colorbar(hb, ax=ax, shrink=0.85, label="Candidate count per hexbin (log scale)")

    ax = axes[1]
    zoom = df[(df["active_pixel_count"] <= 20) & (df["total_ToT"] <= 500)]
    hb = ax.hexbin(
        zoom["active_pixel_count"],
        zoom["total_ToT"],
        gridsize=50,
        extent=(0, 20, 0, 500),
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
    )
    ax.set_title(f"Co60 low-to-mid cluster zoom\nn={len(zoom)}")
    ax.set_xlabel("Active pixels")
    ax.set_ylabel("Total ToT")
    fig.colorbar(hb, ax=ax, shrink=0.85, label="Candidate count per hexbin (log scale)")

    for axis in axes.ravel():
        axis.grid(color="0.9", linewidth=0.5)

    save_figure(fig, output_dir / "co60_active_total_hexbin_count")
    plt.close(fig)


def plot_co60_bbox_aspect_histogram_count(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.0), constrained_layout=True)
    aspect = df["bbox_aspect_ratio"].clip(upper=8)
    bins = np.arange(1, 8.25, 0.25)
    ax.hist(aspect, bins=bins, color="#F28E2B", alpha=0.55, edgecolor="#F28E2B", linewidth=0.8)
    for q in (0.50, 0.75, 0.90, 0.95, 0.99):
        value = float(df["bbox_aspect_ratio"].quantile(q))
        ax.axvline(value, color="black", linestyle="--", linewidth=0.65, alpha=0.65)
        ax.text(value, ax.get_ylim()[1] * 0.95, f"P{int(q * 100)}", rotation=90, va="top", ha="right", fontsize=5.5)
    ax.set_title("Co60 bbox aspect ratio")
    ax.set_xlabel("bbox_long / bbox_short")
    ax.set_ylabel("Candidate count")
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    save_figure(fig, output_dir / "co60_bbox_aspect_histogram_count")
    plt.close(fig)


def plot_co60_feature_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    features = [
        ("active_pixel_count", "Active pixels", np.arange(0.5, 40.5, 1.0)),
        ("total_ToT", "Total ToT", np.linspace(0, float(df["total_ToT"].quantile(0.995)), 90)),
        ("mean_ToT_nonzero", "Mean nonzero ToT", np.linspace(0, float(df["mean_ToT_nonzero"].quantile(0.995)), 80)),
        ("bbox_aspect_ratio", "Bbox aspect ratio", np.arange(1, 8.25, 0.25)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), constrained_layout=True)
    for ax, (col, title, bins) in zip(axes.ravel(), features):
        values = df[col].clip(upper=bins[-1] if col == "bbox_aspect_ratio" else np.inf)
        ax.hist(values, bins=bins, color="#F28E2B", alpha=0.50, edgecolor="#F28E2B", linewidth=0.8)
        for q in (0.50, 0.90, 0.95, 0.99):
            value = float(df[col].quantile(q))
            ax.axvline(value, color="black", linestyle="--", linewidth=0.65, alpha=0.65)
            ax.text(value, ax.get_ylim()[1] * 0.95, f"P{int(q * 100)}", rotation=90, va="top", ha="right", fontsize=5.5)
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("Candidate count")
        ax.grid(axis="y", color="0.9", linewidth=0.5)
    save_figure(fig, output_dir / "co60_feature_histograms_count")
    plt.close(fig)


def write_co60_tables(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for col in ["active_pixel_count", "total_ToT", "mean_ToT_nonzero", "max_ToT", "bbox_aspect_ratio"]:
        values = df[col].to_numpy(dtype=float)
        rows.append(
            {
                "feature": col,
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "q01": float(np.quantile(values, 0.01)),
                "q05": float(np.quantile(values, 0.05)),
                "q10": float(np.quantile(values, 0.10)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "q90": float(np.quantile(values, 0.90)),
                "q95": float(np.quantile(values, 0.95)),
                "q99": float(np.quantile(values, 0.99)),
                "q995": float(np.quantile(values, 0.995)),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "co60_feature_quantiles.csv", index=False)

    masks = {
        "all": np.ones(len(df), dtype=bool),
        "low_pixel_gamma_like_active_1_8_total_15_260": (
            df["active_pixel_count"].between(1, 8) & df["total_ToT"].between(15, 260)
        ).to_numpy(),
        "tighter_active_2_8_total_30_250": (
            df["active_pixel_count"].between(2, 8) & df["total_ToT"].between(30, 250)
        ).to_numpy(),
        "track_like_active_ge_10_total_ge_180": (
            (df["active_pixel_count"] >= 10) & (df["total_ToT"] >= 180)
        ).to_numpy(),
        "elongated_aspect_ge_2": (df["bbox_aspect_ratio"] >= 2).to_numpy(),
    }
    count_rows = []
    for name, mask in masks.items():
        count_rows.append({"region": name, "count": int(mask.sum()), "fraction": float(mask.mean())})
    pd.DataFrame(count_rows).to_csv(output_dir / "co60_exploratory_region_counts.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot detailed Co60 cleaning diagnostics.")
    parser.add_argument(
        "--stage1-root",
        type=Path,
        default=DEFAULT_STAGE1_ROOT,
        help=f"Stage-1 extraction root. Default: {DEFAULT_STAGE1_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <stage1-root>/figures/cleaning_diagnostics/co60_detail",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    output_dir = args.output_dir or (args.stage1_root / "figures" / "cleaning_diagnostics" / "co60_detail")
    df = load_co60(args.stage1_root)
    plot_co60_active_total_hexbin_count(df, output_dir)
    plot_co60_bbox_aspect_histogram_count(df, output_dir)
    plot_co60_feature_histograms(df, output_dir)
    write_co60_tables(df, output_dir)
    print(f"Loaded Co60 candidates: {len(df)}")
    print(f"Co60 detail diagnostics written to: {output_dir}")


if __name__ == "__main__":
    main()
