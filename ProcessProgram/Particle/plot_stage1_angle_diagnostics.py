#!/usr/bin/env python3
"""Plot stage-1 particle cleaning diagnostics stratified by collection angle."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_STAGE1_ROOT = Path(r"E:\TimepixData\particle\stage1_single_particle_candidates_100x100")

PARTICLE_ORDER = ["Am", "Co60", "Sr"]
ANGLE_ORDER = [0, 45, 60, 90]
PARTICLE_COLORS = {
    "Am": "#4E79A7",
    "Co60": "#F28E2B",
    "Sr": "#59A14F",
}

FEATURE_HISTOGRAMS = [
    ("active_pixel_count", "Active pixels", "log10(active pixels + 1)", True),
    ("total_ToT", "Total ToT", "log10(total ToT + 1)", True),
    ("mean_ToT_nonzero", "Mean nonzero ToT", "log10(mean ToT + 1)", True),
    ("bbox_aspect_ratio", "Bbox aspect ratio", "bbox_long / bbox_short", False),
    ("bbox_long", "Bbox long side", "bbox_long", False),
    ("bbox_fill_ratio", "Bbox fill ratio", "active pixels / bbox area", False),
]


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


def parse_angle(source_subdir: object) -> int | None:
    text = str(source_subdir)
    numbers = re.findall(r"\d+", text)
    if not numbers:
        return None
    return int(numbers[-1])


def load_manifest(stage1_root: Path) -> pd.DataFrame:
    manifest_path = stage1_root / "manifests" / "extraction_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage-1 extraction manifest does not exist: {manifest_path}")
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
    ]
    df = pd.read_csv(manifest_path, usecols=usecols)
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
    df = df.dropna(subset=numeric_cols + ["particle", "source_subdir"]).copy()
    df["angle"] = df["source_subdir"].map(parse_angle)
    df = df.dropna(subset=["angle"]).copy()
    df["angle"] = df["angle"].astype(int)
    df["bbox_aspect_ratio"] = df["bbox_long"] / df["bbox_short"].clip(lower=1)
    df["bbox_area"] = df["bbox_long"] * df["bbox_short"]
    df["bbox_fill_ratio"] = df["active_pixel_count"] / df["bbox_area"].clip(lower=1)
    df["log_active"] = np.log10(df["active_pixel_count"].clip(lower=0) + 1.0)
    df["log_total_ToT"] = np.log10(df["total_ToT"].clip(lower=0) + 1.0)
    return df


def transformed(values: pd.Series, log_transform: bool) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    if log_transform:
        arr = np.log10(np.clip(arr, 0, None) + 1.0)
    return arr[np.isfinite(arr)]


def plot_hexbin_count_grid(
    df: pd.DataFrame,
    row_key: str,
    row_values: list[object],
    col_key: str,
    col_values: list[object],
    output_base: Path,
    row_label: str,
    col_label: str,
) -> None:
    fig, axes = plt.subplots(
        len(row_values),
        len(col_values),
        figsize=(2.05 * len(col_values), 1.85 * len(row_values)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(len(row_values), len(col_values))

    xlim = np.nanpercentile(df["log_active"], [0.2, 99.8])
    ylim = np.nanpercentile(df["log_total_ToT"], [0.2, 99.8])
    last_hb = None

    for row_idx, row_value in enumerate(row_values):
        for col_idx, col_value in enumerate(col_values):
            ax = axes[row_idx, col_idx]
            sub = df[(df[row_key] == row_value) & (df[col_key] == col_value)]
            if sub.empty:
                ax.axis("off")
                continue
            last_hb = ax.hexbin(
                sub["log_active"],
                sub["log_total_ToT"],
                gridsize=38,
                extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                mincnt=1,
                bins="log",
                cmap="viridis",
                linewidths=0,
            )
            med_x = float(sub["log_active"].median())
            med_y = float(sub["log_total_ToT"].median())
            ax.scatter([med_x], [med_y], s=10, color="white", edgecolor="black", linewidth=0.35, zorder=3)
            ax.axvline(med_x, color="white", linewidth=0.55, alpha=0.75)
            ax.axhline(med_y, color="white", linewidth=0.55, alpha=0.75)
            if row_idx == 0:
                ax.set_title(f"{col_label} {col_value}")
            if col_idx == 0:
                ax.set_ylabel(f"{row_label} {row_value}\nlog10(total ToT + 1)")
            if row_idx == len(row_values) - 1:
                ax.set_xlabel("log10(active pixels + 1)")
            ax.text(
                0.03,
                0.96,
                f"n={len(sub)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=5.5,
            )
    if last_hb is not None:
        cbar = fig.colorbar(last_hb, ax=axes.ravel().tolist(), shrink=0.75, pad=0.01)
        cbar.set_label("Candidate count per hexbin (log scale)")
    save_figure(fig, output_base)
    plt.close(fig)


def plot_particle_angle_hexbin_count(df: pd.DataFrame, output_dir: Path) -> None:
    for particle in PARTICLE_ORDER:
        sub = df[df["particle"] == particle]
        if sub.empty:
            continue
        plot_hexbin_count_grid(
            sub,
            row_key="particle",
            row_values=[particle],
            col_key="angle",
            col_values=ANGLE_ORDER,
            output_base=output_dir / f"{particle}_by_angle_active_total_hexbin_count",
            row_label="Particle",
            col_label="Angle",
        )


def plot_angle_particle_hexbin_count(df: pd.DataFrame, output_dir: Path) -> None:
    for angle in ANGLE_ORDER:
        sub = df[df["angle"] == angle]
        if sub.empty:
            continue
        plot_hexbin_count_grid(
            sub,
            row_key="angle",
            row_values=[angle],
            col_key="particle",
            col_values=PARTICLE_ORDER,
            output_base=output_dir / f"angle{angle}_by_particle_active_total_hexbin_count",
            row_label="Angle",
            col_label="Particle",
        )


def plot_particle_angle_feature_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    for particle in PARTICLE_ORDER:
        sub_particle = df[df["particle"] == particle]
        if sub_particle.empty:
            continue
        fig, axes = plt.subplots(
            len(FEATURE_HISTOGRAMS),
            len(ANGLE_ORDER),
            figsize=(2.05 * len(ANGLE_ORDER), 1.35 * len(FEATURE_HISTOGRAMS)),
            constrained_layout=True,
        )
        for row_idx, (col, title, xlabel, log_transform) in enumerate(FEATURE_HISTOGRAMS):
            all_values = transformed(sub_particle[col], log_transform)
            lo, hi = np.nanpercentile(all_values, [0.5, 99.5])
            if lo == hi:
                lo, hi = float(np.nanmin(all_values)), float(np.nanmax(all_values))
            bins = np.linspace(lo, hi, 45)
            for col_idx, angle in enumerate(ANGLE_ORDER):
                ax = axes[row_idx, col_idx]
                sub = sub_particle[sub_particle["angle"] == angle]
                if sub.empty:
                    ax.axis("off")
                    continue
                values = transformed(sub[col], log_transform)
                ax.hist(
                    values,
                    bins=bins,
                    color=PARTICLE_COLORS.get(particle, "0.3"),
                    alpha=0.45,
                    edgecolor=PARTICLE_COLORS.get(particle, "0.3"),
                    linewidth=0.7,
                )
                q50 = float(np.quantile(values, 0.50))
                ax.axvline(q50, color="black", linewidth=0.7)
                if row_idx == 0:
                    ax.set_title(f"{angle} deg")
                if col_idx == 0:
                    ax.set_ylabel(f"{title}\nCandidate count")
                ax.set_xlabel(xlabel)
                ax.grid(axis="y", color="0.9", linewidth=0.5)
        save_figure(fig, output_dir / f"{particle}_by_angle_feature_histograms_count")
        plt.close(fig)


def plot_angle_particle_feature_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    for angle in ANGLE_ORDER:
        sub_angle = df[df["angle"] == angle]
        if sub_angle.empty:
            continue
        fig, axes = plt.subplots(
            len(FEATURE_HISTOGRAMS),
            len(PARTICLE_ORDER),
            figsize=(2.15 * len(PARTICLE_ORDER), 1.35 * len(FEATURE_HISTOGRAMS)),
            constrained_layout=True,
        )
        for row_idx, (col, title, xlabel, log_transform) in enumerate(FEATURE_HISTOGRAMS):
            all_values = transformed(sub_angle[col], log_transform)
            lo, hi = np.nanpercentile(all_values, [0.5, 99.5])
            if lo == hi:
                lo, hi = float(np.nanmin(all_values)), float(np.nanmax(all_values))
            bins = np.linspace(lo, hi, 45)
            for col_idx, particle in enumerate(PARTICLE_ORDER):
                ax = axes[row_idx, col_idx]
                sub = sub_angle[sub_angle["particle"] == particle]
                if sub.empty:
                    ax.axis("off")
                    continue
                values = transformed(sub[col], log_transform)
                color = PARTICLE_COLORS.get(particle, "0.3")
                ax.hist(values, bins=bins, color=color, alpha=0.45, edgecolor=color, linewidth=0.7)
                q50 = float(np.quantile(values, 0.50))
                ax.axvline(q50, color="black", linewidth=0.7)
                if row_idx == 0:
                    ax.set_title(particle)
                if col_idx == 0:
                    ax.set_ylabel(f"{title}\nCandidate count")
                ax.set_xlabel(xlabel)
                ax.grid(axis="y", color="0.9", linewidth=0.5)
        save_figure(fig, output_dir / f"angle{angle}_by_particle_feature_histograms_count")
        plt.close(fig)


def write_quantiles(df: pd.DataFrame, output_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    feature_cols = [
        "active_pixel_count",
        "total_ToT",
        "mean_ToT_nonzero",
        "max_ToT",
        "bbox_long",
        "bbox_aspect_ratio",
        "bbox_fill_ratio",
    ]
    for (particle, angle), sub in df.groupby(["particle", "angle"]):
        for col in feature_cols:
            values = sub[col].to_numpy(dtype=float)
            rows.append(
                {
                    "particle": particle,
                    "angle": int(angle),
                    "feature": col,
                    "n": int(len(values)),
                    "q01": float(np.quantile(values, 0.01)),
                    "q05": float(np.quantile(values, 0.05)),
                    "q10": float(np.quantile(values, 0.10)),
                    "q25": float(np.quantile(values, 0.25)),
                    "median": float(np.quantile(values, 0.50)),
                    "q75": float(np.quantile(values, 0.75)),
                    "q90": float(np.quantile(values, 0.90)),
                    "q95": float(np.quantile(values, 0.95)),
                    "q99": float(np.quantile(values, 0.99)),
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["particle", "angle", "feature"]).to_csv(
        output_dir / "stage1_angle_feature_quantiles.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stage-1 angle-stratified particle diagnostics.")
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
        help="Output directory. Default: <stage1-root>/figures/cleaning_diagnostics/angle_diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    stage1_root = args.stage1_root
    output_dir = args.output_dir or (stage1_root / "figures" / "cleaning_diagnostics" / "angle_diagnostics")
    df = load_manifest(stage1_root)
    write_quantiles(df, output_dir)
    plot_particle_angle_hexbin_count(df, output_dir)
    plot_angle_particle_hexbin_count(df, output_dir)
    plot_particle_angle_feature_histograms(df, output_dir)
    plot_angle_particle_feature_histograms(df, output_dir)
    print(f"Loaded candidates: {len(df)}")
    print("Particle x angle counts:")
    counts = df.groupby(["particle", "angle"]).size().reset_index(name="n")
    for _, row in counts.sort_values(["particle", "angle"]).iterrows():
        print(f"  {row['particle']} angle={int(row['angle'])}: {int(row['n'])}")
    print(f"Angle diagnostics written to: {output_dir}")


if __name__ == "__main__":
    main()
