#!/usr/bin/env python3
"""Plot stage-1 particle candidate cleaning diagnostics.

These figures are for deciding stage-2 cleaning thresholds. They are not meant
to compare model features; each nominal particle collection is inspected for its
main candidate cluster and off-cluster contaminants.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_STAGE1_ROOT = Path(r"E:\TimepixData\particle\stage1_single_particle_candidates_100x100")

PARTICLE_ORDER = ["Am", "Co60", "Sr"]
PARTICLE_COLORS = {
    "Am": "#4E79A7",
    "Co60": "#F28E2B",
    "Sr": "#59A14F",
}

FEATURE_HISTOGRAMS = [
    ("active_pixel_count", "Active pixels", "log10(active pixels + 1)", True),
    ("total_ToT", "Total ToT", "log10(total ToT + 1)", True),
    ("mean_ToT_nonzero", "Mean nonzero ToT", "log10(mean ToT + 1)", True),
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
        "tot_path",
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
    df = df.dropna(subset=numeric_cols + ["particle", "tot_path"]).copy()
    df["bbox_aspect_ratio"] = df["bbox_long"] / df["bbox_short"].clip(lower=1)
    df["log_active"] = np.log10(df["active_pixel_count"].clip(lower=0) + 1.0)
    df["log_total_ToT"] = np.log10(df["total_ToT"].clip(lower=0) + 1.0)
    df["log_mean_ToT"] = np.log10(df["mean_ToT_nonzero"].clip(lower=0) + 1.0)
    return df


def transformed(values: pd.Series, log_transform: bool) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    if log_transform:
        arr = np.log10(np.clip(arr, 0, None) + 1.0)
    return arr[np.isfinite(arr)]


def write_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_rows: list[dict[str, object]] = []
    for particle in PARTICLE_ORDER:
        sub = df[df["particle"] == particle]
        if sub.empty:
            continue
        for col in [
            "active_pixel_count",
            "total_ToT",
            "mean_ToT_nonzero",
            "max_ToT",
            "bbox_aspect_ratio",
        ]:
            values = sub[col].to_numpy(dtype=float)
            feature_rows.append(
                {
                    "particle": particle,
                    "feature": col,
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "q01": float(np.quantile(values, 0.01)),
                    "q05": float(np.quantile(values, 0.05)),
                    "q25": float(np.quantile(values, 0.25)),
                    "median": float(np.quantile(values, 0.50)),
                    "q75": float(np.quantile(values, 0.75)),
                    "q95": float(np.quantile(values, 0.95)),
                    "q99": float(np.quantile(values, 0.99)),
                }
            )
    pd.DataFrame(feature_rows).to_csv(output_dir / "stage1_cleaning_feature_quantiles.csv", index=False)

    subdir_rows = []
    grouped = df.groupby(["particle", "source_subdir"], dropna=False)
    for (particle, source_subdir), sub in grouped:
        subdir_rows.append(
            {
                "particle": particle,
                "source_subdir": source_subdir,
                "candidate_count": int(len(sub)),
                "median_active_pixel_count": float(sub["active_pixel_count"].median()),
                "median_total_ToT": float(sub["total_ToT"].median()),
                "median_mean_ToT_nonzero": float(sub["mean_ToT_nonzero"].median()),
                "median_bbox_aspect_ratio": float(sub["bbox_aspect_ratio"].median()),
            }
        )
    pd.DataFrame(subdir_rows).sort_values(["particle", "source_subdir"]).to_csv(
        output_dir / "stage1_cleaning_by_source_subdir.csv",
        index=False,
    )


def plot_active_total_hexbin_count(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharex=True, sharey=True, constrained_layout=True)
    xlim = np.nanpercentile(df["log_active"], [0.2, 99.8])
    ylim = np.nanpercentile(df["log_total_ToT"], [0.2, 99.8])

    for ax, particle in zip(axes, PARTICLE_ORDER):
        sub = df[df["particle"] == particle]
        if sub.empty:
            ax.axis("off")
            continue
        hb = ax.hexbin(
            sub["log_active"],
            sub["log_total_ToT"],
            gridsize=55,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            mincnt=1,
            bins="log",
            cmap="mako" if "mako" in plt.colormaps() else "viridis",
            linewidths=0,
        )
        med_x = float(sub["log_active"].median())
        med_y = float(sub["log_total_ToT"].median())
        ax.axvline(med_x, color="white", linewidth=0.9, alpha=0.8)
        ax.axhline(med_y, color="white", linewidth=0.9, alpha=0.8)
        ax.scatter([med_x], [med_y], s=16, color="white", edgecolor="black", linewidth=0.4, zorder=3)
        ax.set_title(f"{particle} candidates (n={len(sub)})")
        ax.set_xlabel("log10(active pixels + 1)")
        ax.grid(color="white", linewidth=0.3, alpha=0.35)
    axes[0].set_ylabel("log10(total ToT + 1)")
    cbar = fig.colorbar(hb, ax=axes, shrink=0.82, pad=0.015)
    cbar.set_label("Candidate count per hexbin (log scale)")
    save_figure(fig, output_dir / "stage1_cleaning_active_total_hexbin_count")
    plt.close(fig)


def plot_feature_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.0), constrained_layout=True)

    for row_idx, particle in enumerate(PARTICLE_ORDER):
        sub = df[df["particle"] == particle]
        for col_idx, (col, title, xlabel, log_transform) in enumerate(FEATURE_HISTOGRAMS):
            ax = axes[row_idx, col_idx]
            values = transformed(sub[col], log_transform)
            lo, hi = np.nanpercentile(values, [0.5, 99.5])
            bins = np.linspace(lo, hi, 50)
            ax.hist(
                values,
                bins=bins,
                color=PARTICLE_COLORS.get(particle, "0.3"),
                alpha=0.42,
                edgecolor=PARTICLE_COLORS.get(particle, "0.3"),
                linewidth=0.8,
            )
            q05, q50, q95 = np.quantile(values, [0.05, 0.50, 0.95])
            for q, linestyle, label in [(q05, "--", "P5"), (q50, "-", "P50"), (q95, "--", "P95")]:
                ax.axvline(q, color="black", linestyle=linestyle, linewidth=0.75, alpha=0.75)
                if row_idx == 0:
                    ax.text(q, ax.get_ylim()[1] * 0.92, label, rotation=90, va="top", ha="right", fontsize=5.5)
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(f"{particle}\nCandidate count")
            else:
                ax.set_ylabel("Candidate count")
            if row_idx == 2:
                ax.set_xlabel(xlabel)
            ax.grid(axis="y", color="0.9", linewidth=0.5)

    save_figure(fig, output_dir / "stage1_cleaning_feature_histograms_count")
    plt.close(fig)


def plot_source_subdir_counts(df: pd.DataFrame, output_dir: Path) -> None:
    count_df = (
        df.groupby(["particle", "source_subdir"], dropna=False)
        .size()
        .reset_index(name="candidate_count")
        .sort_values(["particle", "candidate_count"], ascending=[True, False])
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    for ax, particle in zip(axes, PARTICLE_ORDER):
        sub = count_df[count_df["particle"] == particle].head(20)
        if sub.empty:
            ax.axis("off")
            continue
        y = np.arange(len(sub))
        ax.barh(y, sub["candidate_count"], color=PARTICLE_COLORS.get(particle, "0.3"), alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["source_subdir"].astype(str), fontsize=5.5)
        ax.invert_yaxis()
        ax.set_title(f"{particle}: top source folders")
        ax.set_xlabel("Candidate count")
        ax.grid(axis="x", color="0.9", linewidth=0.5)
    save_figure(fig, output_dir / "stage1_cleaning_source_subdir_counts")
    plt.close(fig)


def choose_representatives(sub: pd.DataFrame) -> pd.DataFrame:
    z_active = (sub["log_active"] - sub["log_active"].median()) / (sub["log_active"].std(ddof=0) + 1e-9)
    z_total = (sub["log_total_ToT"] - sub["log_total_ToT"].median()) / (sub["log_total_ToT"].std(ddof=0) + 1e-9)
    main_idx = (z_active.pow(2) + z_total.pow(2)).idxmin()
    low_active_idx = sub["active_pixel_count"].idxmin()
    low_total_idx = sub["total_ToT"].idxmin()
    high_total_idx = sub["total_ToT"].idxmax()
    rows = sub.loc[[main_idx, low_active_idx, low_total_idx, high_total_idx]].copy()
    rows["representative_type"] = ["main cluster", "low active", "low total", "high total"]
    return rows


def plot_representative_tot_plate(df: pd.DataFrame, stage1_root: Path, output_dir: Path) -> None:
    reps = []
    for particle in PARTICLE_ORDER:
        sub = df[df["particle"] == particle]
        if not sub.empty:
            reps.append(choose_representatives(sub))
    if not reps:
        return
    rep_df = pd.concat(reps, ignore_index=True)
    fig, axes = plt.subplots(len(PARTICLE_ORDER), 4, figsize=(6.4, 4.8), constrained_layout=True)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for row_idx, particle in enumerate(PARTICLE_ORDER):
        sub = rep_df[rep_df["particle"] == particle]
        for col_idx, rep_type in enumerate(["main cluster", "low active", "low total", "high total"]):
            ax = axes[row_idx, col_idx]
            row = sub[sub["representative_type"] == rep_type]
            if row.empty:
                ax.axis("off")
                continue
            item = row.iloc[0]
            matrix_path = stage1_root / "dataset" / str(item["tot_path"])
            matrix = np.loadtxt(matrix_path, dtype=np.float32)
            image = np.log1p(matrix)
            ax.imshow(image, cmap="magma", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(rep_type)
            if col_idx == 0:
                ax.set_ylabel(particle, rotation=0, labelpad=14, va="center")
            ax.text(
                0.02,
                0.98,
                f"pix={int(item['active_pixel_count'])}\nToT={item['total_ToT']:.0f}",
                transform=ax.transAxes,
                color="white",
                va="top",
                ha="left",
                fontsize=5.5,
            )
    save_figure(fig, output_dir / "stage1_cleaning_representative_tot_plate")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stage-1 particle cleaning diagnostics.")
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
        help="Output directory. Default: <stage1-root>/figures/cleaning_diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    stage1_root = args.stage1_root
    output_dir = args.output_dir or (stage1_root / "figures" / "cleaning_diagnostics")
    df = load_manifest(stage1_root)
    write_summary_tables(df, output_dir)
    plot_active_total_hexbin_count(df, output_dir)
    plot_feature_histograms(df, output_dir)
    plot_source_subdir_counts(df, output_dir)
    plot_representative_tot_plate(df, stage1_root, output_dir)
    print(f"Loaded candidates: {len(df)}")
    print("By particle:")
    for particle, count in df["particle"].value_counts().sort_index().items():
        print(f"  {particle}: {count}")
    print(f"Cleaning diagnostics written to: {output_dir}")


if __name__ == "__main__":
    main()
