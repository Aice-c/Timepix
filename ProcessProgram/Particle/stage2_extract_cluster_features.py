#!/usr/bin/env python3
"""Stage-2a raw cluster feature extraction and distribution diagnostics.

This stage intentionally does not transform, standardize, cluster, or clean the
dataset. It computes a small set of ToT/morphology features so the feature
distributions can be inspected before deciding the transformation and clustering
strategy.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_STAGE1_ROOT = Path(r"E:\TimepixData\particle\stage1_single_particle_candidates_100x100")
DEFAULT_OUTPUT_ROOT = Path(r"E:\TimepixData\particle\stage2_cluster_features_v1")

PARTICLE_ORDER = ["Am", "Co60", "Sr"]
PARTICLE_COLORS = {
    "Am": "#4E79A7",
    "Co60": "#F28E2B",
    "Sr": "#59A14F",
}
FEATURE_COLUMNS = ["Npix", "S_total_ToT", "Pmax", "Rg", "E_pca", "Fbox"]
PCA_VARIANCE_REGULARIZER = 0.25


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
    numbers = re.findall(r"\d+", str(source_subdir))
    if not numbers:
        return None
    return int(numbers[-1])


def compute_cluster_features(matrix: np.ndarray) -> dict[str, float]:
    mask = matrix > 0
    coords = np.argwhere(mask)
    npix = int(coords.shape[0])
    if npix == 0:
        return {
            "Npix": 0,
            "S_total_ToT": 0.0,
            "Pmax": 0.0,
            "Rg": 0.0,
            "E_pca": 1.0,
            "Fbox": 0.0,
        }

    values = matrix[mask].astype(float)
    total_tot = float(values.sum())
    pmax = float(values.max() / total_tot) if total_tot > 0 else 0.0

    centroid = coords.astype(float).mean(axis=0)
    centered = coords.astype(float) - centroid
    squared_distance = np.sum(centered * centered, axis=1)
    rg = float(math.sqrt(float(np.mean(squared_distance)))) if npix > 0 else 0.0

    if npix == 1:
        e_pca = 1.0
    else:
        cov = np.cov(coords.astype(float), rowvar=False, bias=True)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        minor, major = float(eigenvalues[0]), float(eigenvalues[-1])
        e_pca = float(
            math.sqrt(major + PCA_VARIANCE_REGULARIZER)
            / math.sqrt(minor + PCA_VARIANCE_REGULARIZER)
        )

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    bbox_area = int((r1 - r0) * (c1 - c0))
    fbox = float(npix / bbox_area) if bbox_area > 0 else 0.0

    return {
        "Npix": npix,
        "S_total_ToT": total_tot,
        "Pmax": pmax,
        "Rg": rg,
        "E_pca": e_pca,
        "Fbox": fbox,
    }


def load_manifest(stage1_root: Path) -> pd.DataFrame:
    manifest_path = stage1_root / "manifests" / "extraction_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage-1 extraction manifest does not exist: {manifest_path}")
    usecols = [
        "sample_key",
        "particle",
        "condition_label",
        "source_subdir",
        "raw_pair_key",
        "active_pixel_count",
        "total_ToT",
        "max_ToT",
        "bbox_long",
        "bbox_short",
        "tot_path",
        "toa_path",
    ]
    df = pd.read_csv(manifest_path, usecols=usecols)
    df["angle"] = df["source_subdir"].map(parse_angle)
    return df


def compute_feature_table(stage1_root: Path, manifest: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    work = manifest.head(limit).copy() if limit else manifest
    for _, row in work.iterrows():
        matrix_path = stage1_root / "dataset" / str(row["tot_path"])
        matrix = np.loadtxt(matrix_path, dtype=np.float32)
        features = compute_cluster_features(matrix)
        out = {
            "sample_key": row["sample_key"],
            "particle": row["particle"],
            "condition_label": row.get("condition_label"),
            "source_subdir": row.get("source_subdir"),
            "angle": row.get("angle"),
            "raw_pair_key": row.get("raw_pair_key"),
            "tot_path": row.get("tot_path"),
            "toa_path": row.get("toa_path"),
            "manifest_active_pixel_count": row.get("active_pixel_count"),
            "manifest_total_ToT": row.get("total_ToT"),
            "manifest_max_ToT": row.get("max_ToT"),
            "manifest_bbox_long": row.get("bbox_long"),
            "manifest_bbox_short": row.get("bbox_short"),
        }
        out.update(features)
        rows.append(out)
    return pd.DataFrame(rows)


def write_summary(features: pd.DataFrame, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for group_name, group_cols in [
        ("particle", ["particle"]),
        ("particle_angle", ["particle", "angle"]),
    ]:
        grouped = features.groupby(group_cols, dropna=False)
        for keys, sub in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            group_values = dict(zip(group_cols, keys))
            for feature in FEATURE_COLUMNS:
                values = sub[feature].to_numpy(dtype=float)
                row = {
                    "group": group_name,
                    **group_values,
                    "feature": feature,
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
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
                summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_root / "feature_summary.csv", index=False)

    pearson = features[FEATURE_COLUMNS].corr(method="pearson")
    spearman = features[FEATURE_COLUMNS].corr(method="spearman")
    pearson.to_csv(output_root / "feature_correlation_pearson.csv")
    spearman.to_csv(output_root / "feature_correlation_spearman.csv")


def feature_label(feature: str) -> str:
    return {
        "Npix": "Npix (active pixels)",
        "S_total_ToT": "S_total_ToT",
        "Pmax": "Pmax = max ToT / total ToT",
        "Rg": "Rg",
        "E_pca": "E_pca",
        "Fbox": "Fbox = active pixels / bbox area",
    }[feature]


def plot_feature_histograms(features: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(
        len(FEATURE_COLUMNS),
        len(PARTICLE_ORDER),
        figsize=(2.3 * len(PARTICLE_ORDER), 1.35 * len(FEATURE_COLUMNS)),
        constrained_layout=True,
    )
    for row_idx, feature in enumerate(FEATURE_COLUMNS):
        values_all = features[feature].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(values_all, [0.5, 99.5])
        if lo == hi:
            lo, hi = float(np.nanmin(values_all)), float(np.nanmax(values_all))
        bins = np.linspace(lo, hi, 60)
        for col_idx, particle in enumerate(PARTICLE_ORDER):
            ax = axes[row_idx, col_idx]
            sub = features[features["particle"] == particle]
            color = PARTICLE_COLORS.get(particle, "0.3")
            ax.hist(sub[feature].to_numpy(dtype=float), bins=bins, color=color, alpha=0.45, edgecolor=color)
            ax.axvline(float(sub[feature].median()), color="black", linewidth=0.7)
            if row_idx == 0:
                ax.set_title(particle)
            if col_idx == 0:
                ax.set_ylabel(f"{feature}\nCandidate count")
            ax.set_xlabel(feature_label(feature))
            ax.grid(axis="y", color="0.9", linewidth=0.5)
    save_figure(fig, figures_dir / "stage2_raw_feature_histograms_by_particle_count")
    plt.close(fig)


def scatter_sample(features: pd.DataFrame, max_points_per_particle: int, seed: int = 42) -> pd.DataFrame:
    sampled = []
    for particle in PARTICLE_ORDER:
        sub = features[features["particle"] == particle]
        if sub.empty:
            continue
        take = min(max_points_per_particle, len(sub))
        sampled.append(sub.sample(n=take, random_state=seed))
    if not sampled:
        return pd.DataFrame()
    return pd.concat(sampled, ignore_index=True)


def plot_scatter_pairs(features: pd.DataFrame, figures_dir: Path) -> None:
    pairs = [
        ("Npix", "Pmax", "Npix vs Pmax"),
        ("Rg", "E_pca", "Rg vs E_pca"),
        ("Npix", "S_total_ToT", "Npix vs S_total_ToT"),
        ("Fbox", "E_pca", "Fbox vs E_pca"),
    ]
    sampled = scatter_sample(features, max_points_per_particle=8000)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    for ax, (x_col, y_col, title) in zip(axes.ravel(), pairs):
        for particle in PARTICLE_ORDER:
            sub = sampled[sampled["particle"] == particle]
            if sub.empty:
                continue
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=3,
                alpha=0.18,
                linewidths=0,
                color=PARTICLE_COLORS.get(particle, "0.3"),
                label=particle,
                rasterized=True,
            )
        ax.set_title(title)
        ax.set_xlabel(feature_label(x_col))
        ax.set_ylabel(feature_label(y_col))
        ax.grid(color="0.9", linewidth=0.5)
    axes[0, 0].legend(markerscale=3)
    save_figure(fig, figures_dir / "stage2_raw_feature_scatter_pairs")
    plt.close(fig)


def plot_feature_histograms_by_angle(features: pd.DataFrame, figures_dir: Path) -> None:
    for particle in PARTICLE_ORDER:
        sub_particle = features[features["particle"] == particle]
        if sub_particle.empty:
            continue
        angles = [int(a) for a in sorted(sub_particle["angle"].dropna().unique())]
        if not angles:
            continue
        fig, axes = plt.subplots(
            len(FEATURE_COLUMNS),
            len(angles),
            figsize=(2.05 * len(angles), 1.25 * len(FEATURE_COLUMNS)),
            constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(len(FEATURE_COLUMNS), len(angles))
        for row_idx, feature in enumerate(FEATURE_COLUMNS):
            values_all = sub_particle[feature].to_numpy(dtype=float)
            lo, hi = np.nanpercentile(values_all, [0.5, 99.5])
            if lo == hi:
                lo, hi = float(np.nanmin(values_all)), float(np.nanmax(values_all))
            bins = np.linspace(lo, hi, 50)
            for col_idx, angle in enumerate(angles):
                ax = axes[row_idx, col_idx]
                sub = sub_particle[sub_particle["angle"] == angle]
                color = PARTICLE_COLORS.get(particle, "0.3")
                ax.hist(sub[feature].to_numpy(dtype=float), bins=bins, color=color, alpha=0.45, edgecolor=color)
                ax.axvline(float(sub[feature].median()), color="black", linewidth=0.7)
                if row_idx == 0:
                    ax.set_title(f"{angle} deg")
                if col_idx == 0:
                    ax.set_ylabel(f"{feature}\nCandidate count")
                ax.set_xlabel(feature_label(feature))
                ax.grid(axis="y", color="0.9", linewidth=0.5)
        save_figure(fig, figures_dir / f"stage2_{particle}_raw_feature_histograms_by_angle_count")
        plt.close(fig)


def write_feature_notes(output_root: Path) -> None:
    notes = """# Stage-2a Feature Notes

This stage computes raw ToT/morphology features only. It does not apply log
transforms, robust scaling, HDBSCAN, GMM, or dataset cleaning.

Initial clustering candidates:

- Npix: active pixel count from ToT > 0.
- S_total_ToT: sum of ToT over active pixels.
- Pmax: max active-pixel ToT divided by S_total_ToT.
- Rg: unweighted radius of gyration of active-pixel coordinates.
- E_pca: regularized PCA elongation of active-pixel coordinates.
- Fbox: active-pixel count divided by bounding-box area.

E_pca uses a 0.25 pixel-variance regularizer so tiny 1-3 pixel clusters stay
finite instead of producing infinite elongation when the PCA minor axis is zero.
"""
    output_root.joinpath("feature_notes.md").write_text(notes, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Stage-2a raw cluster features.")
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    args.output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_root / "figures"

    manifest = load_manifest(args.stage1_root)
    features = compute_feature_table(args.stage1_root, manifest, limit=args.limit)
    features.to_csv(args.output_root / "features_raw.csv", index=False)
    write_summary(features, args.output_root)
    write_feature_notes(args.output_root)
    plot_feature_histograms(features, figures_dir)
    plot_feature_histograms_by_angle(features, figures_dir)
    plot_scatter_pairs(features, figures_dir)

    print(f"Computed raw features for {len(features)} candidates")
    print(f"Output root: {args.output_root}")
    print("No transformations or clustering were applied.")


if __name__ == "__main__":
    main()
