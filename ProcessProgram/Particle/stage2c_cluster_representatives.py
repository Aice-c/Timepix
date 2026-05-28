#!/usr/bin/env python3
"""Draw representative ToT crops from particle-wise GMM clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_STAGE1_ROOT = Path(r"E:\TimepixData\particle\stage1_single_particle_candidates_100x100")
DEFAULT_STAGE2B_ROOT = Path(r"E:\TimepixData\particle\stage2b_particlewise_clustering_v1")
DEFAULT_OUTPUT_ROOT = Path(r"E:\TimepixData\particle\stage2c_cluster_representatives_v1")

FEATURE_COLUMNS = ("Npix", "S_total_ToT", "Pmax", "Rg", "E_pca", "Fbox")


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


def sample_rows_by_particle_cluster(
    df: pd.DataFrame,
    label_column: str,
    confidence_column: str | None,
    samples_per_cluster: int,
    min_confidence: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled_parts: list[pd.DataFrame] = []

    for (particle, label), sub in df.groupby(["particle", label_column], sort=True):
        total_count = int(len(sub))
        if confidence_column and confidence_column in sub.columns:
            high_conf = sub[sub[confidence_column] >= min_confidence].copy()
        else:
            high_conf = sub.copy()
        high_conf_count = int(len(high_conf))
        pool = high_conf if high_conf_count else sub.copy()
        take = min(samples_per_cluster, len(pool))
        random_state = int(rng.integers(0, 2**31 - 1))
        picked = pool.sample(n=take, random_state=random_state).copy()
        picked["cluster_total_count"] = total_count
        picked["cluster_high_conf_count"] = high_conf_count
        picked["sample_source"] = "high_confidence" if high_conf_count else "fallback_all"
        sampled_parts.append(picked)

    if not sampled_parts:
        return pd.DataFrame()
    return pd.concat(sampled_parts, ignore_index=True)


def matrix_path(stage1_root: Path, row: pd.Series) -> Path:
    return stage1_root / "dataset" / str(row["tot_path"])


def cluster_title(row: pd.Series, label_column: str) -> str:
    return (
        f"cluster {int(row[label_column])}\n"
        f"n={int(row['cluster_total_count'])}, high={int(row['cluster_high_conf_count'])}"
    )


def plot_particle_cluster_samples(
    sampled: pd.DataFrame,
    stage1_root: Path,
    output_dir: Path,
    label_column: str,
    confidence_column: str | None,
    samples_per_cluster: int,
    crop_size: int,
) -> None:
    for particle, particle_rows in sampled.groupby("particle", sort=True):
        labels = sorted(particle_rows[label_column].unique())
        fig, axes = plt.subplots(
            len(labels),
            samples_per_cluster,
            figsize=(1.35 * samples_per_cluster, 1.45 * len(labels)),
            constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(len(labels), samples_per_cluster)
        for ax in axes.ravel():
            ax.axis("off")

        canvases: list[tuple[pd.Series, np.ndarray]] = []
        values_for_scale: list[float] = []
        for _, row in particle_rows.iterrows():
            matrix = np.loadtxt(matrix_path(stage1_root, row), dtype=np.float32)
            canvas = center_nonzero_to_canvas(matrix, size=crop_size)
            canvases.append((row, canvas))
            nonzero = canvas[canvas > 0]
            if nonzero.size:
                values_for_scale.extend(np.log1p(nonzero).tolist())
        vmax = float(np.quantile(values_for_scale, 0.98)) if values_for_scale else 1.0

        for row_idx, label in enumerate(labels):
            sub = [(row, canvas) for row, canvas in canvases if int(row[label_column]) == int(label)]
            for col_idx, (row, canvas) in enumerate(sub[:samples_per_cluster]):
                ax = axes[row_idx, col_idx]
                ax.imshow(np.log1p(canvas), cmap="magma", interpolation="nearest", vmin=0, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                confidence_text = ""
                if confidence_column and confidence_column in row:
                    confidence_text = f"\nconf={float(row[confidence_column]):.2f}"
                ax.set_title(
                    f"pix={int(row['Npix'])}, ToT={row['S_total_ToT']:.0f}\n"
                    f"Pmax={row['Pmax']:.2f}, E={row['E_pca']:.1f}{confidence_text}",
                    fontsize=6,
                )
                if col_idx == 0:
                    ax.set_ylabel(cluster_title(row, label_column), rotation=0, labelpad=30, va="center")

        fig.suptitle(f"{particle} representative samples by {label_column} (ToT, {crop_size}x{crop_size})")
        save_figure(fig, output_dir / f"{particle}_{label_column}_tot_samples_{crop_size}x{crop_size}")
        plt.close(fig)


def write_cluster_summary(sampled: pd.DataFrame, label_column: str, output_path: Path) -> None:
    rows = []
    for (particle, label), sub in sampled.groupby(["particle", label_column], sort=True):
        row = {
            "particle": particle,
            "cluster_label": int(label),
            "sampled_count": int(len(sub)),
            "cluster_total_count": int(sub["cluster_total_count"].iloc[0]),
            "cluster_high_conf_count": int(sub["cluster_high_conf_count"].iloc[0]),
            "sample_source": ";".join(sorted(sub["sample_source"].unique())),
        }
        for feature in FEATURE_COLUMNS:
            row[f"sampled_{feature}_median"] = float(sub[feature].median())
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--stage2b-root", type=Path, default=DEFAULT_STAGE2B_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--label-column", default="gmm_k3_label")
    parser.add_argument("--confidence-column", default="gmm_k3_confidence")
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--samples-per-cluster", type=int, default=10)
    parser.add_argument("--crop-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_style()
    args.output_root.mkdir(parents=True, exist_ok=True)

    table_path = args.stage2b_root / "features_transformed_clustered.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing Stage-2b clustered table: {table_path}")
    df = pd.read_csv(table_path)
    if args.label_column not in df.columns:
        raise ValueError(f"Missing label column: {args.label_column}")
    confidence_column = args.confidence_column if args.confidence_column in df.columns else None

    sampled = sample_rows_by_particle_cluster(
        df,
        label_column=args.label_column,
        confidence_column=confidence_column,
        samples_per_cluster=args.samples_per_cluster,
        min_confidence=args.min_confidence,
        seed=args.seed,
    )
    sampled.to_csv(args.output_root / "cluster_sample_manifest.csv", index=False)
    write_cluster_summary(sampled, args.label_column, args.output_root / "cluster_sample_summary.csv")
    plot_particle_cluster_samples(
        sampled,
        stage1_root=args.stage1_root,
        output_dir=args.output_root / "figures",
        label_column=args.label_column,
        confidence_column=confidence_column,
        samples_per_cluster=args.samples_per_cluster,
        crop_size=args.crop_size,
    )

    notes = {
        "stage": "stage2c_cluster_representatives_v1",
        "stage1_root": str(args.stage1_root),
        "stage2b_root": str(args.stage2b_root),
        "output_root": str(args.output_root),
        "label_column": args.label_column,
        "confidence_column": confidence_column,
        "min_confidence": args.min_confidence,
        "samples_per_cluster": args.samples_per_cluster,
        "crop_size": args.crop_size,
        "interpretation": "Representative morphology only; cluster labels are not physical truth labels.",
    }
    (args.output_root / "stage2c_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

    print(f"Sampled {len(sampled)} representative events")
    print(f"Output root: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
