"""Stage-2b particle-wise feature transformation and clustering diagnostics.

This script consumes Stage-2a raw ToT/morphology features and performs
transform/scaling plus unsupervised clustering independently inside each
particle source. It intentionally does not mix Am, Co60, and Sr in the same
clustering fit.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.mixture import GaussianMixture


DEFAULT_STAGE2A_ROOT = Path(r"E:\TimepixData\particle\stage2_cluster_features_v1")
DEFAULT_OUTPUT_ROOT = Path(r"E:\TimepixData\particle\stage2b_particlewise_clustering_v1")

FEATURE_COLUMNS = ("Npix", "S_total_ToT", "Pmax", "Rg", "E_pca", "Fbox")
LOG1P_COLUMNS = {"Npix", "S_total_ToT", "Rg"}
EPCA_COLUMN = "E_pca"


@dataclass(frozen=True)
class RobustParameter:
    particle: str
    feature: str
    transform: str
    median: float
    iqr: float
    scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2a-root", type=Path, default=DEFAULT_STAGE2A_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=50)
    parser.add_argument("--hdbscan-min-samples", type=int, default=10)
    parser.add_argument("--scatter-sample-size", type=int, default=25000)
    parser.add_argument("--skip-hdbscan", action="store_true")
    return parser.parse_args()


def feature_transform_name(feature: str) -> str:
    if feature in LOG1P_COLUMNS:
        return "log1p"
    if feature == EPCA_COLUMN:
        return "log1p_minus_one"
    return "identity"


def transform_feature(values: pd.Series, feature: str) -> pd.Series:
    clean = values.astype(float).clip(lower=0)
    if feature in LOG1P_COLUMNS:
        return np.log1p(clean)
    if feature == EPCA_COLUMN:
        return np.log1p((clean - 1.0).clip(lower=0))
    return clean


def robust_scale(values: pd.Series) -> tuple[pd.Series, float, float, float]:
    median = float(values.median())
    q25 = float(values.quantile(0.25))
    q75 = float(values.quantile(0.75))
    iqr = q75 - q25
    scale = iqr if iqr > 1e-12 else 1.0
    return (values - median) / scale, median, iqr, scale


def build_particle_scaled_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[RobustParameter]]]:
    result = df.copy()
    params: dict[str, list[RobustParameter]] = {}

    for feature in FEATURE_COLUMNS:
        result[f"trans_{feature}"] = transform_feature(result[feature], feature)

    for particle, index in result.groupby("particle").groups.items():
        particle_params: list[RobustParameter] = []
        for feature in FEATURE_COLUMNS:
            scaled, median, iqr, scale = robust_scale(result.loc[index, f"trans_{feature}"])
            result.loc[index, f"scaled_{feature}"] = scaled
            particle_params.append(
                RobustParameter(
                    particle=str(particle),
                    feature=feature,
                    transform=feature_transform_name(feature),
                    median=median,
                    iqr=iqr,
                    scale=scale,
                )
            )
        params[str(particle)] = particle_params

    return result, params


def parameter_frame(params: dict[str, list[RobustParameter]]) -> pd.DataFrame:
    rows = []
    for particle_params in params.values():
        for item in particle_params:
            rows.append(
                {
                    "particle": item.particle,
                    "feature": item.feature,
                    "transform": item.transform,
                    "median": item.median,
                    "iqr": item.iqr,
                    "scale": item.scale,
                }
            )
    return pd.DataFrame(rows)


def scaled_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [f"scaled_{feature}" for feature in FEATURE_COLUMNS]
    return df.loc[:, cols].to_numpy(dtype=float)


def select_gmm_components(
    x: np.ndarray,
    random_state: int,
    component_counts: Iterable[int] = (1, 2, 3),
) -> pd.DataFrame:
    rows = []
    n_samples = len(x)
    for n_components in component_counts:
        if n_samples <= n_components:
            rows.append(
                {
                    "n_components": n_components,
                    "bic": np.nan,
                    "aic": np.nan,
                    "converged": False,
                    "lower_bound": np.nan,
                }
            )
            continue
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=1e-6,
            n_init=5,
            max_iter=500,
            random_state=random_state,
        )
        model.fit(x)
        rows.append(
            {
                "n_components": n_components,
                "bic": float(model.bic(x)),
                "aic": float(model.aic(x)),
                "converged": bool(model.converged_),
                "lower_bound": float(model.lower_bound_),
            }
        )
    return pd.DataFrame(rows)


def fit_gmm_assignments(x: np.ndarray, n_components: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=1e-6,
        n_init=5,
        max_iter=500,
        random_state=random_state,
    )
    probs = model.fit_predict(x)
    posterior = model.predict_proba(x)
    confidence = posterior.max(axis=1)
    labels = probs.astype(int)
    return labels, confidence


def fit_hdbscan_assignments(
    x: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        allow_single_cluster=False,
        copy=True,
    )
    model.fit(x)
    labels = model.labels_.astype(int)
    probabilities = model.probabilities_.astype(float)
    return labels, probabilities


def summarize_cluster_labels(
    df: pd.DataFrame,
    particle: str,
    label_column: str,
    probability_column: str | None,
) -> pd.DataFrame:
    rows = []
    total = len(df)
    for label, sub in df.groupby(label_column):
        row = {
            "particle": particle,
            "cluster_source": label_column,
            "cluster_label": int(label),
            "n": int(len(sub)),
            "fraction": float(len(sub) / total) if total else 0.0,
        }
        if probability_column is not None:
            row["confidence_median"] = float(sub[probability_column].median())
            row["confidence_p10"] = float(sub[probability_column].quantile(0.10))
        for feature in FEATURE_COLUMNS:
            row[f"{feature}_median"] = float(sub[feature].median())
            row[f"{feature}_p10"] = float(sub[feature].quantile(0.10))
            row[f"{feature}_p90"] = float(sub[feature].quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows)


def write_gmm_bic_plot(gmm_summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for particle, sub in gmm_summary.groupby("particle"):
        ax.plot(sub["n_components"], sub["bic"], marker="o", label=particle)
    ax.set_xlabel("GMM components")
    ax.set_ylabel("BIC")
    ax.set_title("Particle-wise GMM BIC")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"))
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def write_particle_scatter_plots(
    df: pd.DataFrame,
    particle: str,
    output_dir: Path,
    sample_size: int,
    random_state: int,
) -> None:
    rng = np.random.default_rng(random_state)
    if len(df) > sample_size:
        chosen = rng.choice(df.index.to_numpy(), size=sample_size, replace=False)
        plot_df = df.loc[chosen].copy()
    else:
        plot_df = df.copy()

    plot_specs = [
        ("gmm_k2_label", "GMM k=2", "scaled_Npix", "scaled_Pmax"),
        ("gmm_k3_label", "GMM k=3", "scaled_Npix", "scaled_Pmax"),
        ("hdbscan_label", "HDBSCAN", "scaled_Npix", "scaled_Pmax"),
        ("gmm_k2_label", "GMM k=2", "scaled_Rg", "scaled_E_pca"),
        ("hdbscan_label", "HDBSCAN", "scaled_Rg", "scaled_E_pca"),
    ]
    for label_col, title, x_col, y_col in plot_specs:
        if label_col not in plot_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        labels = plot_df[label_col].astype(int).to_numpy()
        scatter = ax.scatter(
            plot_df[x_col],
            plot_df[y_col],
            c=labels,
            s=5,
            alpha=0.45,
            cmap="tab10",
            linewidths=0,
        )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{particle} {title}")
        ax.grid(alpha=0.25)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("cluster label")
        fig.tight_layout()
        name = f"{particle}_{label_col}_{x_col}_vs_{y_col}"
        fig.savefig(output_dir / f"{name}.png")
        fig.savefig(output_dir / f"{name}.pdf")
        plt.close(fig)


def run_particle_clustering(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clustered_parts = []
    gmm_summary_parts = []
    particle_summary_rows = []
    cluster_feature_parts = []

    for particle, sub in df.groupby("particle", sort=True):
        sub = sub.copy()
        x = scaled_matrix(sub)

        gmm_summary = select_gmm_components(x, random_state=args.random_state)
        gmm_summary.insert(0, "particle", particle)
        gmm_summary_parts.append(gmm_summary)

        best_row = gmm_summary.dropna(subset=["bic"]).sort_values("bic").iloc[0]
        best_k = int(best_row["n_components"])

        for k in (2, 3):
            labels, confidence = fit_gmm_assignments(x, k, random_state=args.random_state)
            sub[f"gmm_k{k}_label"] = labels
            sub[f"gmm_k{k}_confidence"] = confidence
            cluster_feature_parts.append(
                summarize_cluster_labels(sub, particle, f"gmm_k{k}_label", f"gmm_k{k}_confidence")
            )

        if best_k == 1:
            sub["gmm_best_label"] = 0
            sub["gmm_best_confidence"] = 1.0
        else:
            labels, confidence = fit_gmm_assignments(x, best_k, random_state=args.random_state)
            sub["gmm_best_label"] = labels
            sub["gmm_best_confidence"] = confidence
        cluster_feature_parts.append(
            summarize_cluster_labels(sub, particle, "gmm_best_label", "gmm_best_confidence")
        )

        hdbscan_clusters = 0
        hdbscan_noise_count = math.nan
        hdbscan_noise_rate = math.nan
        if args.skip_hdbscan:
            sub["hdbscan_label"] = -999
            sub["hdbscan_probability"] = np.nan
        else:
            labels, probabilities = fit_hdbscan_assignments(
                x,
                min_cluster_size=args.hdbscan_min_cluster_size,
                min_samples=args.hdbscan_min_samples,
            )
            sub["hdbscan_label"] = labels
            sub["hdbscan_probability"] = probabilities
            hdbscan_clusters = len({int(v) for v in labels if int(v) >= 0})
            hdbscan_noise_count = int((labels < 0).sum())
            hdbscan_noise_rate = float(hdbscan_noise_count / len(labels))
            cluster_feature_parts.append(
                summarize_cluster_labels(sub, particle, "hdbscan_label", "hdbscan_probability")
            )

        particle_summary_rows.append(
            {
                "particle": particle,
                "n_samples": int(len(sub)),
                "gmm_best_k_by_bic": best_k,
                "gmm_best_bic": float(best_row["bic"]),
                "gmm_best_aic": float(best_row["aic"]),
                "gmm_k2_high_conf_count": int((sub["gmm_k2_confidence"] >= args.confidence_threshold).sum()),
                "gmm_k2_high_conf_rate": float(
                    (sub["gmm_k2_confidence"] >= args.confidence_threshold).mean()
                ),
                "gmm_k3_high_conf_count": int((sub["gmm_k3_confidence"] >= args.confidence_threshold).sum()),
                "gmm_k3_high_conf_rate": float(
                    (sub["gmm_k3_confidence"] >= args.confidence_threshold).mean()
                ),
                "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
                "hdbscan_min_samples": args.hdbscan_min_samples,
                "hdbscan_clusters": hdbscan_clusters,
                "hdbscan_noise_count": hdbscan_noise_count,
                "hdbscan_noise_rate": hdbscan_noise_rate,
            }
        )

        clustered_parts.append(sub)

    return (
        pd.concat(clustered_parts, ignore_index=True),
        pd.concat(gmm_summary_parts, ignore_index=True),
        pd.DataFrame(particle_summary_rows),
        pd.concat(cluster_feature_parts, ignore_index=True),
    )


def write_notes(args: argparse.Namespace, output_root: Path, particle_summary: pd.DataFrame) -> None:
    payload = {
        "stage": "stage2b_particlewise_clustering_v1",
        "stage2a_root": str(args.stage2a_root),
        "output_root": str(output_root),
        "feature_columns": list(FEATURE_COLUMNS),
        "transforms": {feature: feature_transform_name(feature) for feature in FEATURE_COLUMNS},
        "scaling": "robust per particle: (transformed_value - particle_median) / particle_IQR",
        "clustering_scope": "Each particle is transformed, scaled, and clustered independently.",
        "gmm_components_evaluated": [1, 2, 3],
        "gmm_confidence_threshold": args.confidence_threshold,
        "hdbscan": {
            "skipped": bool(args.skip_hdbscan),
            "min_cluster_size": args.hdbscan_min_cluster_size,
            "min_samples": args.hdbscan_min_samples,
        },
        "particle_summary": particle_summary.to_dict(orient="records"),
    }
    (output_root / "stage2b_notes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_root: Path = args.output_root
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    feature_path = args.stage2a_root / "features_raw.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing Stage-2a feature table: {feature_path}")

    df = pd.read_csv(feature_path)
    missing = [feature for feature in FEATURE_COLUMNS if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    scaled_df, params = build_particle_scaled_features(df)
    parameter_frame(params).to_csv(output_root / "transform_parameters.csv", index=False)

    clustered, gmm_summary, particle_summary, cluster_feature_summary = run_particle_clustering(scaled_df, args)

    clustered.to_csv(output_root / "features_transformed_clustered.csv", index=False)
    gmm_summary.to_csv(output_root / "gmm_model_selection.csv", index=False)
    particle_summary.to_csv(output_root / "particle_cluster_summary.csv", index=False)
    cluster_feature_summary.to_csv(output_root / "cluster_feature_summary.csv", index=False)

    write_gmm_bic_plot(gmm_summary, figures_dir / "stage2b_gmm_bic_by_particle")
    for particle, sub in clustered.groupby("particle", sort=True):
        write_particle_scatter_plots(
            sub,
            particle=str(particle),
            output_dir=figures_dir,
            sample_size=args.scatter_sample_size,
            random_state=args.random_state,
        )

    write_notes(args, output_root, particle_summary)

    print(f"Particle-wise clustering finished for {len(clustered)} candidates")
    print(f"Output root: {output_root}")
    print("Important: Am, Co60, and Sr were clustered independently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
