#!/usr/bin/env python3
"""Visual-first particle-wise cluster inspection.

This stage is intentionally not a labeling stage. It draws density plots and
PCA/KMeans reference plots per particle source so that we can inspect whether
source-internal structures have visible boundaries before deciding any cleaning
rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


DEFAULT_STAGE2B_ROOT = Path(r"E:\TimepixData\particle\stage2b_particlewise_clustering_v1")
DEFAULT_OUTPUT_ROOT = Path(r"E:\TimepixData\particle\stage2d_visual_cluster_inspection_v1")

RAW_FEATURE_COLUMNS = ("Npix", "S_total_ToT", "Pmax", "Rg", "E_pca", "Fbox")
SCALED_FEATURE_COLUMNS = tuple(f"scaled_{name}" for name in RAW_FEATURE_COLUMNS)
METADATA_COLUMNS = ("sample_key", "particle", "condition_label", "source_subdir", "angle", "raw_pair_key", "tot_path", "toa_path")


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.linewidth": 0.8,
            "figure.dpi": 120,
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")


def sample_for_plot(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df.copy()
    return df.sample(n=sample_size, random_state=seed).copy()


def compute_pca_table(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    x = df.loc[:, SCALED_FEATURE_COLUMNS].to_numpy(dtype=float)
    pca = PCA(n_components=3, random_state=0)
    scores = pd.DataFrame(pca.fit_transform(x), columns=["PC1", "PC2", "PC3"], index=df.index)
    loadings = pd.DataFrame(
        {
            "feature": list(SCALED_FEATURE_COLUMNS),
            "PC1_loading": pca.components_[0],
            "PC2_loading": pca.components_[1],
            "PC3_loading": pca.components_[2],
        }
    )
    return scores, pca.explained_variance_ratio_, loadings


def add_derived_visual_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["log1p_Npix"] = np.log1p(result["Npix"].clip(lower=0))
    result["log1p_S_total_ToT"] = np.log1p(result["S_total_ToT"].clip(lower=0))
    result["log1p_Rg"] = np.log1p(result["Rg"].clip(lower=0))
    result["log1p_E_pca_minus_1"] = np.log1p((result["E_pca"] - 1.0).clip(lower=0))
    return result


def plot_hexbin(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_base: Path,
    gridsize: int = 80,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    hb = ax.hexbin(df[x_col], df[y_col], gridsize=gridsize, bins="log", cmap="viridis", mincnt=1)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("candidate count per bin (log color)")
    fig.tight_layout()
    save_figure(fig, output_base)
    plt.close(fig)


def plot_scatter_by_label(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    label_prefix: str,
    title: str,
    output_base: Path,
    sample_size: int,
    seed: int,
) -> None:
    plot_df = sample_for_plot(df, sample_size=sample_size, seed=seed)
    labels = sorted(plot_df[label_col].unique())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    for idx, label in enumerate(labels):
        sub = plot_df[plot_df[label_col] == label]
        ax.scatter(sub[x_col], sub[y_col], s=5, alpha=0.45, color=cmap(idx), label=f"{label_prefix} {label}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    save_figure(fig, output_base)
    plt.close(fig)


def fit_kmeans_reference(scores: pd.DataFrame, k: int, seed: int) -> np.ndarray:
    model = KMeans(n_clusters=k, n_init=20, random_state=seed)
    return model.fit_predict(scores[["PC1", "PC2"]].to_numpy(dtype=float))


def fit_gmm_reference(
    scores: pd.DataFrame,
    k: int,
    seed: int,
    columns: tuple[str, ...] = ("PC1", "PC2"),
) -> tuple[np.ndarray, np.ndarray]:
    model = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-6, n_init=5, random_state=seed)
    x = scores.loc[:, list(columns)].to_numpy(dtype=float)
    labels = model.fit_predict(x)
    probabilities = model.predict_proba(x)
    confidence = probabilities.max(axis=1)
    return labels.astype(int), confidence.astype(float)


def make_particle_pca_table(sub: pd.DataFrame, scores: pd.DataFrame, seed: int) -> pd.DataFrame:
    columns = [col for col in METADATA_COLUMNS if col in sub.columns]
    columns.extend([col for col in RAW_FEATURE_COLUMNS if col in sub.columns])
    particle_pca = pd.concat([sub[columns].reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    for k in (2, 3):
        particle_pca[f"kmeans_k{k}_reference"] = fit_kmeans_reference(scores, k=k, seed=seed)
        labels, confidence = fit_gmm_reference(scores, k=k, seed=seed)
        particle_pca[f"gmm_pca_k{k}_reference"] = labels
        particle_pca[f"gmm_pca_k{k}_confidence"] = confidence
    labels, confidence = fit_gmm_reference(scores, k=3, seed=seed, columns=("PC1", "PC2", "PC3"))
    particle_pca["gmm_pca3_k3_reference"] = labels
    particle_pca["gmm_pca3_k3_confidence"] = confidence
    return particle_pca


def angle_sort_key(value: object) -> tuple[int, str]:
    if pd.isna(value):
        return (1, "missing")
    try:
        return (0, f"{float(value):08.3f}")
    except (TypeError, ValueError):
        return (0, str(value))


def angle_label(value: object) -> str:
    if pd.isna(value):
        return "angle missing"
    try:
        number = float(value)
        if number.is_integer():
            return f"{int(number)} deg"
        return f"{number:g} deg"
    except (TypeError, ValueError):
        return str(value)


def plot_angle_hexbin_grid(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_base: Path,
    gridsize: int = 35,
) -> None:
    if "angle" not in df.columns or df["angle"].nunique(dropna=False) <= 1:
        return
    grouped = [(angle, sub) for angle, sub in df.groupby("angle", dropna=False)]
    grouped.sort(key=lambda item: angle_sort_key(item[0]))
    n = len(grouped)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), dpi=150, squeeze=False)
    xlim = (float(df[x_col].min()), float(df[x_col].max()))
    ylim = (float(df[y_col].min()), float(df[y_col].max()))
    last_hb = None
    for ax, (angle, sub) in zip(axes.ravel(), grouped):
        last_hb = ax.hexbin(sub[x_col], sub[y_col], gridsize=gridsize, bins="log", cmap="viridis", mincnt=1)
        ax.set_title(f"{angle_label(angle)} (n={len(sub)})")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.2)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if last_hb is not None:
        cbar = fig.colorbar(last_hb, ax=axes.ravel().tolist(), shrink=0.82)
        cbar.set_label("candidate count per bin (log color)")
    fig.suptitle(title)
    fig.subplots_adjust(top=0.86, right=0.88, hspace=0.45, wspace=0.35)
    save_figure(fig, output_base)
    plt.close(fig)


def plot_angle_scatter_by_label_grid(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    label_prefix: str,
    title: str,
    output_base: Path,
    sample_size: int,
    seed: int,
) -> None:
    if "angle" not in df.columns or df["angle"].nunique(dropna=False) <= 1:
        return
    plot_df = sample_for_plot(df, sample_size=sample_size, seed=seed)
    grouped = [(angle, sub) for angle, sub in plot_df.groupby("angle", dropna=False)]
    grouped.sort(key=lambda item: angle_sort_key(item[0]))
    n = len(grouped)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), dpi=150, squeeze=False)
    cmap = plt.get_cmap("tab10")
    labels = sorted(plot_df[label_col].dropna().unique())
    xlim = (float(df[x_col].min()), float(df[x_col].max()))
    ylim = (float(df[y_col].min()), float(df[y_col].max()))
    for ax, (angle, sub) in zip(axes.ravel(), grouped):
        for idx, label in enumerate(labels):
            label_sub = sub[sub[label_col] == label]
            if label_sub.empty:
                continue
            ax.scatter(label_sub[x_col], label_sub[y_col], s=5, alpha=0.5, color=cmap(idx), label=f"{label_prefix} {label}")
        ax.set_title(f"{angle_label(angle)} (n={len(sub)})")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.2)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper right", frameon=False, markerscale=2)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.86, right=0.88, hspace=0.45, wspace=0.35)
    save_figure(fig, output_base)
    plt.close(fig)


def plot_3d_scatter_by_label(
    df: pd.DataFrame,
    label_col: str,
    label_prefix: str,
    title: str,
    output_base: Path,
    sample_size: int,
    seed: int,
) -> None:
    plot_df = sample_for_plot(df, sample_size=sample_size, seed=seed)
    labels = sorted(plot_df[label_col].dropna().unique())
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(7, 5.6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    for idx, label in enumerate(labels):
        sub = plot_df[plot_df[label_col] == label]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            sub["PC3"],
            s=5,
            alpha=0.45,
            color=cmap(idx),
            label=f"{label_prefix} {label}",
            depthshade=False,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.view_init(elev=22, azim=-58)
    ax.legend(frameon=False, markerscale=2, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_base)
    plt.close(fig)


def build_interactive_hover_text(df: pd.DataFrame) -> list[str]:
    texts: list[str] = []
    for row in df.itertuples(index=False):
        get = row._asdict()
        texts.append(
            "<br>".join(
                [
                    f"sample: {get.get('sample_key', '')}",
                    f"particle: {get.get('particle', '')}",
                    f"angle: {get.get('angle', '')}",
                    f"Npix: {get.get('Npix', '')}",
                    f"S_total_ToT: {float(get.get('S_total_ToT', 0.0)):.3f}",
                    f"Pmax: {float(get.get('Pmax', 0.0)):.3f}",
                    f"Rg: {float(get.get('Rg', 0.0)):.3f}",
                    f"E_pca: {float(get.get('E_pca', 0.0)):.3f}",
                    f"Fbox: {float(get.get('Fbox', 0.0)):.3f}",
                    f"GMM3D: {get.get('gmm_pca3_k3_reference', '')}",
                    f"GMM3D confidence: {float(get.get('gmm_pca3_k3_confidence', 0.0)):.3f}",
                ]
            )
        )
    return texts


def write_interactive_3d_html(
    df: pd.DataFrame,
    output_path: Path,
    sample_size: int,
    seed: int,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for --interactive-html. Install it in timepix-local with: "
            "D:\\Program\\Anaconda\\envs\\timepix-local\\python.exe -m pip install plotly"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = sample_for_plot(df, sample_size=sample_size, seed=seed)
    labels = sorted(plot_df["gmm_pca3_k3_reference"].dropna().unique())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig = go.Figure()
    for idx, label in enumerate(labels):
        sub = plot_df[plot_df["gmm_pca3_k3_reference"] == label]
        fig.add_trace(
            go.Scatter3d(
                x=sub["PC1"],
                y=sub["PC2"],
                z=sub["PC3"],
                mode="markers",
                name=f"GMM3D {label}",
                text=build_interactive_hover_text(sub),
                hoverinfo="text",
                marker={
                    "size": 3,
                    "opacity": 0.55,
                    "color": colors[idx % len(colors)],
                },
            )
        )
    particle = str(plot_df["particle"].iloc[0]) if len(plot_df) else "particle"
    fig.update_layout(
        title=f"{particle}: interactive PCA 3D with GMM k=3 reference colors",
        scene={
            "xaxis_title": "PC1",
            "yaxis_title": "PC2",
            "zaxis_title": "PC3",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)


def run_particle_visuals(
    df: pd.DataFrame,
    output_root: Path,
    sample_size: int,
    seed: int,
    interactive_html: bool,
    interactive_sample_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    figures_dir = output_root / "figures"
    pca_rows: list[pd.DataFrame] = []
    loading_rows: list[pd.DataFrame] = []
    summary_rows = []

    for particle, sub in df.groupby("particle", sort=True):
        sub = add_derived_visual_columns(sub)
        scores, explained, loadings = compute_pca_table(sub)
        particle_pca = make_particle_pca_table(sub, scores, seed=seed)
        pca_rows.append(particle_pca)

        loadings = loadings.copy()
        loadings.insert(0, "particle", particle)
        loadings["PC1_explained_variance"] = float(explained[0])
        loadings["PC2_explained_variance"] = float(explained[1])
        loadings["PC3_explained_variance"] = float(explained[2])
        loading_rows.append(loadings)

        summary_rows.append(
            {
                "particle": particle,
                "n_samples": int(len(sub)),
                "PC1_explained_variance": float(explained[0]),
                "PC2_explained_variance": float(explained[1]),
                "PC3_explained_variance": float(explained[2]),
                "PC1_PC2_total_explained": float(explained[:2].sum()),
                "PC1_PC2_PC3_total_explained": float(explained[:3].sum()),
            }
        )

        plot_hexbin(
            sub,
            "log1p_Npix",
            "Pmax",
            f"{particle}: raw feature density, log1p_Npix vs Pmax",
            figures_dir / f"{particle}_density_log1p_Npix_vs_Pmax",
        )
        plot_hexbin(
            sub,
            "log1p_Rg",
            "log1p_E_pca_minus_1",
            f"{particle}: raw feature density, log1p_Rg vs log1p(E_pca - 1)",
            figures_dir / f"{particle}_density_log1p_Rg_vs_log1p_Epca_minus1",
        )
        plot_hexbin(
            particle_pca,
            "PC1",
            "PC2",
            f"{particle}: PCA density, PC1 vs PC2",
            figures_dir / f"{particle}_pca_density_PC1_vs_PC2",
        )
        for k in (2, 3):
            plot_scatter_by_label(
                particle_pca,
                "PC1",
                "PC2",
                f"kmeans_k{k}_reference",
                "kmeans",
                f"{particle}: PCA with KMeans k={k} reference colors",
                figures_dir / f"{particle}_pca_kmeans_k{k}_reference",
                sample_size=sample_size,
                seed=seed,
            )
            plot_scatter_by_label(
                particle_pca,
                "PC1",
                "PC2",
                f"gmm_pca_k{k}_reference",
                "gmm",
                f"{particle}: PCA with GMM k={k} reference colors",
                figures_dir / f"{particle}_pca_gmm_k{k}_reference",
                sample_size=sample_size,
                seed=seed,
            )
        plot_3d_scatter_by_label(
            particle_pca,
            "gmm_pca3_k3_reference",
            "gmm3d",
            f"{particle}: PCA 3D with GMM k=3 reference colors",
            figures_dir / f"{particle}_pca3_gmm_k3_reference",
            sample_size=sample_size,
            seed=seed,
        )
        if interactive_html:
            write_interactive_3d_html(
                particle_pca,
                output_root / "interactive" / f"{particle}_pca3_gmm_k3_interactive.html",
                sample_size=interactive_sample_size,
                seed=seed,
            )
        plot_angle_hexbin_grid(
            sub,
            "log1p_Npix",
            "Pmax",
            f"{particle}: angle-faceted raw density, log1p_Npix vs Pmax",
            figures_dir / f"{particle}_angle_density_log1p_Npix_vs_Pmax",
        )
        plot_angle_hexbin_grid(
            sub,
            "log1p_Rg",
            "log1p_E_pca_minus_1",
            f"{particle}: angle-faceted raw density, log1p_Rg vs log1p(E_pca - 1)",
            figures_dir / f"{particle}_angle_density_log1p_Rg_vs_log1p_Epca_minus1",
        )
        plot_angle_hexbin_grid(
            particle_pca,
            "PC1",
            "PC2",
            f"{particle}: angle-faceted PCA density, PC1 vs PC2",
            figures_dir / f"{particle}_angle_pca_density_PC1_vs_PC2",
        )
        plot_angle_scatter_by_label_grid(
            particle_pca,
            "PC1",
            "PC2",
            "kmeans_k3_reference",
            "kmeans",
            f"{particle}: angle-faceted PCA with KMeans k=3 reference colors",
            figures_dir / f"{particle}_angle_pca_kmeans_k3_reference",
            sample_size=sample_size,
            seed=seed,
        )
        plot_angle_scatter_by_label_grid(
            particle_pca,
            "PC1",
            "PC2",
            "gmm_pca_k3_reference",
            "gmm",
            f"{particle}: angle-faceted PCA with GMM k=3 reference colors",
            figures_dir / f"{particle}_angle_pca_gmm_k3_reference",
            sample_size=sample_size,
            seed=seed,
        )

    pca_table = pd.concat(pca_rows, ignore_index=True)
    loading_table = pd.concat(loading_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    return pca_table, pd.concat([loading_table, summary], ignore_index=True, sort=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2b-root", type=Path, default=DEFAULT_STAGE2B_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-size", type=int, default=30000)
    parser.add_argument("--interactive-html", action="store_true", help="Write draggable Plotly 3D HTML views.")
    parser.add_argument("--interactive-sample-size", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_style()
    args.output_root.mkdir(parents=True, exist_ok=True)

    input_path = args.stage2b_root / "features_transformed_clustered.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing Stage-2b table: {input_path}")
    df = pd.read_csv(input_path)
    missing = [col for col in SCALED_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing scaled columns: {missing}")

    pca_table, pca_aux = run_particle_visuals(
        df,
        args.output_root,
        args.sample_size,
        args.seed,
        args.interactive_html,
        args.interactive_sample_size,
    )
    pca_table.to_csv(args.output_root / "pca_scores_with_kmeans_reference.csv", index=False)

    loadings = pca_aux[pca_aux["feature"].notna()].copy()
    summary = pca_aux[pca_aux["feature"].isna()].drop(
        columns=["feature", "PC1_loading", "PC2_loading", "PC3_loading"],
        errors="ignore",
    )
    loadings.to_csv(args.output_root / "pca_loadings.csv", index=False)
    summary.to_csv(args.output_root / "pca_summary.csv", index=False)

    notes = {
        "stage": "stage2d_visual_cluster_inspection_v1",
        "goal": "Visual inspection only; KMeans labels are reference colors, not final training labels.",
        "input": str(input_path),
        "output_root": str(args.output_root),
        "plots": {
            "density_*": "hexbin density plots; colorbar is candidate count per bin.",
            "pca_kmeans_*": "PCA scatter with KMeans reference colors for visual boundary inspection.",
            "pca_gmm_*": "PCA scatter with GMM reference colors and confidence values.",
            "pca3_gmm_*": "3D PCA scatter with GMM k=3 fitted in PC1/PC2/PC3 space.",
            "interactive/*.html": "draggable Plotly 3D PCA/GMM views when --interactive-html is set.",
            "angle_*": "angle-faceted plots for checking whether source-internal structures remain visible within the same angle.",
        },
    }
    (args.output_root / "stage2d_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

    print(f"Stage-2d visual inspection written to: {args.output_root}")
    print("KMeans/GMM colors are reference diagnostics only, not final labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
