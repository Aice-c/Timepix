#!/usr/bin/env python
"""Analyze C/proton near-vertical angular resolution limits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timepix.analysis.features import extract_feature_table, feature_summary_by_angle
from timepix.analysis.io import class_counts, make_output_layout, scan_dataset, write_manifest
from timepix.analysis.ml import (
    aggregate_ml_results,
    auc_by_gap,
    numeric_feature_columns,
    pairwise_auc_by_gap,
    run_ml_baselines,
)
from timepix.analysis.plotting import (
    plot_adjacent_difference_maps,
    plot_confusion_matrix,
    plot_embedding,
    plot_feature_kde,
    plot_feature_violin,
    plot_heatmap,
    plot_mean_images_by_angle,
    plot_metric_by_gap,
    plot_representative_grid,
)
from timepix.analysis.reports import resolution_report
from timepix.analysis.representative import deterministic_sample, select_representatives
from timepix.analysis.stats import feature_distance_summary, feature_pair_effects, pivot_metric
from timepix.analysis.tables import write_markdown, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze near-vertical C/proton angle separability")
    parser.add_argument("--data-root", default="Data")
    parser.add_argument("--dataset", default="Proton_C")
    parser.add_argument("--angles", nargs="+", type=float, default=[80, 82, 84, 86, 88, 90])
    parser.add_argument("--modality", default="ToT")
    parser.add_argument("--output-root", default="outputs/resolution_limit")
    parser.add_argument("--sample-cap-plot", type=int, default=5000)
    parser.add_argument("--sample-cap-ml", type=int, default=10000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--tsne", action="store_true", help="Run optional t-SNE embedding")
    return parser.parse_args()


def _subset(index_df: pd.DataFrame, dataset: str, modality: str, angles: list[float]) -> pd.DataFrame:
    values = pd.to_numeric(index_df["angle_value"], errors="coerce")
    return index_df[
        (index_df["dataset"] == dataset)
        & (index_df["modality"] == modality)
        & (index_df["status"] == "ok")
        & values.isin([float(a) for a in angles])
    ].copy()


def _embedding_inputs(features: pd.DataFrame, feature_cols: list[str], cap: int, seed: int):
    from sklearn.preprocessing import StandardScaler

    sampled = deterministic_sample(features, cap, seed, stratify="angle")
    x = sampled[feature_cols].apply(pd.to_numeric, errors="coerce").replace([float("inf"), -float("inf")], pd.NA)
    keep = x.notna().all(axis=1)
    x = x.loc[keep].to_numpy(dtype=float)
    labels = sampled.loc[keep, "angle"].astype(str).to_numpy()
    x = StandardScaler().fit_transform(x)
    return x, labels


def _write_optional_note(layout, notes: list[str], message: str) -> None:
    notes.append(message)
    print(message)


def _pca_2d(x):
    import numpy as np

    if len(x) == 0:
        return np.empty((0, 2), dtype=float)
    centered = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    points = centered @ vt[:2].T
    if points.shape[1] == 1:
        points = np.column_stack([points[:, 0], np.zeros(len(points), dtype=float)])
    return points[:, :2]


def main() -> int:
    args = parse_args()
    layout = make_output_layout(args.output_root)
    write_manifest(
        layout,
        {
            "script": "scripts/analyze_resolution_limit.py",
            "data_root": args.data_root,
            "dataset": args.dataset,
            "angles": args.angles,
            "modality": args.modality,
            "sample_cap_plot": args.sample_cap_plot,
            "sample_cap_ml": args.sample_cap_ml,
            "seeds": args.seeds,
        },
    )
    figures: list[Path] = []
    tables: list[Path] = []
    notes: list[str] = []

    index_df = scan_dataset(args.data_root, args.dataset)
    index_df = _subset(index_df, args.dataset, args.modality, args.angles)
    features = extract_feature_table(index_df, dataset=args.dataset, modality=args.modality)
    features = features[features["angle_value"].isin([float(a) for a in args.angles])].copy()
    feature_csv = layout.tables / "proton_c_near_vertical_features.csv"
    features.to_csv(feature_csv, index=False, encoding="utf-8-sig")
    tables.append(feature_csv)
    features.to_csv(layout.cache / "proton_c_near_vertical_features.csv", index=False, encoding="utf-8-sig")

    counts = class_counts(index_df, args.dataset, [args.modality])
    dataset_summary = counts.rename(columns={"count": "samples"})
    csv_path, md_path = write_table(dataset_summary, layout.tables / "near_vertical_dataset_summary")
    tables.extend([csv_path, md_path])
    if features.empty:
        report = resolution_report(layout.root, {"Dataset Summary": dataset_summary}, figures, tables)
        write_markdown(layout.root / "resolution_limit_report.md", report)
        print(f"No near-vertical samples found. Wrote empty report to {layout.root}")
        return 0

    summary = feature_summary_by_angle(features)
    csv_path, md_path = write_table(summary, layout.tables / "near_vertical_feature_summary_by_angle")
    tables.extend([csv_path, md_path])

    reps = select_representatives(features)
    png, pdf = plot_representative_grid(reps, layout.figures / "near_vertical_representative_tot", "Near-vertical representative ToT samples")
    figures.extend([png, pdf])
    png, pdf = plot_mean_images_by_angle(
        deterministic_sample(features, args.sample_cap_plot, args.seeds[0]),
        layout.figures / "near_vertical_mean_tot_by_angle",
        "Near-vertical mean ToT by angle",
    )
    figures.extend([png, pdf])
    png, pdf = plot_adjacent_difference_maps(
        deterministic_sample(features, args.sample_cap_plot, args.seeds[0]),
        args.angles,
        layout.figures / "near_vertical_adjacent_difference_maps",
    )
    figures.extend([png, pdf])

    feature_cols = numeric_feature_columns(features)
    effects = feature_pair_effects(features, args.angles, feature_cols)
    csv_path, md_path = write_table(effects, layout.tables / "near_vertical_pairwise_effect_sizes", markdown_rows=100)
    tables.extend([csv_path, md_path])
    ks = effects[["angle_pair", "feature", "ks_statistic", "ks_pvalue"]].sort_values("ks_statistic", ascending=False)
    wasserstein = effects[["angle_pair", "feature", "wasserstein_distance"]].sort_values("wasserstein_distance", ascending=False)
    csv_path, md_path = write_table(ks, layout.tables / "near_vertical_adjacent_ks", markdown_rows=100)
    tables.extend([csv_path, md_path])
    csv_path, md_path = write_table(wasserstein, layout.tables / "near_vertical_adjacent_wasserstein", markdown_rows=100)
    tables.extend([csv_path, md_path])
    distance_summary = feature_distance_summary(effects)
    csv_path, md_path = write_table(distance_summary, layout.tables / "near_vertical_feature_distance_summary")
    tables.extend([csv_path, md_path])

    png, pdf = plot_feature_violin(
        deterministic_sample(features, args.sample_cap_plot, args.seeds[0]),
        layout.figures / "near_vertical_feature_violin_core",
        "Near-vertical core feature violin plots",
    )
    figures.extend([png, pdf])
    png, pdf = plot_feature_kde(
        deterministic_sample(features, args.sample_cap_plot, args.seeds[0]),
        layout.figures / "near_vertical_feature_kde_core",
        "Near-vertical core feature KDE/histogram overlays",
    )
    figures.extend([png, pdf])
    png, pdf = plot_heatmap(pivot_metric(effects, "ks_statistic"), layout.figures / "near_vertical_ks_heatmap", "Adjacent-angle KS statistic")
    figures.extend([png, pdf])
    png, pdf = plot_heatmap(
        pivot_metric(effects, "wasserstein_distance"),
        layout.figures / "near_vertical_wasserstein_heatmap",
        "Adjacent-angle Wasserstein distance",
    )
    figures.extend([png, pdf])

    x_embed, labels_embed = _embedding_inputs(features, feature_cols, args.sample_cap_plot, args.seeds[0])
    pca_points = _pca_2d(x_embed)
    png, pdf = plot_embedding(pca_points, labels_embed, layout.figures / "near_vertical_pca_features", "PCA of near-vertical handcrafted features")
    figures.extend([png, pdf])

    if sys.platform.startswith("win"):
        _write_optional_note(layout, notes, "Skipped UMAP on Windows: optional UMAP backends can terminate the process in this local environment.")
    else:
        try:
            import umap  # type: ignore

            umap_points = umap.UMAP(n_components=2, random_state=args.seeds[0]).fit_transform(x_embed)
            png, pdf = plot_embedding(
                umap_points,
                labels_embed,
                layout.figures / "near_vertical_umap_features",
                "UMAP of near-vertical handcrafted features",
            )
            figures.extend([png, pdf])
        except Exception as exc:  # noqa: BLE001 - optional dependency
            _write_optional_note(layout, notes, f"Skipped UMAP: {exc}")

    if args.tsne:
        try:
            from sklearn.manifold import TSNE

            tsne_cap = min(len(x_embed), 3000)
            tsne_points = TSNE(n_components=2, random_state=args.seeds[0], init="pca", learning_rate="auto").fit_transform(
                x_embed[:tsne_cap]
            )
            png, pdf = plot_embedding(
                tsne_points,
                labels_embed[:tsne_cap],
                layout.figures / "near_vertical_tsne_features",
                "t-SNE of near-vertical handcrafted features",
            )
            figures.extend([png, pdf])
        except Exception as exc:  # noqa: BLE001
            _write_optional_note(layout, notes, f"Skipped t-SNE: {exc}")
    else:
        _write_optional_note(layout, notes, "Skipped t-SNE: pass --tsne to enable the optional slow embedding.")

    ml_results, confusion_by_model, labels = run_ml_baselines(
        features,
        feature_cols=feature_cols,
        sample_cap=args.sample_cap_ml,
        seeds=args.seeds,
        include_rbf=True,
    )
    csv_path, md_path = write_table(ml_results, layout.tables / "near_vertical_ml_baselines")
    tables.extend([csv_path, md_path])
    ml_mean_std = aggregate_ml_results(ml_results)
    csv_path, md_path = write_table(ml_mean_std, layout.tables / "near_vertical_ml_baselines_mean_std")
    tables.extend([csv_path, md_path])

    for model_name, fig_name in [
        ("logistic_regression", "near_vertical_confusion_matrix_logreg"),
        ("random_forest", "near_vertical_confusion_matrix_rf"),
        ("mlp", "near_vertical_confusion_matrix_mlp"),
    ]:
        if model_name in confusion_by_model:
            png, pdf = plot_confusion_matrix(confusion_by_model[model_name], labels, layout.figures / fig_name, f"{model_name} confusion matrix")
            figures.extend([png, pdf])

    pairwise_auc = pairwise_auc_by_gap(features, feature_cols, args.seeds, args.sample_cap_ml)
    csv_path, md_path = write_table(pairwise_auc, layout.tables / "near_vertical_pairwise_auc")
    tables.extend([csv_path, md_path])
    auc_gap = auc_by_gap(pairwise_auc)
    csv_path, md_path = write_table(auc_gap, layout.tables / "near_vertical_auc_by_angle_gap")
    tables.extend([csv_path, md_path])
    if not pairwise_auc.empty:
        png, pdf = plot_metric_by_gap(pairwise_auc, "auc", layout.figures / "near_vertical_auc_by_angle_gap", "AUC by angle gap")
        figures.extend([png, pdf])
        png, pdf = plot_metric_by_gap(
            pairwise_auc,
            "balanced_accuracy",
            layout.figures / "near_vertical_balanced_acc_by_angle_gap",
            "Balanced accuracy by angle gap",
        )
        figures.extend([png, pdf])

    overfit_status = pd.DataFrame(
        [
            {
                "experiment": "near_vertical_overfit",
                "status": "not_run_by_analysis_script",
                "note": "This script records analysis outputs only; deep-learning overfit curves should be imported if already available.",
            },
            {
                "experiment": "positive_control_overfit",
                "status": "not_run_by_analysis_script",
                "note": "Use a matched 10-70 deg positive control if deep-learning overfit evidence is generated later.",
            },
        ]
    )
    csv_path, md_path = write_table(overfit_status, layout.tables / "near_vertical_overfit_experiment")
    tables.extend([csv_path, md_path])

    if notes:
        write_markdown(layout.root / "analysis_notes.md", "\n".join(f"- {note}" for note in notes) + "\n")

    report = resolution_report(
        layout.root,
        {
            "Dataset Summary": dataset_summary,
            "Feature Distance Summary": distance_summary.head(20),
            "ML Baseline Mean/Std": ml_mean_std,
            "AUC by Angle Gap": auc_gap,
        },
        figures,
        tables,
    )
    write_markdown(layout.root / "resolution_limit_report.md", report)
    print(f"Wrote resolution-limit analysis to {layout.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
