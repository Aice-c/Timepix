#!/usr/bin/env python
"""Analyze Proton/C near-vertical angular resolution limits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timepix.analysis.features import extract_feature_table, feature_summary_by_angle
from timepix.analysis.io import make_output_layout, scan_dataset, write_manifest
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
    plot_feature_panels,
    plot_heatmap,
    plot_mean_or_occupancy_grid,
    plot_metric_by_gap,
    plot_representative_grid,
)
from timepix.analysis.reports import resolution_report
from timepix.analysis.representative import deterministic_sample, select_representatives
from timepix.analysis.stats import feature_distance_summary, feature_pair_effects, pivot_metric
from timepix.analysis.tables import write_markdown, write_table
from timepix.analysis.workbook import write_analysis_workbook


CORE_SIZE = ["active_count", "bbox_width", "bbox_height", "bbox_area", "bbox_fill_ratio"]
CORE_INTENSITY = ["active_sum", "active_mean", "active_max", "tot_density_active"]
CORE_GEOMETRY = ["aspect_ratio", "pca_major_axis", "pca_minor_axis", "pca_axis_ratio", "weighted_radius_mean", "central_energy_ratio_r1", "central_energy_ratio_r2", "spatial_entropy"]
CORE_FEATURES = CORE_SIZE + CORE_INTENSITY + CORE_GEOMETRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze near-vertical Proton/C angle separability")
    parser.add_argument("--data-root", default="Data")
    parser.add_argument("--dataset-root", default=None, help="Optional concrete dataset directory, e.g. E:\\C1Analysis\\Proton_C")
    parser.add_argument("--dataset", default="Proton_C")
    parser.add_argument("--angles", nargs="+", type=float, default=[80, 82, 84, 86, 88, 90])
    parser.add_argument("--modality", default="ToT")
    parser.add_argument("--output-root", default="outputs/resolution_limit")
    parser.add_argument("--sample-cap-plot", type=int, default=5000)
    parser.add_argument("--sample-cap-ml", type=int, default=10000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--pixel-pitch-um", type=float, default=55.0)
    parser.add_argument("--sensor-thickness-um", type=float, default=None)
    parser.add_argument("--tsne", action="store_true", help="Run optional t-SNE embedding")
    return parser.parse_args()


def _write(layout, df: pd.DataFrame, stem: str, registry: list[tuple[str, str, str, pd.DataFrame]], title: str, note: str = ""):
    csv_path, md_path = write_table(df, layout.root / stem, markdown_rows=100)
    registry.append((stem, title, note, df))
    return csv_path, md_path


def _scan(data_root: str | Path, dataset: str, dataset_root: str | None) -> pd.DataFrame:
    if dataset_root:
        path = Path(dataset_root)
        frame = scan_dataset(path.parent, path.name, read_shapes=False)
        if path.name != dataset and not frame.empty:
            frame["dataset"] = dataset
            frame["dataset_root"] = str(path)
        return frame
    return scan_dataset(data_root, dataset, read_shapes=False)


def _subset(index_df: pd.DataFrame, dataset: str, modality: str, angles: list[float]) -> pd.DataFrame:
    values = pd.to_numeric(index_df["angle_value"], errors="coerce")
    return index_df[
        (index_df["dataset"] == dataset)
        & (index_df["modality"] == modality)
        & index_df["status"].isin(["ok", "unknown"])
        & values.isin([float(a) for a in angles])
    ].copy()


def _near_inventory(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for angle, group in features.groupby("angle"):
        shape = (group["input_shape_h"].astype(str) + "x" + group["input_shape_w"].astype(str)).mode()
        rows.append(
            {
                "angle": angle,
                "num_samples": len(group),
                "input_shape": shape.iloc[0] if not shape.empty else "unknown",
                "active_count_mean": group["active_count"].mean(),
                "active_count_std": group["active_count"].std(ddof=0),
                "active_sum_mean": group["active_sum"].mean(),
                "active_sum_std": group["active_sum"].std(ddof=0),
                "bbox_area_mean": group["bbox_area"].mean(),
                "pca_major_axis_mean": group["pca_major_axis"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("angle", key=lambda s: s.astype(float))


def _geometry_projection(angles: list[float], pixel_pitch_um: float, sensor_thickness_um: float | None, out_root: Path) -> pd.DataFrame:
    if sensor_thickness_um is None:
        write_markdown(
            out_root / "01_geometry_projection_missing.md",
            "# Geometry Projection Missing\n\n"
            "传感器厚度未确认，未生成定量投影长度表。\n\n"
            "论文中只能保留定性解释：接近 90° 时横向投影长度趋近于 0，相邻 2° 的形态差异可能被像素化、电荷扩散、阈值效应和能量沉积涨落淹没。\n",
        )
        return pd.DataFrame()
    rows = []
    for idx, angle in enumerate(angles):
        theta = np.radians(float(angle))
        projected_um = float(sensor_thickness_um / max(np.tan(theta), 1e-12))
        projected_px = projected_um / pixel_pitch_um
        next_delta = np.nan
        if idx < len(angles) - 1:
            next_theta = np.radians(float(angles[idx + 1]))
            next_px = float(sensor_thickness_um / max(np.tan(next_theta), 1e-12)) / pixel_pitch_um
            next_delta = abs(projected_px - next_px)
        rows.append(
            {
                "angle": angle,
                "sensor_thickness_um": sensor_thickness_um,
                "pixel_pitch_um": pixel_pitch_um,
                "projected_length_um": projected_um,
                "projected_length_pixel": projected_px,
                "delta_to_next_angle_pixel": next_delta,
            }
        )
    return pd.DataFrame(rows)


def _embedding_inputs(features: pd.DataFrame, feature_cols: list[str], cap: int, seed: int):
    from sklearn.preprocessing import StandardScaler

    sampled = deterministic_sample(features, cap, seed, stratify="angle")
    x = sampled[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = x.notna().all(axis=1)
    x = x.loc[keep].to_numpy(dtype=float)
    labels = sampled.loc[keep, "angle"].astype(str).to_numpy()
    x = StandardScaler().fit_transform(x)
    return x, labels


def _pca_2d(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.empty((0, 2), dtype=float)
    centered = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    points = centered @ vt[:2].T
    if points.shape[1] == 1:
        points = np.column_stack([points[:, 0], np.zeros(len(points), dtype=float)])
    return points[:, :2]


def _single_feature_auc(features: pd.DataFrame, feature_cols: list[str], angles: list[float]) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    rows = []
    for left, right in zip(angles[:-1], angles[1:]):
        subset = features[features["angle_value"].isin([float(left), float(right)])]
        y = (subset["angle_value"] == float(right)).astype(int).to_numpy()
        for feature in feature_cols:
            x = pd.to_numeric(subset[feature], errors="coerce").to_numpy(dtype=float)
            keep = np.isfinite(x)
            if keep.sum() < 3 or len(np.unique(y[keep])) != 2:
                continue
            auc_raw = float(roc_auc_score(y[keep], x[keep]))
            if auc_raw >= 0.5:
                auc = auc_raw
                direction = f"higher_{feature}_toward_{right:g}"
            else:
                auc = 1.0 - auc_raw
                direction = f"lower_{feature}_toward_{right:g}"
            rows.append(
                {
                    "feature": feature,
                    "angle_a": left,
                    "angle_b": right,
                    "angle_pair": f"{left:g}-{right:g}",
                    "auc": auc,
                    "raw_auc": auc_raw,
                    "auc_abs_from_random": abs(auc - 0.5),
                    "best_direction": direction,
                }
            )
    return pd.DataFrame(rows)


def _feature_sets(features: pd.DataFrame, all_cols: list[str]) -> dict[str, list[str]]:
    return {
        "basic_size_intensity": [c for c in ["active_count", "active_sum", "active_mean", "active_max"] if c in features.columns],
        "geometry_only": [c for c in ["bbox_width", "bbox_height", "bbox_area", "bbox_fill_ratio", "aspect_ratio", "pca_major_axis", "pca_minor_axis", "pca_axis_ratio"] if c in features.columns],
        "tot_statistics_only": [c for c in ["active_count", "active_sum", "active_mean", "active_std", "active_min", "active_max", "active_median", "active_q25", "active_q75"] if c in features.columns],
        "spatial_distribution": [c for c in ["weighted_radius_mean", "weighted_radius_std", "central_energy_ratio_r1", "central_energy_ratio_r2", "central_energy_ratio_r3", "spatial_entropy", "row_entropy", "col_entropy"] if c in features.columns],
        "all_features": all_cols,
    }


def _bar_accuracy(results: pd.DataFrame, out_path: Path):
    from timepix.analysis.plotting import save_figure, _plt

    plt = _plt()
    if results.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No ML results", ha="center", va="center")
        return save_figure(fig, out_path)
    all_features = results[results["feature_set"] == "all_features"] if "feature_set" in results else results
    summary = all_features.groupby("model", as_index=False)["test_acc"].mean().sort_values("test_acc", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(summary["model"], summary["test_acc"], color="#0072B2")
    ax.axhline(1 / 6, color="#D55E00", linestyle="--", linewidth=1.2, label="Random Baseline")
    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("Classical Model")
    ax.set_title("Classical ML Accuracy on Near-Vertical Angles")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    ax.grid(axis="y", alpha=0.22)
    return save_figure(fig, out_path)


def main() -> int:
    args = parse_args()
    layout = make_output_layout(args.output_root)
    figures_dir = layout.root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        layout,
        {
            "script": "scripts/analyze_resolution_limit.py",
            "data_root": args.data_root,
            "dataset_root": args.dataset_root,
            "dataset": args.dataset,
            "angles": args.angles,
            "modality": args.modality,
            "sample_cap_plot": args.sample_cap_plot,
            "sample_cap_ml": args.sample_cap_ml,
            "seeds": args.seeds,
            "pixel_pitch_um": args.pixel_pitch_um,
            "sensor_thickness_um": args.sensor_thickness_um,
        },
    )
    registry: list[tuple[str, str, str, pd.DataFrame]] = []
    figures: list[Path] = []
    tables: list[Path] = []
    notes: list[str] = []

    index_df = _subset(_scan(args.data_root, args.dataset, args.dataset_root), args.dataset, args.modality, args.angles)
    features = extract_feature_table(index_df, dataset=args.dataset, modality=args.modality)
    features = features[features["angle_value"].isin([float(a) for a in args.angles])].copy()
    feature_path = layout.root / "proton_c_near_vertical_features.csv"
    features.to_csv(feature_path, index=False, encoding="utf-8-sig")
    tables.append(feature_path)
    if features.empty:
        write_markdown(layout.root / "resolution_limit_report.md", "# Resolution-Limit Analysis Report\n\nNo near-vertical samples found.\n")
        print(f"No near-vertical samples found. Wrote empty report to {layout.root}")
        return 0

    inventory = _near_inventory(features)
    csv, md = _write(layout, inventory, "00_near_vertical_inventory", registry, "表 6-1 近垂直样本数与基础统计", "80-90 deg 近垂直角度清洗后 ToT 样本统计。")
    tables.extend([csv, md])
    projection = _geometry_projection(args.angles, args.pixel_pitch_um, args.sensor_thickness_um, layout.root)
    if not projection.empty:
        csv, md = _write(layout, projection, "01_geometry_projection", registry, "表 6-2 几何投影长度", "仅在传感器厚度已确认时生成。")
        tables.extend([csv, md])

    reps = select_representatives(features, per_group=5, group_cols=["dataset", "modality", "angle"])
    reps = reps.assign(selection_rule="medoid_nearest_samples")
    rep_table = reps[["dataset", "angle", "modality", "sample_id", "selection_rule", "active_count", "active_sum", "representative_distance", "source_path"]]
    csv, md = _write(layout, rep_table, "02_near_vertical_representative_samples", registry, "近垂直代表性样本表", "每角度自动选择距离类中心最近的样本。")
    tables.extend([csv, md])
    png, pdf = plot_representative_grid(reps, figures_dir / "02_near_vertical_representative_tot_grid", "Near-Vertical Representative ToT Samples", max_cols=6)
    figures.extend([png, pdf])

    sampled_plot = deterministic_sample(features, args.sample_cap_plot, args.seeds[0])
    png, pdf = plot_mean_or_occupancy_grid(sampled_plot, figures_dir / "03_near_vertical_mean_tot_heatmap", "Near-Vertical Mean ToT Heatmap", mode="mean")
    figures.extend([png, pdf])
    png, pdf = plot_mean_or_occupancy_grid(sampled_plot, figures_dir / "03_near_vertical_occupancy_heatmap", "Near-Vertical Occupancy Probability", mode="occupancy")
    figures.extend([png, pdf])
    png, pdf = plot_adjacent_difference_maps(sampled_plot, args.angles, figures_dir / "03_near_vertical_adjacent_difference_maps")
    figures.extend([png, pdf])

    feature_cols = [col for col in CORE_FEATURES if col in features.columns]
    summary = feature_summary_by_angle(features, feature_cols)
    csv, md = _write(layout, summary, "04_near_vertical_feature_summary", registry, "近垂直核心特征统计", "按角度统计核心 ToT、形态和空间分布特征。")
    tables.extend([csv, md])
    for cols, name, title in [
        (CORE_SIZE, "04_near_vertical_feature_violin_size", "Near-Vertical Size Features"),
        (CORE_INTENSITY, "04_near_vertical_feature_violin_intensity", "Near-Vertical Intensity Features"),
        (CORE_GEOMETRY, "04_near_vertical_feature_violin_geometry", "Near-Vertical Geometry Features"),
    ]:
        png, pdf = plot_feature_panels(sampled_plot, cols, figures_dir / name, title)
        figures.extend([png, pdf])

    effects_adj = feature_pair_effects(features, args.angles, feature_cols, adjacent_only=True)
    csv, md = _write(layout, effects_adj, "05_pairwise_effect_size_adjacent", registry, "相邻角度效应量", "包含 KS、Wasserstein、Cliff's delta、median difference 和 IQR overlap ratio。")
    tables.extend([csv, md])
    csv, md = _write(layout, effects_adj, "near_vertical_pairwise_effect_size", registry, "近垂直相邻角度效应量", "用户指定文件名；内容同相邻角度效应量表。")
    tables.extend([csv, md])
    effects_all = feature_pair_effects(features, args.angles, feature_cols, adjacent_only=False)
    csv, md = _write(layout, effects_all, "05_pairwise_effect_size_all_pairs", registry, "全部角度对效应量", "用于补充相邻角度之外的两两分布距离。")
    tables.extend([csv, md])
    effect_summary = feature_distance_summary(effects_adj)
    csv, md = _write(layout, effect_summary, "05_feature_max_effect_size", registry, "特征最大效应量汇总", "按特征汇总相邻角度最大分布差异。")
    tables.extend([csv, md])
    for metric, name, title in [
        ("ks_statistic", "05_ks_heatmap_by_feature", "KS Statistic by Feature"),
        ("wasserstein_distance_normalized", "05_wasserstein_heatmap_by_feature", "Normalized Wasserstein Distance by Feature"),
        ("cliffs_delta", "05_cliffs_delta_heatmap_by_feature", "Cliff's Delta by Feature"),
    ]:
        png, pdf = plot_heatmap(pivot_metric(effects_adj, metric), figures_dir / name, title)
        figures.extend([png, pdf])

    single_auc = _single_feature_auc(features, feature_cols, args.angles)
    csv, md = _write(layout, single_auc, "06_single_feature_pairwise_auc", registry, "单特征 pairwise AUC", "AUC 接近 0.5 表明单个特征难以区分相邻角度。")
    tables.extend([csv, md])
    csv, md = _write(layout, single_auc, "near_vertical_single_feature_auc", registry, "近垂直单特征 pairwise AUC", "用户指定文件名；仅相邻角度对。")
    tables.extend([csv, md])
    if not single_auc.empty:
        png, pdf = plot_heatmap(single_auc.pivot_table(index="feature", columns="angle_pair", values="auc", aggfunc="max").reset_index(), figures_dir / "06_single_feature_auc_heatmap", "Single Feature Pairwise AUC")
        figures.extend([png, pdf])

    all_cols = numeric_feature_columns(features)
    x_embed, labels_embed = _embedding_inputs(features, all_cols, args.sample_cap_plot, args.seeds[0])
    pca_points = _pca_2d(x_embed)
    png, pdf = plot_embedding(pca_points, labels_embed, figures_dir / "07_pca_near_vertical_features", "PCA of Near-Vertical Handcrafted Features")
    figures.extend([png, pdf])
    try:
        import umap  # type: ignore

        umap_points = umap.UMAP(n_components=2, random_state=args.seeds[0]).fit_transform(x_embed)
        png, pdf = plot_embedding(umap_points, labels_embed, figures_dir / "07_umap_near_vertical_features", "UMAP of Near-Vertical Handcrafted Features")
        figures.extend([png, pdf])
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Skipped UMAP: {exc}")
        print(f"Skipped UMAP: {exc}")
    if args.tsne:
        try:
            from sklearn.manifold import TSNE

            tsne_cap = min(len(x_embed), 3000)
            tsne_points = TSNE(n_components=2, random_state=args.seeds[0], init="pca", learning_rate="auto").fit_transform(x_embed[:tsne_cap])
            png, pdf = plot_embedding(tsne_points, labels_embed[:tsne_cap], figures_dir / "07_tsne_near_vertical_features", "t-SNE of Near-Vertical Handcrafted Features")
            figures.extend([png, pdf])
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Skipped t-SNE: {exc}")
            print(f"Skipped t-SNE: {exc}")
    else:
        notes.append("Skipped t-SNE: pass --tsne to enable optional slow embedding.")
        print("Skipped t-SNE: pass --tsne to enable the optional slow embedding.")

    ml_results, confusion_by_model, labels = run_ml_baselines(
        features,
        feature_cols=all_cols,
        sample_cap=args.sample_cap_ml,
        seeds=args.seeds,
        include_rbf=True,
        feature_sets=_feature_sets(features, all_cols),
    )
    csv, md = _write(layout, ml_results, "08_classical_ml_by_seed", registry, "传统机器学习逐 seed 结果", "不使用 test set 调参；test split 只用于最终指标报告。")
    tables.extend([csv, md])
    ml_summary = aggregate_ml_results(ml_results)
    csv, md = _write(layout, ml_summary, "08_classical_ml_summary", registry, "表 6-4 传统机器学习六分类结果", "随机基线为 1/6 = 16.67%。")
    tables.extend([csv, md])
    requested_models = {"random_forest", "extra_trees", "logistic_regression", "linear_svm"}
    requested_ml = ml_results[
        (ml_results["feature_set"] == "all_features")
        & (ml_results["model"].isin(requested_models))
    ].copy()
    csv, md = _write(layout, requested_ml, "near_vertical_classical_ml_summary", registry, "近垂直传统机器学习六分类逐 seed 结果", "包含 RandomForest、ExtraTrees、LogisticRegression、LinearSVM 的 train/val/test 指标。")
    tables.extend([csv, md])
    (layout.root / "08_classical_ml_confusion_matrices.json").write_text(
        json.dumps({name: cm.tolist() for name, cm in confusion_by_model.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for model_name, fig_name in [
        ("random_forest", "08_classical_ml_confusion_matrix_rf"),
        ("extra_trees", "08_classical_ml_confusion_matrix_extra_trees"),
        ("logistic_regression", "08_classical_ml_confusion_matrix_logreg"),
        ("mlp", "08_classical_ml_confusion_matrix_mlp"),
    ]:
        if model_name in confusion_by_model:
            png, pdf = plot_confusion_matrix(confusion_by_model[model_name], labels, figures_dir / fig_name, f"{model_name} Confusion Matrix")
            figures.extend([png, pdf])
    png, pdf = _bar_accuracy(ml_results, figures_dir / "08_classical_ml_accuracy_bar")
    figures.extend([png, pdf])

    pairwise_auc = pairwise_auc_by_gap(features, all_cols, args.seeds, args.sample_cap_ml).assign(model="logistic_regression")
    csv, md = _write(layout, pairwise_auc, "09_pairwise_classical_ml", registry, "相邻/两两角度传统 ML 可分性", "使用 LogisticRegression 基线计算 pairwise AUC。")
    tables.extend([csv, md])
    auc_gap = auc_by_gap(pairwise_auc)
    csv, md = _write(layout, auc_gap, "09_auc_by_angle_gap", registry, "Pairwise AUC by Angle Gap", "角度间隔越大，AUC 若仍低则说明形态可分性弱。")
    tables.extend([csv, md])
    if not pairwise_auc.empty:
        png, pdf = plot_metric_by_gap(pairwise_auc, "auc", figures_dir / "09_pairwise_auc_by_angle_gap", "Pairwise AUC by Angle Gap")
        figures.extend([png, pdf])
        png, pdf = plot_metric_by_gap(pairwise_auc, "balanced_accuracy", figures_dir / "09_pairwise_balanced_acc_by_angle_gap", "Pairwise Balanced Accuracy by Angle Gap")
        figures.extend([png, pdf])

    dl_audit = pd.DataFrame(
        [
            {
                "run_name": "near_vertical_dl_runs",
                "model": "unknown_or_external",
                "input_shape": inventory["input_shape"].mode().iloc[0] if not inventory.empty else "unknown",
                "angles": ",".join(f"{a:g}" for a in args.angles),
                "epochs": np.nan,
                "train_acc": np.nan,
                "val_acc": np.nan,
                "test_acc": np.nan,
                "has_curve": False,
                "has_confusion_matrix": False,
                "usable_for_paper": "diagnostic_only",
                "notes": "Existing informal DL failure observations should be audited before being used as paper evidence.",
            }
        ]
    )
    csv, md = _write(layout, dl_audit, "10_existing_dl_runs_audit", registry, "已有深度学习结果审计", "不完整结果标记为 diagnostic_only。")
    tables.extend([csv, md])
    csv, md = _write(layout, dl_audit, "near_vertical_dl_failure_audit", registry, "近垂直深度学习失败结果审计", "用户指定字段格式；不完整结果标记为 diagnostic_only。")
    tables.extend([csv, md])

    if notes:
        write_markdown(layout.root / "analysis_notes.md", "\n".join(f"- {note}" for note in notes) + "\n")

    workbook_path = write_analysis_workbook(registry, layout.root / "resolution_limit_tables.xlsx", title="Timepix 近垂直分辨极限统计表")
    print(f"Wrote resolution workbook: {workbook_path}")
    cautious = (
        "在当前探测器设置、事件提取方法、当前清洗流程、ToT 单模态矩阵表示和已测试特征/模型族条件下，"
        "Proton/C 近垂直角度 80-90 deg、2 deg 间隔数据没有表现出足够稳定的可分性；"
        "该结论不等价于证明所有探测器设置、所有模态或所有物理重建方法都无法区分近垂直角度。"
    )
    report = resolution_report(
        layout.root,
        {
            "Near-Vertical Inventory": inventory,
            "Feature Effect Summary": effect_summary.head(20),
            "Classical ML Summary": ml_summary,
            "AUC by Angle Gap": auc_gap,
        },
        figures,
        tables,
    )
    report += "\n## Thesis Caveats\n\n" + cautious + "\n\nPCA、t-SNE 和 UMAP 仅作为辅助可视化，不能作为唯一证据。\n"
    write_markdown(layout.root / "resolution_limit_report.md", report)
    print(f"Wrote resolution-limit analysis to {layout.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
