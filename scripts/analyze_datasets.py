#!/usr/bin/env python
"""Generate thesis-oriented dataset construction and quality analysis outputs."""

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
from timepix.analysis.io import (
    class_counts,
    make_output_layout,
    paired_modality_report,
    read_matrix,
    scan_dataset,
    write_manifest,
)
from timepix.analysis.ml import numeric_feature_columns
from timepix.analysis.plotting import (
    plot_alpha_pair_grid,
    plot_box_by_angle,
    plot_class_counts,
    plot_embedding,
    plot_feature_panels,
    plot_feature_scatter,
    plot_mean_or_occupancy_grid,
    plot_preprocessing_pipeline,
    plot_representative_grid,
)
from timepix.analysis.progress import iter_progress
from timepix.analysis.reports import dataset_report
from timepix.analysis.representative import deterministic_sample, select_representatives
from timepix.analysis.tables import write_markdown, write_table
from timepix.analysis.workbook import write_analysis_workbook


REGULAR_PROTON_ANGLES = [10, 20, 30, 45, 50, 60, 70]
NEAR_VERTICAL_ANGLES = [80, 82, 84, 86, 88, 90]
EXPECTED_PROTON_ANGLES = [10, 20, 30, 45, 50, 60, 70, 80, 82, 84, 86, 88, 90]
PROTON_TRAINING_CONFIGS = [
    "configs/experiments/b1_proton_c7_resnet18_tot_best_patience8_3seed.yaml",
    "configs/experiments/b3b_proton_c7_expected_mae_3seed.yaml",
    "configs/experiments/proton_resnet18_tot.yaml",
]

SIZE_FEATURES = ["active_count", "bbox_width", "bbox_height", "bbox_area", "bbox_fill_ratio"]
INTENSITY_FEATURES = ["active_sum", "active_mean", "active_max", "tot_density_active", "tot_density_bbox"]
GEOMETRY_FEATURES = ["aspect_ratio", "pca_major_axis", "pca_minor_axis", "pca_axis_ratio", "weighted_radius_mean", "central_energy_ratio_r1"]
TOA_FEATURES = ["toa_span", "toa_std", "toa_major_axis_corr_abs"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Timepix datasets for thesis figures/tables")
    parser.add_argument("--data-root", default="Data", help="Default parent directory containing dataset folders")
    parser.add_argument("--dataset-root", action="append", default=[], help="Optional mapping like Alpha_100=D:\\Data\\Alpha_100")
    parser.add_argument("--output-root", default="outputs/data_analysis", help="Output root")
    parser.add_argument("--datasets", nargs="+", default=["Alpha_100", "Proton_C"], help="Datasets to analyze")
    parser.add_argument("--sample-cap-plot", type=int, default=5000, help="Per-plot deterministic sample cap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-dir", default="outputs/splits", help="Directory containing split manifests")
    parser.add_argument("--reuse-feature-cache", action="store_true", help="Reuse 04_event_features_*.csv in output-root when available")
    return parser.parse_args()


def _dataset_root_map(items: list[str]) -> dict[str, Path]:
    mapping = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--dataset-root must use DATASET=PATH, got {item!r}")
        name, path = item.split("=", 1)
        mapping[name] = Path(path)
    return mapping


def _scan_all(data_root: str | Path, datasets: list[str], dataset_roots: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for dataset in datasets:
        if dataset in dataset_roots:
            dataset_path = dataset_roots[dataset]
            parent = dataset_path.parent
            dataset_name = dataset_path.name
            if dataset_name != dataset:
                print(f"Warning: dataset-root folder name {dataset_name!r} differs from dataset name {dataset!r}; using folder name.")
            frame = scan_dataset(parent, dataset_name, read_shapes=False)
            if dataset_name != dataset and not frame.empty:
                frame["dataset"] = dataset
                frame["dataset_root"] = str(dataset_path)
        else:
            frame = scan_dataset(data_root, dataset, read_shapes=False)
        frames.append(frame)
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        from timepix.analysis.io import INDEX_COLUMNS

        return pd.DataFrame(columns=INDEX_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _filter_angles(frame: pd.DataFrame, angles: list[float]) -> pd.DataFrame:
    if frame.empty:
        return frame
    values = pd.to_numeric(frame["angle_value"] if "angle_value" in frame.columns else frame["angle"], errors="coerce")
    return frame[values.isin([float(a) for a in angles])].copy()


def _write(layout, df: pd.DataFrame, stem: str, table_registry: list[tuple[str, str, str, pd.DataFrame]], title: str, note: str = ""):
    csv_path, md_path = write_table(df, layout.root / stem, markdown_rows=80)
    table_registry.append((stem, title, note, df))
    return csv_path, md_path


def _load_feature_cache(root: Path) -> pd.DataFrame:
    paths = [
        root / "04_event_features_alpha_100.csv",
        root / "04_event_features_proton_c.csv",
        root / "04_event_features_proton_c_7.csv",
    ]
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _split_rows(split_dir: str | Path, datasets: list[str]) -> pd.DataFrame:
    split_dir = Path(split_dir)
    rows = []
    if not split_dir.exists():
        return pd.DataFrame(columns=["dataset", "modality", "split_file", "split_name", "angle", "num_samples", "class_fraction"])
    known_datasets = set(datasets)
    if "Proton_C" in known_datasets:
        known_datasets.add("Proton_C_7")
    for path in split_dir.glob("*.json"):
        name = path.name
        matched = [dataset for dataset in known_datasets if name.startswith(f"{dataset}_")]
        if not matched:
            continue
        dataset = max(matched, key=len)
        modality = "ToT-ToA" if "ToT-ToA" in name else ("ToA" if "_ToA_" in name else ("ToT" if "_ToT_" in name else "unknown"))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for split_name in ["train", "val", "test"]:
            keys = payload.get(split_name, [])
            angles = []
            for key in keys:
                try:
                    angles.append(str(key).split("/", 1)[0])
                except Exception:
                    angles.append("unknown")
            counts = pd.Series(angles).value_counts().sort_index(key=lambda idx: [float(x) if str(x).replace(".", "", 1).isdigit() else 999 for x in idx])
            total = int(counts.sum())
            for angle, count in counts.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "modality": modality,
                        "split_file": str(path),
                        "split_name": split_name,
                        "angle": angle,
                        "num_samples": int(count),
                        "class_fraction": float(count / max(total, 1)),
                    }
                )
    return pd.DataFrame(rows)


def _split_lookup(split_distribution: pd.DataFrame, split_dir: str | Path, dataset: str, modality_hint: str = "ToT") -> dict[str, str]:
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return {}
    candidates = sorted(split_dir.glob(f"{dataset}_{modality_hint}_*.json"))
    if not candidates:
        candidates = sorted(split_dir.glob(f"{dataset}*.json"))
    if not candidates:
        return {}
    try:
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    except Exception:
        return {}
    lookup = {}
    for split_name in ["train", "val", "test"]:
        for key in payload.get(split_name, []):
            lookup[str(key)] = split_name
    return lookup


def _inventory_from_features(features: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if features.empty:
        return pd.DataFrame()
    for (dataset, angle, modality), group in features.groupby(["dataset", "angle", "modality"]):
        ok = group[group["feature_error"].fillna("") == ""]
        shape = "unknown"
        if not ok.empty:
            shape_counts = (ok["input_shape_h"].astype(str) + "x" + ok["input_shape_w"].astype(str)).value_counts()
            shape = shape_counts.index[0]
        source_paths = index_df[(index_df["dataset"] == dataset) & (index_df["angle"] == angle) & (index_df["modality"] == modality)]
        rows.append(
            {
                "dataset": dataset,
                "particle_type": "Alpha" if str(dataset).startswith("Alpha") else "Proton/C",
                "angle": angle,
                "modality": modality,
                "num_samples": int(len(group)),
                "input_shape": shape,
                "dtype": ok["matrix_dtype"].mode().iloc[0] if not ok.empty and not ok["matrix_dtype"].mode().empty else "unknown",
                "min_value": ok["matrix_min"].min() if not ok.empty else np.nan,
                "max_value": ok["matrix_max"].max() if not ok.empty else np.nan,
                "num_all_zero_samples": int(ok["is_all_zero"].sum()) if not ok.empty else 0,
                "num_nan_samples": int((ok["num_nan_values"] > 0).sum()) if not ok.empty else 0,
                "num_inf_samples": int((ok["num_inf_values"] > 0).sum()) if not ok.empty else 0,
                "num_negative_samples": int((ok["num_negative_values"] > 0).sum()) if not ok.empty else 0,
                "source_path": str(source_paths["path"].iloc[0]) if not source_paths.empty else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "modality", "angle"], key=lambda s: s.map(lambda x: float(x) if str(x).replace(".", "", 1).isdigit() else 999))


def _alpha_pairing(index_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    alpha = index_df[index_df["dataset"] == "Alpha_100"]
    for angle, group in alpha.groupby("angle"):
        tot = set(group.loc[group["modality"] == "ToT", "sample_key"])
        toa = set(group.loc[group["modality"] == "ToA", "sample_key"])
        paired = tot & toa
        rows.append(
            {
                "angle": angle,
                "num_tot": len(tot),
                "num_toa": len(toa),
                "num_paired": len(paired),
                "num_tot_without_toa": len(tot - toa),
                "num_toa_without_tot": len(toa - tot),
                "pairing_rate": len(paired) / max(len(tot | toa), 1),
                "example_missing_tot_keys": "; ".join(sorted(list(toa - tot))[:5]),
                "example_missing_toa_keys": "; ".join(sorted(list(tot - toa))[:5]),
            }
        )
    return pd.DataFrame(rows)


def _cleaning_thresholds(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    alpha = features[(features["dataset"] == "Alpha_100") & (features["modality"] == "ToT")]
    if not alpha.empty:
        rows.append(
            {
                "dataset": "Alpha_100",
                "angle": "all",
                "threshold_scope": "global",
                "active_count_min": int(alpha["active_count"].min()),
                "active_count_max": int(alpha["active_count"].max()),
                "active_sum_min": float(alpha["active_sum"].min()),
                "active_sum_max": float(alpha["active_sum"].max()),
                "source_config_file": "unavailable",
                "notes": "阈值由清洗后最终数据集反推；原始候选事件与阈值配置未找到。",
            }
        )
    proton = features[(features["dataset"] == "Proton_C") & (features["modality"] == "ToT")]
    for angle, group in proton.groupby("angle"):
        rows.append(
            {
                "dataset": "Proton_C",
                "angle": angle,
                "threshold_scope": "per_angle",
                "active_count_min": int(group["active_count"].min()),
                "active_count_max": int(group["active_count"].max()),
                "active_sum_min": float(group["active_sum"].min()),
                "active_sum_max": float(group["active_sum"].max()),
                "source_config_file": "unavailable",
                "notes": "Proton/C 按角度清洗；此处仅根据清洗后最终样本分布推断阈值范围。",
            }
        )
    return pd.DataFrame(rows)


def _sample_counts(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, angle, modality), group in features.groupby(["dataset", "angle", "modality"]):
        total = len(features[(features["dataset"] == dataset) & (features["modality"] == modality)])
        shape = (group["input_shape_h"].astype(str) + "x" + group["input_shape_w"].astype(str)).mode()
        rows.append(
            {
                "dataset": dataset,
                "angle": angle,
                "modality": modality,
                "num_samples": len(group),
                "fraction": len(group) / max(total, 1),
                "input_shape": shape.iloc[0] if not shape.empty else "unknown",
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "modality", "angle"], key=lambda s: s.map(lambda x: float(x) if str(x).replace(".", "", 1).isdigit() else 999))


def _representatives(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    rows = []
    medoids = select_representatives(features, group_cols=["dataset", "modality", "angle"])
    if not medoids.empty:
        rows.append(medoids.assign(selection_rule="medoid"))
    for rule, q in [("low_active_sum_q25", 0.25), ("high_active_sum_q75", 0.75)]:
        pieces = []
        for _, group in features.groupby(["dataset", "modality", "angle"]):
            target = group["active_sum"].quantile(q)
            idx = (group["active_sum"] - target).abs().sort_values().head(1).index
            pieces.append(group.loc[idx].assign(representative_distance=(group.loc[idx, "active_sum"] - target).abs().to_numpy()))
        if pieces:
            rows.append(pd.concat(pieces, ignore_index=True).assign(selection_rule=rule))
    reps = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return reps[
        [
            "dataset",
            "angle",
            "modality",
            "sample_id",
            "selection_rule",
            "active_count",
            "active_sum",
            "representative_distance",
            "source_path",
        ]
    ].rename(columns={"representative_distance": "distance_to_class_center"})


def _class_summary_alpha(alpha_tot: pd.DataFrame, alpha_toa: pd.DataFrame, split_distribution: pd.DataFrame) -> pd.DataFrame:
    rows = []
    alpha_splits = split_distribution[split_distribution["dataset"] == "Alpha_100"].copy()
    if not alpha_splits.empty:
        preferred = alpha_splits[
            alpha_splits["split_file"].astype(str).str.contains("Alpha_100_ToT_seed", regex=False)
            & ~alpha_splits["split_file"].astype(str).str.contains("ToT-ToA", regex=False)
        ]
        if preferred.empty:
            preferred = alpha_splits[alpha_splits["split_file"].astype(str).str.contains("Alpha_100_ToT-ToA", regex=False)]
        alpha_splits = preferred if not preferred.empty else alpha_splits
    for angle, group in alpha_tot.groupby("angle"):
        toa = alpha_toa[alpha_toa["angle"] == angle]
        split = alpha_splits[alpha_splits["angle"].astype(str) == str(angle)]
        counts = split.pivot_table(index="angle", columns="split_name", values="num_samples", aggfunc="sum").to_dict("records")
        counts = counts[0] if counts else {}
        rows.append(
            {
                "angle": angle,
                "num_samples": len(group),
                "num_train": counts.get("train", np.nan),
                "num_val": counts.get("val", np.nan),
                "num_test": counts.get("test", np.nan),
                "tot_active_count_mean": group["active_count"].mean(),
                "tot_active_count_std": group["active_count"].std(ddof=0),
                "tot_active_sum_mean": group["active_sum"].mean(),
                "tot_active_sum_std": group["active_sum"].std(ddof=0),
                "toa_span_mean": toa["toa_span"].mean() if "toa_span" in toa else np.nan,
                "toa_span_std": toa["toa_span"].std(ddof=0) if "toa_span" in toa else np.nan,
                "bbox_area_mean": group["bbox_area"].mean(),
                "bbox_area_std": group["bbox_area"].std(ddof=0),
                "pca_major_axis_mean": group["pca_major_axis"].mean(),
                "pca_major_axis_std": group["pca_major_axis"].std(ddof=0),
            }
        )
    return pd.DataFrame(rows)


def _mode_shape(frame: pd.DataFrame) -> str:
    if frame.empty or "input_shape_h" not in frame.columns or "input_shape_w" not in frame.columns:
        return "unknown"
    shapes = (
        pd.to_numeric(frame["input_shape_h"], errors="coerce").astype("Int64").astype(str)
        + "x"
        + pd.to_numeric(frame["input_shape_w"], errors="coerce").astype("Int64").astype(str)
    )
    shapes = shapes[~shapes.str.contains("<NA>", regex=False)]
    if shapes.empty:
        return "unknown"
    return str(shapes.value_counts().index[0])


def _config_value(path: Path, dotted: str, default="unavailable"):
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return default
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _training_crop_sizes() -> str:
    values = []
    for item in PROTON_TRAINING_CONFIGS:
        value = _config_value(ROOT / item, "data.crop_size", default=None)
        if value is not None:
            values.append(str(value))
    return ",".join(sorted(set(values))) if values else "unavailable"


def _proton_input_shape_audit(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    crop_sizes = _training_crop_sizes()
    sample_shape = _config_value(ROOT / "configs/datasets/proton_c_7.yaml", "sample_shape", default="unavailable")
    sample_shape_text = "x".join(map(str, sample_shape)) if isinstance(sample_shape, list) else str(sample_shape)
    for dataset in ["Proton_C", "Proton_C_7"]:
        subset = features[(features["dataset"] == dataset) & (features["modality"] == "ToT")]
        file_shape = _mode_shape(subset)
        try:
            crop_size = int(str(crop_sizes).split(",", 1)[0])
        except Exception:
            crop_size = 0
        if crop_size > 0:
            loader_shape = f"1x{crop_size}x{crop_size}"
        elif "x" in file_shape and file_shape != "unknown":
            loader_shape = f"1x{file_shape}"
        else:
            loader_shape = "unknown"
        rows.append(
            {
                "dataset": dataset,
                "modality": "ToT",
                "data_source": "full Proton_C directory" if dataset == "Proton_C" else "derived from Proton_C angles 10,20,30,45,50,60,70",
                "file_saved_shape_mode": file_shape,
                "config_sample_shape": sample_shape_text if dataset == "Proton_C_7" else "not configured for full Proton_C analysis",
                "training_crop_size_values": crop_sizes,
                "dataset_loader_output_shape_inferred": loader_shape,
                "training_model_input_shape_inferred": loader_shape,
                "loader_resize_or_crop_rule": "TimepixDataset only applies center_crop_array when data.crop_size > 0; crop_size=0 keeps saved matrix shape.",
                "notes": "ResNet-based Proton training has no fixed image_size resize; effective spatial input follows loader output.",
            }
        )
    return pd.DataFrame(rows)


def _corresponding_tot_path(toa_path: str) -> str:
    path = Path(toa_path)
    parts = ["ToT" if part == "ToA" else part for part in path.parts]
    if parts:
        parts[-1] = path.name.replace("ToA", "ToT")
    return str(Path(*parts))


def _alpha_toa_negative_audit(alpha_toa: pd.DataFrame) -> pd.DataFrame:
    rows = []
    negatives = alpha_toa[pd.to_numeric(alpha_toa.get("num_negative_values", 0), errors="coerce").fillna(0) > 0]
    for _, row in negatives.iterrows():
        source_path = str(row.get("source_path", ""))
        try:
            array = read_matrix(source_path)
            finite = array[np.isfinite(array)]
            nonzero = finite[finite != 0]
            negative = finite[finite < 0]
            nonzero_summary = {
                "nonzero_count": int(nonzero.size),
                "nonzero_min": float(np.min(nonzero)) if nonzero.size else np.nan,
                "nonzero_q01": float(np.quantile(nonzero, 0.01)) if nonzero.size else np.nan,
                "nonzero_q25": float(np.quantile(nonzero, 0.25)) if nonzero.size else np.nan,
                "nonzero_median": float(np.quantile(nonzero, 0.50)) if nonzero.size else np.nan,
                "nonzero_q75": float(np.quantile(nonzero, 0.75)) if nonzero.size else np.nan,
                "nonzero_q99": float(np.quantile(nonzero, 0.99)) if nonzero.size else np.nan,
                "nonzero_max": float(np.max(nonzero)) if nonzero.size else np.nan,
                "negative_count": int(negative.size),
                "negative_min": float(np.min(negative)) if negative.size else np.nan,
                "negative_max": float(np.max(negative)) if negative.size else np.nan,
            }
            read_status = "ok"
            interpretation = "single_negative_sample" if len(negatives) == 1 else "multiple_negative_samples"
        except Exception as exc:  # noqa: BLE001
            nonzero_summary = {
                "nonzero_count": np.nan,
                "nonzero_min": np.nan,
                "nonzero_q01": np.nan,
                "nonzero_q25": np.nan,
                "nonzero_median": np.nan,
                "nonzero_q75": np.nan,
                "nonzero_q99": np.nan,
                "nonzero_max": np.nan,
                "negative_count": row.get("num_negative_values", np.nan),
                "negative_min": np.nan,
                "negative_max": np.nan,
            }
            read_status = f"read_error: {exc}"
            interpretation = "read_error"
        tot_path = _corresponding_tot_path(source_path)
        rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "angle": row.get("angle", ""),
                "source_path": source_path,
                "corresponding_tot_path": tot_path,
                "corresponding_tot_exists": Path(tot_path).exists(),
                "matrix_min": row.get("matrix_min", np.nan),
                "matrix_max": row.get("matrix_max", np.nan),
                **nonzero_summary,
                "read_status": read_status,
                "preliminary_judgement": interpretation,
                "notes": "Negative raw ToA values are audited before any log/relative transform; inspect timestamp handling if more samples appear.",
            }
        )
    columns = [
        "sample_id",
        "angle",
        "source_path",
        "corresponding_tot_path",
        "corresponding_tot_exists",
        "matrix_min",
        "matrix_max",
        "nonzero_count",
        "nonzero_min",
        "nonzero_q01",
        "nonzero_q25",
        "nonzero_median",
        "nonzero_q75",
        "nonzero_q99",
        "nonzero_max",
        "negative_count",
        "negative_min",
        "negative_max",
        "read_status",
        "preliminary_judgement",
        "notes",
    ]
    return pd.DataFrame(rows, columns=columns)


def _proton_angle_consistency_audit(features: pd.DataFrame, split_distribution: pd.DataFrame) -> pd.DataFrame:
    proton = features[(features["dataset"] == "Proton_C") & (features["modality"] == "ToT")]
    inventory_counts = proton.groupby("angle", as_index=False).size().rename(columns={"size": "num_inventory_samples"})
    inventory_counts["angle_text"] = inventory_counts["angle"].astype(str)
    split = split_distribution[
        (split_distribution["dataset"] == "Proton_C")
        & (split_distribution["modality"] == "ToT")
        & (split_distribution["split_file"].astype(str).str.contains("Proton_C_ToT_seed", regex=False))
    ].copy()
    split_counts = split.groupby("angle", as_index=False).agg(
        split_total_samples=("num_samples", "sum"),
        split_files=("split_file", lambda s: "; ".join(sorted(set(map(str, s))))),
    )
    split_counts["angle_text"] = split_counts["angle"].astype(str)
    merged = pd.merge(inventory_counts, split_counts, on="angle_text", how="outer", suffixes=("_inventory", "_split"))
    rows = []
    for _, row in merged.iterrows():
        angle = row.get("angle_text")
        in_inventory = not pd.isna(row.get("num_inventory_samples"))
        in_split = not pd.isna(row.get("split_total_samples"))
        if in_inventory and in_split:
            status = "consistent"
        elif in_split and not in_inventory:
            status = "split_residual_no_current_data"
        else:
            status = "current_data_without_full_split"
        rows.append(
            {
                "angle": angle,
                "in_current_inventory": bool(in_inventory),
                "in_full_proton_split": bool(in_split),
                "num_inventory_samples": int(row.get("num_inventory_samples", 0)) if in_inventory else 0,
                "split_total_samples": int(row.get("split_total_samples", 0)) if in_split else 0,
                "split_files": row.get("split_files", "") if in_split else "",
                "status": status,
                "interpretation": "legacy split residue or unsynced old data" if status == "split_residual_no_current_data" else "",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("angle", key=lambda s: pd.to_numeric(s, errors="coerce"))


def _association(features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    try:
        from scipy.stats import kruskal, pearsonr, spearmanr
    except ImportError:
        return pd.DataFrame()
    rows = []
    y = pd.to_numeric(features["angle_value"], errors="coerce").to_numpy(dtype=float)
    for feature in feature_cols:
        x = pd.to_numeric(features[feature], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.sum() < 3:
            continue
        groups = [pd.to_numeric(g[feature], errors="coerce").dropna().to_numpy() for _, g in features.groupby("angle")]
        groups = [g for g in groups if len(g) > 1]
        stat = kruskal(*groups).statistic if len(groups) >= 2 else np.nan
        rows.append(
            {
                "feature": feature,
                "spearman_corr_with_angle": spearmanr(x[keep], y[keep]).statistic,
                "pearson_corr_with_angle": pearsonr(x[keep], y[keep]).statistic,
                "anova_or_kruskal_stat": stat,
                "ks_max_adjacent": np.nan,
                "notes": "Correlation is descriptive; effect size and overlap should be reported together.",
            }
        )
    return pd.DataFrame(rows)


def _pca_points(features: pd.DataFrame, feature_cols: list[str], cap: int, seed: int):
    from sklearn.preprocessing import StandardScaler

    sampled = deterministic_sample(features, cap, seed, stratify="angle")
    x = sampled[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = x.notna().all(axis=1)
    x = StandardScaler().fit_transform(x.loc[keep].to_numpy(dtype=float))
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    points = centered @ vt[:2].T
    return points, sampled.loc[keep, "angle"].astype(str).to_numpy()


def _proton_relation(proton: pd.DataFrame, proton7: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for angle in EXPECTED_PROTON_ANGLES:
        full_count = len(proton[proton["angle_value"] == float(angle)])
        c7_count = len(proton7[proton7["angle_value"] == float(angle)]) if not proton7.empty else 0
        rows.append(
            {
                "angle": angle,
                "num_full_proton_c": full_count,
                "num_proton_c7": c7_count,
                "included_in_training_mainline": angle in REGULAR_PROTON_ANGLES,
                "relation_note": "derived_from_full_Proton_C" if angle in REGULAR_PROTON_ANGLES else "held_out_for_near_vertical_analysis",
            }
        )
    return pd.DataFrame(rows)


def _group_summary_regular_near(proton: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for name, angles in [("regular_10_70", REGULAR_PROTON_ANGLES), ("near_vertical_80_90", NEAR_VERTICAL_ANGLES)]:
        group = proton[proton["angle_value"].isin([float(a) for a in angles])]
        row = {"group": name, "angles": ",".join(map(str, angles)), "num_samples": len(group)}
        for feature in ["active_count", "active_sum", "bbox_area", "pca_major_axis", "central_energy_ratio_r1"]:
            row[f"{feature}_median"] = group[feature].median()
            row[f"{feature}_iqr"] = group[feature].quantile(0.75) - group[feature].quantile(0.25)
        frames.append(row)
    return pd.DataFrame(frames)


def main() -> int:
    args = parse_args()
    layout = make_output_layout(args.output_root)
    figures_dir = layout.root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        layout,
        {
            "script": "scripts/analyze_datasets.py",
            "data_root": args.data_root,
            "dataset_roots": args.dataset_root,
            "datasets": args.datasets,
            "sample_cap_plot": args.sample_cap_plot,
            "seed": args.seed,
            "split_dir": args.split_dir,
            "reuse_feature_cache": args.reuse_feature_cache,
        },
    )
    table_registry: list[tuple[str, str, str, pd.DataFrame]] = []
    figures: list[Path] = []
    tables: list[Path] = []

    dataset_roots = _dataset_root_map(args.dataset_root)
    index_df = _scan_all(args.data_root, args.datasets, dataset_roots)
    split_distribution = _split_rows(args.split_dir, args.datasets)
    split_lookups = {dataset: _split_lookup(split_distribution, args.split_dir, dataset) for dataset in args.datasets}

    features = _load_feature_cache(layout.root) if args.reuse_feature_cache else pd.DataFrame()
    if features.empty:
        feature_frames = []
        for dataset in args.datasets:
            lookup = split_lookups.get(dataset, {})
            feature_frames.append(extract_feature_table(index_df, dataset=dataset, split_lookup=lookup))
        features = pd.concat([f for f in feature_frames if not f.empty], ignore_index=True) if feature_frames else pd.DataFrame()
    if "Proton_C" in set(features["dataset"]) and "Proton_C_7" not in set(features["dataset"]):
        proton7_from_full = features[
            (features["dataset"] == "Proton_C")
            & (features["modality"] == "ToT")
            & (features["angle_value"].isin([float(a) for a in REGULAR_PROTON_ANGLES]))
        ].copy()
        proton7_from_full["dataset"] = "Proton_C_7"
        features = pd.concat([features, proton7_from_full], ignore_index=True)

    inventory = _inventory_from_features(features, index_df)
    csv, md = _write(layout, inventory, "00_dataset_inventory", table_registry, "表 3-1 数据集完整性审计", "清洗后最终数据集的样本规模、shape 与数值质量统计。")
    tables.extend([csv, md])
    alpha_pairing = _alpha_pairing(index_df)
    csv, md = _write(layout, alpha_pairing, "01_alpha_pairing_audit", table_registry, "Alpha_100 ToT/ToA 配对审计", "用于确认 Alpha 双模态样本是否一一对应。")
    tables.extend([csv, md])
    csv, md = _write(layout, split_distribution, "02_split_distribution", table_registry, "split 分布审计", "统计已有 split manifest 中 train/val/test 的角度分布。")
    tables.extend([csv, md])

    thresholds = _cleaning_thresholds(features)
    csv, md = _write(layout, thresholds, "03_cleaning_thresholds", table_registry, "数据清洗阈值反推表", "原始候选事件与清洗日志缺失，阈值范围由最终清洗后数据集反推。")
    tables.extend([csv, md])
    missing = (
        "# Cleaning Information Missing Audit\n\n"
        "- 未找到 raw candidate event 文件。\n"
        "- 未找到 cleaning threshold config。\n"
        "- 当前分析仅针对清洗后的最终模型输入数据集。\n"
        "- Alpha 使用统一清洗阈值，Proton/C 使用每角度单独阈值；由于原始数据缺失，本报告仅从最终分布反推阈值范围。\n"
        "- 论文中应限定为“清洗后数据集的统计分析”。\n"
    )
    write_markdown(layout.root / "03_cleaning_info_missing.md", missing)

    for dataset, stem in [("Alpha_100", "04_event_features_alpha_100"), ("Proton_C", "04_event_features_proton_c"), ("Proton_C_7", "04_event_features_proton_c_7")]:
        subset = features[features["dataset"] == dataset]
        path = layout.root / f"{stem}.csv"
        subset.to_csv(path, index=False, encoding="utf-8-sig")
        tables.append(path)

    sample_counts = _sample_counts(features)
    csv, md = _write(layout, sample_counts, "05_sample_count_by_angle", table_registry, "表 3-2 每角度样本数", "按数据集、角度和模态统计样本数与类别占比。")
    tables.extend([csv, md])
    for dataset, counts_name in [("Alpha_100", "05_sample_count_alpha"), ("Proton_C", "14_proton_full_angle_count")]:
        counts = class_counts(index_df, dataset)
        if not counts.empty:
            png, pdf = plot_class_counts(counts, figures_dir / counts_name, f"{dataset} Sample Count by Angle")
            figures.extend([png, pdf])

    reps = _representatives(features)
    csv, md = _write(layout, reps, "06_representative_samples", table_registry, "代表性样本自动选择表", "medoid 与 active_sum 分位样本均由固定规则自动选择。")
    tables.extend([csv, md])
    for data, name, title in [
        (reps[(reps["dataset"] == "Alpha_100") & (reps["modality"] == "ToT")], "06_alpha_tot_representative_grid", "Alpha ToT Representative Samples"),
        (reps[(reps["dataset"] == "Alpha_100") & (reps["modality"] == "ToA")], "06_alpha_toa_representative_grid", "Alpha ToA Representative Samples"),
        (reps[(reps["dataset"] == "Proton_C") & (reps["modality"] == "ToT")], "06_proton_tot_representative_grid", "Proton/C ToT Representative Samples"),
    ]:
        png, pdf = plot_representative_grid(data, figures_dir / name, title, max_cols=6)
        figures.extend([png, pdf])
    alpha_tot = features[(features["dataset"] == "Alpha_100") & (features["modality"] == "ToT")]
    alpha_toa = features[(features["dataset"] == "Alpha_100") & (features["modality"] == "ToA")]
    proton = features[(features["dataset"] == "Proton_C") & (features["modality"] == "ToT")]
    proton7 = features[(features["dataset"] == "Proton_C_7") & (features["modality"] == "ToT")]

    proton_shape_audit = _proton_input_shape_audit(features)
    csv, md = _write(layout, proton_shape_audit, "proton_input_shape_audit", table_registry, "Proton_C / Proton_C_7 输入尺寸审计", "核实文件保存尺寸、Dataset loader 输出尺寸和训练模型有效输入尺寸。")
    tables.extend([csv, md])

    alpha_negative_audit = _alpha_toa_negative_audit(alpha_toa)
    csv, md = _write(layout, alpha_negative_audit, "alpha_toa_negative_audit", table_registry, "Alpha_100 ToA 负值样本审计", "定位 raw ToA 负值样本、对应 ToT 文件和非零值分布。")
    tables.extend([csv, md])

    proton_angle_audit = _proton_angle_consistency_audit(features, split_distribution)
    csv, md = _write(layout, proton_angle_audit, "proton_angle_consistency_audit", table_registry, "Proton_C 角度一致性审计", "比较当前 inventory 与 full Proton split 中的角度，识别 legacy split 残留。")
    tables.extend([csv, md])

    if not alpha_tot.empty:
        png, pdf = plot_mean_or_occupancy_grid(alpha_tot, figures_dir / "07_alpha_mean_tot_heatmap", "Alpha Mean ToT Heatmap", mode="mean")
        figures.extend([png, pdf])
        png, pdf = plot_mean_or_occupancy_grid(alpha_tot, figures_dir / "07_alpha_occupancy_heatmap", "Alpha Occupancy Probability", mode="occupancy")
        figures.extend([png, pdf])
    if not alpha_toa.empty:
        png, pdf = plot_mean_or_occupancy_grid(alpha_toa, figures_dir / "07_alpha_mean_toa_heatmap", "Alpha Mean ToA Heatmap", mode="mean", cmap="magma")
        figures.extend([png, pdf])
    if not proton.empty:
        png, pdf = plot_mean_or_occupancy_grid(proton, figures_dir / "07_proton_mean_tot_heatmap", "Proton/C Mean ToT Heatmap", mode="mean")
        figures.extend([png, pdf])
        png, pdf = plot_mean_or_occupancy_grid(proton, figures_dir / "07_proton_occupancy_heatmap", "Proton/C Occupancy Probability", mode="occupancy")
        figures.extend([png, pdf])

    for data, suffix, title in [(alpha_tot, "alpha", "Alpha"), (proton, "proton_full", "Proton/C Full"), (proton7, "proton_c7", "Proton/C 7-Class")]:
        if data.empty:
            continue
        sampled = deterministic_sample(data, args.sample_cap_plot, args.seed)
        for features_set, label in [(SIZE_FEATURES, "size"), (INTENSITY_FEATURES, "intensity"), (GEOMETRY_FEATURES, "geometry")]:
            png, pdf = plot_feature_panels(sampled, features_set, figures_dir / f"08{label[0]}_{label}_features_{suffix}", f"{title} {label.title()} Features")
            figures.extend([png, pdf])
    if not alpha_toa.empty:
        png, pdf = plot_feature_panels(deterministic_sample(alpha_toa, args.sample_cap_plot, args.seed), TOA_FEATURES, figures_dir / "08d_toa_features_alpha", "Alpha ToA Features")
        figures.extend([png, pdf])
    if not alpha_tot.empty:
        png, pdf = plot_feature_scatter(deterministic_sample(alpha_tot, args.sample_cap_plot, args.seed), figures_dir / "09_alpha_active_count_vs_sum", "Alpha Active Pixels vs Total ToT")
        figures.extend([png, pdf])
    if not proton.empty:
        png, pdf = plot_feature_scatter(deterministic_sample(proton, args.sample_cap_plot, args.seed), figures_dir / "09_proton_active_count_vs_sum", "Proton/C Active Pixels vs Total ToT")
        figures.extend([png, pdf])

    alpha_class = _class_summary_alpha(alpha_tot, alpha_toa, split_distribution)
    csv, md = _write(layout, alpha_class, "10_alpha_class_summary", table_registry, "Alpha_100 类别统计表", "汇总 Alpha 各角度样本数、split 与核心特征。")
    tables.extend([csv, md])
    alpha_scale = feature_summary_by_angle(alpha_toa, ["toa_span", "toa_std", "toa_min", "toa_max"]) if not alpha_toa.empty else pd.DataFrame()
    csv, md = _write(layout, alpha_scale, "11_alpha_modality_scale_summary", table_registry, "Alpha ToT/ToA 尺度统计", "用于解释 ToT 与 raw ToA 数值尺度差异。")
    tables.extend([csv, md])
    if not alpha_toa.empty:
        png, pdf = plot_box_by_angle(alpha_toa, "toa_span", figures_dir / "11_alpha_tot_toa_value_ranges", "Alpha ToA Span by Angle")
        figures.extend([png, pdf])
    alpha_assoc = _association(alpha_tot, [f for f in SIZE_FEATURES + INTENSITY_FEATURES + GEOMETRY_FEATURES if f in alpha_tot.columns])
    csv, md = _write(layout, alpha_assoc, "12_alpha_feature_angle_association", table_registry, "Alpha 特征与角度关联", "相关性和 Kruskal 统计仅作描述，论文应结合效应量与重叠度。")
    tables.extend([csv, md])
    if not alpha_tot.empty:
        cols = numeric_feature_columns(alpha_tot)
        points, labels = _pca_points(alpha_tot, cols, args.sample_cap_plot, args.seed)
        png, pdf = plot_embedding(points, labels, figures_dir / "13_alpha_pca_features", "PCA of Alpha Handcrafted Features")
        figures.extend([png, pdf])

    proton_full = feature_summary_by_angle(proton, ["active_count", "active_sum", "bbox_area", "pca_major_axis", "central_energy_ratio_r1"])
    csv, md = _write(layout, proton_full, "14_proton_full_angle_summary", table_registry, "Proton_C 全角度统计", "覆盖全量 Proton_C 角度；缺失角度应在报告中说明。")
    tables.extend([csv, md])
    relation = _proton_relation(proton, proton7)
    csv, md = _write(layout, relation, "15_proton_c7_relation_to_full", table_registry, "Proton_C_7 与全量 Proton_C 关系", "说明训练主线 10-70 deg 与近垂直极限分析对象的关系。")
    tables.extend([csv, md])
    missing_training_angles = relation[(relation["included_in_training_mainline"]) & (relation["num_full_proton_c"] == 0)]["angle"].tolist()
    if missing_training_angles:
        write_markdown(
            layout.root / "15_proton_c7_missing_angles.md",
            "# Proton_C_7 Missing Angle Audit\n\n"
            f"文档定义的 Proton_C_7 角度为 {REGULAR_PROTON_ANGLES}，但全量 Proton_C 当前缺少这些角度：{missing_training_angles}。\n\n"
            "请确认是数据未同步、角度命名差异，还是历史实验实际使用了不同角度集合。\n",
        )
    if not proton.empty:
        trend_rows = []
        for angle, group in proton.groupby("angle"):
            row = {"angle": float(angle)}
            for feature in ["active_count", "active_sum", "bbox_area", "pca_major_axis", "central_energy_ratio_r1"]:
                row[f"{feature}_median"] = group[feature].median()
                row[f"{feature}_q25"] = group[feature].quantile(0.25)
                row[f"{feature}_q75"] = group[feature].quantile(0.75)
            trend_rows.append(row)
        trend = pd.DataFrame(trend_rows).sort_values("angle")
        csv, md = _write(layout, trend, "16_proton_feature_trends_vs_angle", table_registry, "Proton/C 特征随角度趋势", "展示 10-70 deg 与近垂直区间的趋势差异。")
        tables.extend([csv, md])
        png, pdf = plot_feature_panels(deterministic_sample(proton, args.sample_cap_plot, args.seed), ["active_count", "active_sum", "bbox_area", "pca_major_axis", "central_energy_ratio_r1"], figures_dir / "16_proton_feature_trends_vs_angle", "Proton/C Feature Trends by Angle", kind="box")
        figures.extend([png, pdf])
    regular_near = _group_summary_regular_near(proton)
    csv, md = _write(layout, regular_near, "17_proton_regular_vs_near_vertical_summary", table_registry, "Proton/C 常规角度与近垂直角度对比", "将 10-70 deg 角度识别与 80-90 deg 分辨极限拆成两个问题。")
    tables.extend([csv, md])
    if not proton.empty:
        comparison = proton.copy()
        comparison["angle_group"] = np.where(comparison["angle_value"].isin([float(a) for a in REGULAR_PROTON_ANGLES]), "Regular 10-70", "Near-Vertical 80-90")
        png, pdf = plot_feature_panels(deterministic_sample(comparison, args.sample_cap_plot, args.seed), ["active_count", "active_sum", "bbox_area", "pca_axis_ratio"], figures_dir / "17_regular_vs_near_vertical_feature_overlap", "Regular vs Near-Vertical Feature Overlap", kind="box")
        figures.extend([png, pdf])

    png, pdf = plot_preprocessing_pipeline(figures_dir / "03_data_processing_pipeline")
    figures.extend([png, pdf])

    try:
        workbook_path = write_analysis_workbook(table_registry, layout.root / "analysis_tables.xlsx", title="Timepix 数据集分析统计表")
    except PermissionError:
        fallback = layout.root / "analysis_tables_refresh.xlsx"
        workbook_path = write_analysis_workbook(table_registry, fallback, title="Timepix 数据集分析统计表")
        print(f"Warning: analysis_tables.xlsx is locked; wrote fallback workbook: {workbook_path}")
    else:
        print(f"Wrote analysis workbook: {workbook_path}")
    report = dataset_report(layout.root, inventory, figures, tables)
    report += (
        "\n## Cleaning Caveat\n\n"
        "Proton/C 的清洗阈值是在每个角度内部设定的，目的是提高目标粒子事件纯度。清洗后的 active_count 和 active_sum 分布会受到阈值影响，因此这些分布主要反映最终模型输入数据集的统计特征，而不完全等同于未筛选原始响应的物理分布。\n"
    )
    write_markdown(layout.root / "dataset_analysis_report.md", report)
    print(f"Wrote dataset analysis to {layout.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
