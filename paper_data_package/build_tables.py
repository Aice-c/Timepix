from __future__ import annotations

import csv
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "paper_data_package"
OUTPUTS = ROOT / "outputs"
EXPERIMENTS = OUTPUTS / "experiments"


MAIN_FIELDS = [
    "experiment_id",
    "method_id",
    "method_label",
    "dataset",
    "particle",
    "modalities",
    "role",
    "stage_type",
    "paper_section",
    "selection_metric",
    "selection_basis",
    "include_level",
    "n_runs",
    "seeds",
    "model",
    "fusion_mode",
    "loss",
    "label_encoding",
    "handcrafted_features",
    "source_file",
    "val_accuracy_mean",
    "val_accuracy_std",
    "val_mae_argmax_mean",
    "val_mae_argmax_std",
    "val_p90_error_mean",
    "val_p90_error_std",
    "val_macro_f1_mean",
    "val_macro_f1_std",
    "val_high_angle_macro_f1_mean",
    "val_high_angle_macro_f1_std",
    "val_far_error_rate_abs_ge_20_mean",
    "val_far_error_rate_abs_ge_20_std",
    "test_accuracy_mean",
    "test_accuracy_std",
    "test_mae_argmax_mean",
    "test_mae_argmax_std",
    "test_p90_error_mean",
    "test_p90_error_std",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "test_high_angle_macro_f1_mean",
    "test_high_angle_macro_f1_std",
    "test_far_error_rate_abs_ge_20_mean",
    "test_far_error_rate_abs_ge_20_std",
    "validation_note",
    "test_note",
    "notes",
]

RUN_FIELDS = [
    "experiment_id",
    "method_id",
    "method_label",
    "seed",
    "dataset",
    "particle",
    "modalities",
    "model",
    "fusion_mode",
    "loss",
    "label_encoding",
    "handcrafted_features",
    "best_epoch",
    "stopped_epoch",
    "max_epochs",
    "early_stopped",
    "val_accuracy",
    "val_mae_argmax",
    "val_p90_error",
    "val_macro_f1",
    "val_high_angle_macro_f1",
    "val_far_error_rate_abs_ge_20",
    "test_accuracy",
    "test_mae_argmax",
    "test_p90_error",
    "test_macro_f1",
    "test_high_angle_macro_f1",
    "test_far_error_rate_abs_ge_20",
    "source_file",
    "notes",
]

PER_CLASS_FIELDS = [
    "experiment_id",
    "method_id",
    "method_label",
    "seed",
    "dataset",
    "split",
    "class_index",
    "class_angle",
    "n",
    "precision",
    "recall",
    "f1",
    "source_file",
    "notes",
]


METRICS = [
    "val_accuracy",
    "val_mae_argmax",
    "val_p90_error",
    "val_macro_f1",
    "val_high_angle_macro_f1",
    "val_far_error_rate_abs_ge_20",
    "test_accuracy",
    "test_mae_argmax",
    "test_p90_error",
    "test_macro_f1",
    "test_high_angle_macro_f1",
    "test_far_error_rate_abs_ge_20",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def long_path(path: Path) -> str:
    """Return a Windows long-path-safe string when needed."""
    resolved = str(path.resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.lstrip("\\")
    return "\\\\?\\" + resolved


def file_exists(path: Path) -> bool:
    if path.exists():
        return True
    try:
        with open(long_path(path), "rb"):
            return True
    except OSError:
        return False


def read_json(path: Path) -> dict:
    with open(long_path(path), "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt(value: object) -> str:
    number = to_float(value)
    if number is None:
        return "" if value is None else str(value)
    return f"{number:.10g}"


def mean_std(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return fmt(mean), fmt(std)


def summarize_runs(rows: list[dict[str, object]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for metric in METRICS:
        values = [v for v in (to_float(row.get(metric)) for row in rows) if v is not None]
        out[f"{metric}_mean"], out[f"{metric}_std"] = mean_std(values)
    seeds = sorted({str(row.get("seed", "")).strip() for row in rows if str(row.get("seed", "")).strip()})
    out["n_runs"] = str(len(rows))
    out["seeds"] = ";".join(seeds)
    return out


def index_rows() -> dict[str, dict[str, str]]:
    rows = read_csv(PKG / "00_experiment_index.csv")
    return {row["experiment_id"]: row for row in rows}


def base_from_index(index: dict[str, dict[str, str]], experiment_id: str) -> dict[str, str]:
    info = index[experiment_id]
    return {
        "experiment_id": experiment_id,
        "dataset": info.get("dataset", ""),
        "particle": info.get("particle", ""),
        "modalities": info.get("modalities", ""),
        "role": info.get("role", ""),
        "stage_type": info.get("stage_type", ""),
        "paper_section": info.get("paper_section", ""),
        "selection_metric": info.get("selection_metric", ""),
        "include_level": info.get("include_level", ""),
        "notes": info.get("notes", ""),
    }


def local_path(path_text: str) -> Path:
    text = path_text.strip()
    if text.startswith("/root/Timepix/"):
        return ROOT / text.removeprefix("/root/Timepix/")
    return ROOT / text


def seed_from_text(text: str) -> str:
    matches = re.findall(r"seed[-_](\d+)", text)
    return matches[-1] if matches else ""


def angle_values(dataset: str) -> list[float]:
    return [float(value) for value in class_angles(dataset)]


def derive_high_angle_macro_f1(split_data: dict, dataset: str) -> object:
    value = split_data.get("high_angle_macro_f1")
    if value not in (None, ""):
        return value
    angles = angle_values(dataset)
    per_class = split_data.get("per_class", [])
    values = [
        to_float(item.get("f1"))
        for item, angle in zip(per_class, angles)
        if angle >= 45.0
    ]
    values = [value for value in values if value is not None]
    if not values:
        return ""
    return statistics.mean(values)


def derive_far_error_rate(split_data: dict, dataset: str) -> object:
    value = split_data.get("far_error_rate_abs_ge_20")
    if value not in (None, ""):
        return value
    angles = angle_values(dataset)
    confusion = split_data.get("confusion_matrix", [])
    if not angles or not confusion:
        return ""
    total = 0
    far = 0
    for i, row in enumerate(confusion):
        for j, count in enumerate(row):
            if i >= len(angles) or j >= len(angles):
                continue
            n = int(count)
            total += n
            if abs(angles[i] - angles[j]) >= 20.0:
                far += n
    return far / total if total else ""


def split_metrics(metrics: dict, split: str, dataset: str = "") -> dict[str, object]:
    data = metrics.get(split, {})
    prefix = "val" if split == "validation" else split
    return {
        f"{prefix}_accuracy": data.get("accuracy", ""),
        f"{prefix}_mae_argmax": data.get("mae_argmax", ""),
        f"{prefix}_p90_error": data.get("p90_error", ""),
        f"{prefix}_macro_f1": data.get("macro_f1", ""),
        f"{prefix}_high_angle_macro_f1": derive_high_angle_macro_f1(data, dataset),
        f"{prefix}_far_error_rate_abs_ge_20": derive_far_error_rate(data, dataset),
    }


def a2_runs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((EXPERIMENTS / "a2_best_3seed").glob("*/metrics.json")):
        metrics = read_json(path)
        seed = seed_from_text(str(path.parent))
        row = {
            "experiment_id": "A2-best",
            "method_id": "alpha_tot_resnet18_no_maxpool_baseline",
            "method_label": "Alpha ToT baseline: ResNet18 no maxpool",
            "seed": seed,
            "dataset": "Alpha_100",
            "particle": "alpha",
            "modalities": "ToT",
            "model": "resnet18_no_maxpool",
            "fusion_mode": "none",
            "loss": "cross_entropy",
            "label_encoding": "onehot",
            "handcrafted_features": "",
            "best_epoch": metrics.get("best_epoch", ""),
            "stopped_epoch": metrics.get("stopped_epoch", ""),
            "max_epochs": metrics.get("max_epochs", ""),
            "early_stopped": metrics.get("early_stopped", ""),
            "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "notes": "formal Alpha ToT-only baseline",
        }
        row.update(split_metrics(metrics, "validation", "Alpha_100"))
        row.update(split_metrics(metrics, "test", "Alpha_100"))
        rows.append(row)
    return rows


def row_from_run_csv(
    source: dict[str, str],
    experiment_id: str,
    method_id: str,
    method_label: str,
    source_file: Path,
    notes: str = "",
) -> dict[str, object]:
    row = {
        "experiment_id": experiment_id,
        "method_id": method_id,
        "method_label": method_label,
        "seed": source.get("seed", ""),
        "dataset": source.get("dataset", ""),
        "particle": source.get("particle", ""),
        "modalities": source.get("modalities", ""),
        "model": source.get("model", ""),
        "fusion_mode": source.get("fusion_mode", ""),
        "loss": source.get("loss", ""),
        "label_encoding": source.get("label_encoding", ""),
        "handcrafted_features": source.get("handcrafted_features", ""),
        "best_epoch": source.get("best_epoch", ""),
        "stopped_epoch": source.get("stopped_epoch", ""),
        "max_epochs": source.get("max_epochs", ""),
        "early_stopped": source.get("early_stopped", ""),
        "source_file": str(source_file.relative_to(ROOT)).replace("\\", "/"),
        "notes": notes,
    }
    json_metrics: dict[str, object] = {}
    metric_path = metrics_path_from_source(source)
    if metric_path:
        try:
            metric_data = read_json(metric_path)
            dataset = source.get("dataset", "")
            json_metrics.update(split_metrics(metric_data, "validation", dataset))
            json_metrics.update(split_metrics(metric_data, "test", dataset))
        except (OSError, json.JSONDecodeError):
            json_metrics = {}
    for metric in METRICS:
        row[metric] = source.get(metric, "") or json_metrics.get(metric, "")
    return row


def method_id_from_features(features: str) -> str:
    if features == "active_pixel_count;bbox_fill_ratio;ToT_density;ToA_span;ToA_major_axis_corr_abs":
        return "main_5feat"
    if features == "ToA_span;ToA_major_axis_corr_abs":
        return "toa_only_diag"
    if features == "active_pixel_count;bbox_fill_ratio":
        return "geometry_lowcorr"
    if features == "active_pixel_count;bbox_fill_ratio;ToT_density":
        return "geometry_plus_tot_density"
    return re.sub(r"[^A-Za-z0-9]+", "_", features).strip("_").lower() or "unknown_features"


def method_label_from_features(features: str) -> str:
    labels = {
        "main_5feat": "5 handcrafted physical features",
        "toa_only_diag": "ToA-only diagnostic handcrafted features",
        "geometry_lowcorr": "Geometry low-correlation features",
        "geometry_plus_tot_density": "Geometry + ToT density features",
    }
    return labels.get(method_id_from_features(features), features)


def metrics_path_from_source(source: dict[str, object]) -> Path | None:
    exp_dir = str(source.get("experiment_dir", "")).strip()
    if not exp_dir:
        return None
    path = local_path(exp_dir) / "metrics.json"
    return path if file_exists(path) else None


def runs_from_csv_group(
    path: Path,
    experiment_id: str,
    group_by: str,
    label_prefix: str = "",
    notes: str = "",
) -> list[dict[str, object]]:
    rows = []
    for src in read_csv(path):
        key = src.get(group_by, "")
        method_id = method_id_from_features(key) if group_by == "handcrafted_features" else key
        method_label = method_label_from_features(key) if group_by == "handcrafted_features" else f"{label_prefix}{key}"
        rows.append(row_from_run_csv(src, experiment_id, method_id, method_label, path, notes))
    return rows


def loss_method_id(source: dict[str, object]) -> str:
    loss = str(source.get("loss", "")).strip()
    label = str(source.get("label_encoding", "")).strip()
    gaussian_sigma = str(source.get("gaussian_sigma", "")).strip()
    expected_mae_weight = str(source.get("expected_mae_weight", "")).strip()
    emd_weight = str(source.get("emd_weight", "")).strip()
    if loss == "cross_entropy" and label == "onehot":
        return "cross_entropy_onehot"
    if loss == "cross_entropy" and label == "gaussian":
        return f"cross_entropy_gaussian_sigma_{gaussian_sigma}"
    if loss == "ce_expected_mae":
        return f"ce_expected_mae_lambda_{expected_mae_weight}"
    if loss == "ce_emd":
        return f"ce_emd_lambda_{emd_weight}"
    return re.sub(r"[^A-Za-z0-9.]+", "_", loss or "unknown_loss").strip("_")


def loss_method_label(source: dict[str, object]) -> str:
    loss = str(source.get("loss", "")).strip()
    label = str(source.get("label_encoding", "")).strip()
    gaussian_sigma = str(source.get("gaussian_sigma", "")).strip()
    expected_mae_weight = str(source.get("expected_mae_weight", "")).strip()
    emd_weight = str(source.get("emd_weight", "")).strip()
    if loss == "cross_entropy" and label == "onehot":
        return "CE one-hot"
    if loss == "cross_entropy" and label == "gaussian":
        return f"Gaussian soft CE, sigma={gaussian_sigma}"
    if loss == "ce_expected_mae":
        return f"CE + ExpectedMAE, lambda={expected_mae_weight}"
    if loss == "ce_emd":
        return f"CE + EMD, lambda={emd_weight}"
    return loss or "unknown loss"


def runs_from_loss_csv(
    path: Path,
    experiment_id: str,
    notes: str,
) -> list[dict[str, object]]:
    rows = []
    if not path.exists():
        return rows
    for src in read_csv(path):
        rows.append(row_from_run_csv(
            src,
            experiment_id,
            loss_method_id(src),
            loss_method_label(src),
            path,
            notes,
        ))
    return rows


def a7_runs() -> list[dict[str, object]]:
    path = OUTPUTS / "a7_final_gmu_main5feat_gated_3seed_runs.csv"
    rows = []
    if not path.exists():
        return rows
    for src in read_csv(path):
        rows.append(row_from_run_csv(
            src,
            "A7",
            "gmu_aux_main_5feat_gated",
            "GMU_aux + main_5feat gated",
            path,
            "A7 final GMU component check; handcrafted features are not selected by validation",
        ))
    return rows


def high_angle_from_by_class(path: Path) -> dict[tuple[str, str, str], float]:
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    if not path.exists():
        return {}
    for src in read_csv(path):
        split = src.get("split", "")
        if split not in {"val", "test"}:
            continue
        angle = to_float(src.get("class_angle", ""))
        f1 = to_float(src.get("f1", ""))
        if angle is None or f1 is None or angle < 45.0:
            continue
        strategy = src.get("strategy", src.get("method_id", ""))
        values[(src.get("seed", ""), strategy, split)].append(f1)
    return {key: statistics.mean(items) for key, items in values.items() if items}


def a4b_selected_high_angle_stats(stem: str) -> dict[str, str]:
    val_values: list[float] = []
    test_values: list[float] = []
    for seed in ("42", "43", "44"):
        summary_path = OUTPUTS / f"{stem}_seed{seed}_summary.csv"
        by_class = high_angle_from_by_class(OUTPUTS / f"{stem}_seed{seed}_by_class.csv")
        selected = [row for row in read_csv(summary_path) if row.get("selected_by_val") == "True"]
        if not selected:
            continue
        strategy = selected[0].get("strategy", "")
        val = by_class.get((seed, strategy, "val"))
        test = by_class.get((seed, strategy, "test"))
        if val is not None:
            val_values.append(val)
        if test is not None:
            test_values.append(test)
    val_mean, val_std = mean_std(val_values)
    test_mean, test_std = mean_std(test_values)
    return {
        "val_high_angle_macro_f1_mean": val_mean,
        "val_high_angle_macro_f1_std": val_std,
        "test_high_angle_macro_f1_mean": test_mean,
        "test_high_angle_macro_f1_std": test_std,
    }


def a4b_modality_high_angle_stats(stem: str, method: str) -> dict[str, str]:
    val_values: list[float] = []
    test_values: list[float] = []
    selected_method = method.startswith("validation_selected")
    for seed in ("42", "43", "44"):
        path = OUTPUTS / f"{stem}_seed{seed}_by_class.csv"
        if not path.exists():
            continue
        split_values: dict[str, list[float]] = {"val": [], "test": []}
        for src in read_csv(path):
            split = src.get("split", "")
            if split not in split_values:
                continue
            angle = to_float(src.get("class_angle", ""))
            f1 = to_float(src.get("f1", ""))
            if angle is None or f1 is None or angle < 45.0:
                continue
            if selected_method:
                if str(src.get("selected_strategy", "")).lower() != "true":
                    continue
            elif src.get("strategy", "") != method:
                continue
            split_values[split].append(f1)
        if split_values["val"]:
            val_values.append(statistics.mean(split_values["val"]))
        if split_values["test"]:
            test_values.append(statistics.mean(split_values["test"]))
    val_mean, val_std = mean_std(val_values)
    test_mean, test_std = mean_std(test_values)
    return {
        "val_high_angle_macro_f1_mean": val_mean,
        "val_high_angle_macro_f1_std": val_std,
        "test_high_angle_macro_f1_mean": test_mean,
        "test_high_angle_macro_f1_std": test_std,
    }


def a4b_run_rows(path: Path, experiment_id: str, keep: set[str]) -> list[dict[str, object]]:
    rows = []
    by_class = high_angle_from_by_class(Path(str(path).replace("_summary.csv", "_by_class.csv")))
    for src in read_csv(path):
        strategy = src.get("strategy", "")
        if strategy not in keep:
            continue
        row = {
            "experiment_id": experiment_id,
            "method_id": strategy,
            "method_label": strategy,
            "seed": src.get("seed", ""),
            "dataset": "Alpha_100",
            "particle": "alpha",
            "modalities": "ToT+ToA",
            "model": "late_prediction_fusion",
            "fusion_mode": "selector" if experiment_id == "A4b-5" else "residual_selector",
            "loss": "cross_entropy",
            "label_encoding": "onehot",
            "handcrafted_features": "",
            "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "notes": "A4b prediction-level fusion summary; primary_only is A2 baseline reference",
        }
        for split in ("val", "test"):
            for metric in ("accuracy", "mae_argmax", "p90_error", "macro_f1"):
                row[f"{split}_{metric}"] = src.get(f"{split}_{metric}", "")
            row[f"{split}_high_angle_macro_f1"] = by_class.get((src.get("seed", ""), strategy, split), "")
        rows.append(row)
    return rows


def summary_row(
    index: dict[str, dict[str, str]],
    experiment_id: str,
    method_id: str,
    method_label: str,
    run_rows: list[dict[str, object]],
    selection_basis: str,
    source_file: str,
    validation_note: str,
    test_note: str = "test is report-only",
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = base_from_index(index, experiment_id)
    row.update(
        {
            "method_id": method_id,
            "method_label": method_label,
            "selection_basis": selection_basis,
            "source_file": source_file,
            "validation_note": validation_note,
            "test_note": test_note,
        }
    )
    row.update(summarize_runs(run_rows))
    if run_rows:
        first = run_rows[0]
        for field in ("model", "fusion_mode", "loss", "label_encoding", "handcrafted_features"):
            row[field] = first.get(field, "")
    if extra:
        row.update(extra)
    return row


def summary_row_from_mean_std(
    index: dict[str, dict[str, str]],
    path: Path,
    source: dict[str, str],
    experiment_id: str,
    method_id: str,
    method_label: str,
    selection_basis: str,
    validation_note: str,
    test_note: str = "test is report-only",
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = base_from_index(index, experiment_id)
    row.update(
        {
            "method_id": method_id,
            "method_label": method_label,
            "selection_basis": selection_basis,
            "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "n_runs": source.get("n_runs", ""),
            "seeds": source.get("seeds", ""),
            "model": source.get("model", ""),
            "fusion_mode": source.get("fusion_mode", ""),
            "loss": source.get("loss", ""),
            "label_encoding": source.get("label_encoding", ""),
            "handcrafted_features": source.get("handcrafted_features", ""),
            "validation_note": validation_note,
            "test_note": test_note,
        }
    )
    for field in MAIN_FIELDS:
        if field.endswith("_mean") or field.endswith("_std"):
            row[field] = source.get(field, "")
    if extra:
        row.update(extra)
    return row


def build_run_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.extend(a2_runs())
    rows.extend(a4b_run_rows(OUTPUTS / "a4b_5_gated_late_fusion_seed42_summary.csv", "A4b-5", {
        "primary_only",
        "candidate_only",
        "oracle",
        "a4b5d_class_aware_prob_train",
        "a4b5d_class_aware_prob_val-cv",
    }))
    for seed in ("43", "44"):
        path = OUTPUTS / f"a4b_5_gated_late_fusion_seed{seed}_summary.csv"
        rows.extend(a4b_run_rows(path, "A4b-5", {
            "primary_only",
            "candidate_only",
            "oracle",
            "a4b5d_class_aware_prob_train",
            "a4b5d_class_aware_prob_val-cv",
        }))
    for seed in ("42", "43", "44"):
        path = OUTPUTS / f"a4b_6_residual_gated_fusion_seed{seed}_summary.csv"
        rows.extend(a4b_run_rows(path, "A4b-6", {
            "primary_only",
            "candidate_only",
            "oracle",
            "a4b6b_per_class_beta_grid",
            "a4b6e_entropy_residual_t0p1_k5_b0p5",
        }))
    rows.extend(runs_from_csv_group(
        OUTPUTS / "a4c_end_to_end_bimodal_fusion_runs.csv",
        "A4c-1-3",
        "model",
        label_prefix="A4c ",
        notes="end-to-end bimodal fusion",
    ))
    rows.extend(a7_runs())
    rows.extend(runs_from_csv_group(
        OUTPUTS / "a5d_alpha_handcrafted_gated_3seed_runs.csv",
        "A5d",
        "handcrafted_features",
        notes="handcrafted feature three-seed verification",
    ))
    rows.extend(runs_from_loss_csv(
        OUTPUTS / "a6b_alpha_tot_ce_emd_0p02_3seed_runs.csv",
        "A6b",
        "Alpha ordered-loss three-seed verification; rejected by validation relative to A2 CE baseline",
    ))
    rows.extend(runs_from_csv_group(
        OUTPUTS / "b1_proton_c7_resnet18_tot_best_patience8_3seed_runs.csv",
        "B1-best",
        "model",
        label_prefix="Proton baseline ",
        notes="formal Proton_C_7 baseline",
    ))
    rows.extend(runs_from_loss_csv(
        OUTPUTS / "b3b_proton_c7_expected_mae_3seed_runs.csv",
        "B3b-main",
        "validation-selected ordered loss",
    ))
    rows.extend(runs_from_loss_csv(
        OUTPUTS / "b3b_proton_c7_ce_emd_optional_3seed_runs.csv",
        "B3b-optional",
        "optional ordered-loss control",
    ))
    return rows


def build_main_summary(index: dict[str, dict[str, str]], run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_exp_method: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        by_exp_method[(str(row["experiment_id"]), str(row["method_id"]))].append(row)

    rows.append(summary_row(
        index,
        "A2-best",
        "alpha_tot_resnet18_no_maxpool_baseline",
        "Alpha ToT baseline: ResNet18 no maxpool",
        by_exp_method[("A2-best", "alpha_tot_resnet18_no_maxpool_baseline")],
        "formal three-seed baseline",
        "outputs/experiments/a2_best_3seed/*/metrics.json",
        "baseline reference for Alpha experiments",
    ))

    for path, experiment_id, method_id, label, basis, note, stem, fusion_mode in [
        (
            OUTPUTS / "a4b_5_gated_late_fusion_mean_std.csv",
            "A4b-5",
            "validation_selected_selector",
            "A4b-5 validation-selected late fusion selector",
            "validation-selected selector within A4b-5",
            "best selector chosen by validation, not by test",
            "a4b_5_gated_late_fusion",
            "selector",
        ),
        (
            OUTPUTS / "a4b_6_residual_gated_fusion_mean_std.csv",
            "A4b-6",
            "validation_selected_residual_validation_grid",
            "A4b-6 validation-selected residual fusion",
            "validation-selected residual rule within A4b-6",
            "best residual selector chosen by validation, not by test",
            "a4b_6_residual_gated_fusion",
            "residual_selector",
        ),
    ]:
        for src in read_csv(path):
            if src.get("method") == method_id:
                extra = {
                    "model": "late_prediction_fusion",
                    "fusion_mode": fusion_mode,
                    "loss": "cross_entropy",
                    "label_encoding": "onehot",
                }
                extra.update(a4b_selected_high_angle_stats(stem))
                rows.append(summary_row_from_mean_std(index, path, src, experiment_id, method_id, label, basis, note, extra=extra))

    a4c_path = OUTPUTS / "a4c_end_to_end_bimodal_fusion_mean_std.csv"
    for src in read_csv(a4c_path):
        model = src.get("model", "")
        extra = summarize_runs(by_exp_method[("A4c-1-3", model)])
        rows.append(summary_row_from_mean_std(
            index,
            a4c_path,
            src,
            "A4c-1-3",
            model,
            f"A4c {model}",
            "three-seed end-to-end bimodal comparison; selection by validation",
            "A4c formal end-to-end bimodal comparison; do not choose by test",
            extra=extra,
        ))

    a7_path = OUTPUTS / "a7_final_gmu_main5feat_gated_3seed_mean_std.csv"
    if a7_path.exists():
        a7_src = read_csv(a7_path)[0]
        a7_extra = summarize_runs(by_exp_method[("A7", "gmu_aux_main_5feat_gated")])
        a7_extra["handcrafted_features"] = a7_src.get("handcrafted_features", "")
        rows.append(summary_row_from_mean_std(
            index,
            a7_path,
            a7_src,
            "A7",
            "gmu_aux_main_5feat_gated",
            "GMU_aux + main_5feat gated",
            "final GMU component check; selection by validation against A4c GMU",
            "A7 ties A4c GMU on Val Acc but worsens Val MAE; not selected for final Alpha model",
            extra=a7_extra,
        ))

    a5d_path = OUTPUTS / "a5d_alpha_handcrafted_gated_3seed_mean_std.csv"
    for src in read_csv(a5d_path):
        method = method_id_from_features(src.get("handcrafted_features", ""))
        runs = by_exp_method[("A5d", method)]
        extra = summarize_runs(runs)
        rows.append(summary_row_from_mean_std(
            index,
            a5d_path,
            src,
            "A5d",
            method,
            method_label_from_features(src.get("handcrafted_features", "")),
            "A5d internal comparison selected by validation only",
            "handcrafted feature ablation; test differences are report-only",
            extra=extra,
        ))

    a6b_path = OUTPUTS / "a6b_alpha_tot_ce_emd_0p02_3seed_mean_std.csv"
    if a6b_path.exists():
        a6b_src = read_csv(a6b_path)[0]
        a6b_extra = summarize_runs(by_exp_method[("A6b", "ce_emd_lambda_0.02")])
        rows.append(summary_row_from_mean_std(
            index,
            a6b_path,
            a6b_src,
            "A6b",
            "ce_emd_lambda_0.02",
            "CE + EMD, lambda=0.02",
            "A6a validation-selected tie-break candidate; A6b three-seed verification",
            "A6b is weaker than A2 CE baseline on validation; Alpha loss remains CE one-hot",
            extra=a6b_extra,
        ))

    b1_path = OUTPUTS / "b1_proton_c7_resnet18_tot_best_patience8_3seed_mean_std.csv"
    b1_src = read_csv(b1_path)[0]
    b1_runs = by_exp_method[("B1-best", "resnet18_no_maxpool")]
    b1_extra = summarize_runs(b1_runs)
    rows.append(summary_row_from_mean_std(
        index,
        b1_path,
        b1_src,
        "B1-best",
        "proton_tot_resnet18_no_maxpool_baseline",
        "Proton_C_7 ToT baseline: ResNet18 no maxpool",
        "formal three-seed baseline",
        "baseline reference for Proton_C_7 experiments",
        extra=b1_extra,
    ))

    for path, experiment_id, method_id, label, basis, note in [
        (
            OUTPUTS / "b3b_proton_c7_expected_mae_3seed_mean_std.csv",
            "B3b-main",
            "ce_expected_mae_lambda_0.05",
            "CE + ExpectedMAE, lambda=0.05",
            "selected from B3a by validation accuracy",
            "current recommended Proton ordered-loss result by validation",
        ),
        (
            OUTPUTS / "b3b_proton_c7_ce_emd_optional_3seed_mean_std.csv",
            "B3b-optional",
            "ce_emd_lambda_0.05",
            "CE + EMD, lambda=0.05",
            "optional control; close validation result",
            "strong optional ordered-loss control; not primary unless thesis chooses MAE emphasis",
        ),
    ]:
        src = read_csv(path)[0]
        rows.append(summary_row_from_mean_std(index, path, src, experiment_id, method_id, label, basis, note))
    return rows


def class_angles(dataset: str) -> list[str]:
    if dataset == "Alpha_100":
        return ["15", "30", "45", "60"]
    if dataset == "Proton_C_7":
        return ["10", "20", "30", "40", "45", "50", "60"]
    return []


def per_class_from_metrics(
    path: Path,
    experiment_id: str,
    method_id: str,
    method_label: str,
    dataset: str,
    seed: str,
    notes: str,
) -> list[dict[str, object]]:
    metrics = read_json(path)
    angles = class_angles(dataset)
    out = []
    for split_name, split_key in [("val", "validation"), ("test", "test")]:
        split_data = metrics.get(split_key, {})
        confusion = split_data.get("confusion_matrix", [])
        for item in split_data.get("per_class", []):
            idx = int(item["class_index"])
            n = sum(confusion[idx]) if idx < len(confusion) else ""
            out.append({
                "experiment_id": experiment_id,
                "method_id": method_id,
                "method_label": method_label,
                "seed": seed,
                "dataset": dataset,
                "split": split_name,
                "class_index": idx,
                "class_angle": angles[idx] if idx < len(angles) else "",
                "n": n,
                "precision": item.get("precision", ""),
                "recall": item.get("recall", ""),
                "f1": item.get("f1", ""),
                "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "notes": notes,
            })
    return out


def metrics_path_from_run(row: dict[str, object]) -> Path | None:
    exp_dir = str(row.get("experiment_dir", "")).strip()
    if not exp_dir:
        return None
    path = local_path(exp_dir) / "metrics.json"
    return path if file_exists(path) else None


def build_per_class_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in a2_runs():
        path = ROOT / str(run["source_file"])
        rows.extend(per_class_from_metrics(
            path,
            "A2-best",
            "alpha_tot_resnet18_no_maxpool_baseline",
            "Alpha ToT baseline: ResNet18 no maxpool",
            "Alpha_100",
            str(run["seed"]),
            "formal Alpha baseline",
        ))

    for experiment_id, by_class_path, keep in [
        ("A4b-5", OUTPUTS / "a4b_5_gated_late_fusion_seed42_by_class.csv", {"primary_only", "candidate_only", "oracle", "a4b5d_class_aware_prob_train", "a4b5d_class_aware_prob_val-cv"}),
        ("A4b-6", OUTPUTS / "a4b_6_residual_gated_fusion_seed42_by_class.csv", {"primary_only", "candidate_only", "oracle", "a4b6b_per_class_beta_grid", "a4b6e_entropy_residual_t0p1_k5_b0p5"}),
    ]:
        # Seed 42 path is replaced below for all seeds.
        stem = "a4b_5_gated_late_fusion" if experiment_id == "A4b-5" else "a4b_6_residual_gated_fusion"
        for seed in ("42", "43", "44"):
            path = OUTPUTS / f"{stem}_seed{seed}_by_class.csv"
            for src in read_csv(path):
                if src.get("strategy", "") not in keep or src.get("split") not in {"val", "test"}:
                    continue
                rows.append({
                    "experiment_id": experiment_id,
                    "method_id": src.get("strategy", ""),
                    "method_label": src.get("strategy", ""),
                    "seed": src.get("seed", ""),
                    "dataset": "Alpha_100",
                    "split": src.get("split", ""),
                    "class_index": src.get("class_index", ""),
                    "class_angle": src.get("class_angle", ""),
                    "n": src.get("n", ""),
                    "precision": src.get("precision", ""),
                    "recall": src.get("recall", ""),
                    "f1": src.get("f1", ""),
                    "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "notes": "A4b prediction-level fusion per-class result",
                })

    for path, experiment_id, group_by, label_prefix, dataset, notes in [
        (OUTPUTS / "a4c_end_to_end_bimodal_fusion_runs.csv", "A4c-1-3", "model", "A4c ", "Alpha_100", "A4c end-to-end bimodal"),
        (OUTPUTS / "b1_proton_c7_resnet18_tot_best_patience8_3seed_runs.csv", "B1-best", "model", "Proton baseline ", "Proton_C_7", "Proton baseline"),
        (OUTPUTS / "b3b_proton_c7_expected_mae_3seed_runs.csv", "B3b-main", "loss", "", "Proton_C_7", "B3b ExpectedMAE"),
        (OUTPUTS / "b3b_proton_c7_ce_emd_optional_3seed_runs.csv", "B3b-optional", "loss", "", "Proton_C_7", "B3b CE+EMD optional"),
    ]:
        for src in read_csv(path):
            metric_path = metrics_path_from_run(src)
            if not metric_path:
                continue
            key = src.get(group_by, "")
            if experiment_id.startswith("B3b"):
                method_id = loss_method_id(src)
                method_label = loss_method_label(src)
            else:
                method_id = key
                method_label = f"{label_prefix}{key}"
            rows.extend(per_class_from_metrics(
                metric_path,
                experiment_id,
                method_id,
                method_label,
                dataset,
                src.get("seed", ""),
                notes,
            ))

    for path, experiment_id, notes in [
        (OUTPUTS / "a6a_alpha_tot_ordinal_loss_seed42_runs.csv", "A6a", "A6a Alpha ordered-loss seed42 screening"),
        (OUTPUTS / "a6b_alpha_tot_ce_emd_0p02_3seed_runs.csv", "A6b", "A6b Alpha CE+EMD three-seed verification"),
    ]:
        if not path.exists():
            continue
        for src in read_csv(path):
            metric_path = metrics_path_from_run(src)
            if not metric_path:
                continue
            rows.extend(per_class_from_metrics(
                metric_path,
                experiment_id,
                loss_method_id(src),
                loss_method_label(src),
                "Alpha_100",
                src.get("seed", ""),
                notes,
            ))

    a7_path = OUTPUTS / "a7_final_gmu_main5feat_gated_3seed_runs.csv"
    for src in read_csv(a7_path) if a7_path.exists() else []:
        metric_path = metrics_path_from_run(src)
        if not metric_path:
            continue
        rows.extend(per_class_from_metrics(
            metric_path,
            "A7",
            "gmu_aux_main_5feat_gated",
            "GMU_aux + main_5feat gated",
            "Alpha_100",
            src.get("seed", ""),
            "A7 final GMU component check per-class result",
        ))

    # A5d main_5feat metrics are not present in the pulled folder; include available ToA-only runs.
    for src in read_csv(OUTPUTS / "a5d_alpha_handcrafted_gated_3seed_runs.csv"):
        method = method_id_from_features(src.get("handcrafted_features", ""))
        metric_path = metrics_path_from_run(src)
        if not metric_path:
            continue
        rows.extend(per_class_from_metrics(
            metric_path,
            "A5d",
            method,
            method_label_from_features(src.get("handcrafted_features", "")),
            "Alpha_100",
            src.get("seed", ""),
            "A5d per-class only for runs whose metrics.json exists locally",
        ))
    return rows


def build_error_rows(main_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = [
        "experiment_id",
        "method_id",
        "method_label",
        "dataset",
        "selection_basis",
        "val_mae_argmax_mean",
        "val_mae_argmax_std",
        "val_p90_error_mean",
        "val_p90_error_std",
        "val_high_angle_macro_f1_mean",
        "val_high_angle_macro_f1_std",
        "val_far_error_rate_abs_ge_20_mean",
        "val_far_error_rate_abs_ge_20_std",
        "test_mae_argmax_mean",
        "test_mae_argmax_std",
        "test_p90_error_mean",
        "test_p90_error_std",
        "test_high_angle_macro_f1_mean",
        "test_high_angle_macro_f1_std",
        "test_far_error_rate_abs_ge_20_mean",
        "test_far_error_rate_abs_ge_20_std",
        "source_file",
        "notes",
    ]
    return [{field: row.get(field, "") for field in fields} for row in main_rows]


def build_modality_gate_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path, experiment_id in [
        (OUTPUTS / "a4b_4e_rule_selector_mean_std.csv", "A4b-candidate"),
        (OUTPUTS / "a4b_5_gated_late_fusion_mean_std.csv", "A4b-5"),
        (OUTPUTS / "a4b_6_residual_gated_fusion_mean_std.csv", "A4b-6"),
    ]:
        if not path.exists():
            continue
        for src in read_csv(path):
            row = {
                "experiment_id": experiment_id,
                "method_id": src.get("method", ""),
                "method_label": src.get("method", ""),
                "dataset": "Alpha_100",
                "particle": "alpha",
                "modalities": "ToT+ToA",
                "model": "late_prediction_fusion",
                "fusion_mode": "selector_or_residual",
                "gate_context": "posthoc prediction-level selector/residual diagnostics",
                "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "notes": "A4b prediction-level fusion / oracle / selector diagnostics",
            }
            for key, value in src.items():
                if key.startswith(("val_confusion_", "test_confusion_")):
                    continue
                if key.startswith(("val_", "test_")) or key in {"n_runs", "seeds", "selected_strategies"}:
                    row[key] = value
            if experiment_id == "A4b-candidate":
                stem = "a4b_4e_rule_selector"
            elif experiment_id == "A4b-5":
                stem = "a4b_5_gated_late_fusion"
            else:
                stem = "a4b_6_residual_gated_fusion"
            for key, value in a4b_modality_high_angle_stats(stem, src.get("method", "")).items():
                if value and not row.get(key):
                    row[key] = value
            rows.append(row)

    for path, experiment_id in [
        (OUTPUTS / "a4c_end_to_end_bimodal_fusion_mean_std.csv", "A4c-1-3"),
        (OUTPUTS / "a4c_warm_started_expert_gate_mean_std.csv", "A4c-4"),
    ]:
        if not path.exists():
            continue
        if experiment_id == "A4c-1-3":
            run_summaries = run_summaries_by_key(OUTPUTS / "a4c_end_to_end_bimodal_fusion_runs.csv", ["model"])
        else:
            run_summaries = run_summaries_by_key(OUTPUTS / "a4c_warm_started_expert_gate_runs.csv", ["model", "expert_gate_freeze_experts"])
        for src in read_csv(path):
            row = {
                "experiment_id": experiment_id,
                "method_id": src.get("model", ""),
                "method_label": src.get("model", ""),
                "dataset": src.get("dataset", "Alpha_100"),
                "particle": src.get("particle", "alpha"),
                "modalities": src.get("modalities", "ToT+ToA"),
                "gate_context": "end-to-end image-branch ToT/ToA gate or FiLM diagnostic",
                "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "notes": "A4c end-to-end fusion gate/FiLM diagnostics",
            }
            for key, value in src.items():
                if key.startswith(("val_confusion_", "test_confusion_")):
                    continue
                if key.startswith(("val_", "test_")) or key in {"n_runs", "seeds", "model", "fusion_mode", "expert_gate_freeze_experts"}:
                    row[key] = value
            if experiment_id == "A4c-1-3":
                summary_key = (src.get("model", ""),)
            else:
                summary_key = (src.get("model", ""), src.get("expert_gate_freeze_experts", ""))
            for key, value in run_summaries.get(summary_key, {}).items():
                if (key.endswith("_mean") or key.endswith("_std")) and not row.get(key):
                    row[key] = value
            rows.append(row)
    a7_path = OUTPUTS / "a7_final_gmu_main5feat_gated_3seed_mean_std.csv"
    if a7_path.exists():
        run_summaries = run_summaries_by_key(OUTPUTS / "a7_final_gmu_main5feat_gated_3seed_runs.csv", ["model", "handcrafted_features"])
        for src in read_csv(a7_path):
            row = {
                "experiment_id": "A7",
                "method_id": "gmu_aux_main_5feat_gated",
                "method_label": "GMU_aux + main_5feat gated",
                "dataset": src.get("dataset", "Alpha_100"),
                "particle": src.get("particle", "alpha"),
                "modalities": src.get("modalities", "ToT+ToA"),
                "model": src.get("model", "dual_stream_gmu_aux"),
                "fusion_mode": src.get("fusion_mode", "gated"),
                "gate_context": "GMU image-branch ToT/ToA gate; not the handcrafted scalar fusion gate",
                "handcrafted_features": src.get("handcrafted_features", ""),
                "source_file": str(a7_path.relative_to(ROOT)).replace("\\", "/"),
                "notes": "A7 final component check; gate_tot/gate_toa describe GMU image branches only",
            }
            for key, value in src.items():
                if key.startswith(("val_confusion_", "test_confusion_")):
                    continue
                if key.startswith(("val_", "test_")) or key in {"n_runs", "seeds", "model", "fusion_mode"}:
                    row[key] = value
            summary_key = (src.get("model", ""), src.get("handcrafted_features", ""))
            for key, value in run_summaries.get(summary_key, {}).items():
                if (key.endswith("_mean") or key.endswith("_std")) and not row.get(key):
                    row[key] = value
            rows.append(row)
    return rows


def write_dynamic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    write_csv(path, rows, fields)


def run_summaries_by_key(path: Path, key_fields: list[str]) -> dict[tuple[str, ...], dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    if not path.exists():
        return {}
    for src in read_csv(path):
        key = tuple(src.get(field, "") for field in key_fields)
        grouped[key].append(row_from_run_csv(src, "", "", "", path))
    return {key: summarize_runs(rows) for key, rows in grouped.items()}


def build_handcrafted_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    a5a_dir = OUTPUTS / "a5a_alpha_handcrafted_screening" / "a5a_alpha_handcrafted_screening"
    for src in read_csv(a5a_dir / "model_metrics.csv"):
        rows.append({
            "row_type": "classical_model_metrics",
            "experiment_id": "A5a",
            "method_id": src.get("model", ""),
            "method_label": src.get("model", ""),
            "dataset": "Alpha_100",
            "particle": "alpha",
            "modalities": "handcrafted_only",
            "fusion_mode": "classical_ml",
            "split": src.get("split", ""),
            "accuracy": src.get("accuracy", ""),
            "macro_f1": src.get("macro_f1", ""),
            "mae_argmax": src.get("mae_argmax", ""),
            "source_file": "outputs/a5a_alpha_handcrafted_screening/a5a_alpha_handcrafted_screening/model_metrics.csv",
            "notes": "handcrafted-only screening; diagnostic",
        })
    for src in read_csv(a5a_dir / "group_permutation_importance_val.csv"):
        rows.append({
            "row_type": "feature_group_importance",
            "experiment_id": "A5a",
            "method_id": f"{src.get('model', '')}:{src.get('group', '')}",
            "method_label": f"{src.get('model', '')} group {src.get('group', '')}",
            "dataset": "Alpha_100",
            "particle": "alpha",
            "modalities": "handcrafted_only",
            "fusion_mode": "permutation_importance",
            "handcrafted_features": src.get("features", ""),
            "val_baseline_accuracy": src.get("baseline_accuracy", ""),
            "importance_mean": src.get("importance_mean", ""),
            "importance_std": src.get("importance_std", ""),
            "source_file": "outputs/a5a_alpha_handcrafted_screening/a5a_alpha_handcrafted_screening/group_permutation_importance_val.csv",
            "notes": "feature group importance on validation split",
        })

    for path, experiment_id in [
        (OUTPUTS / "a5b_alpha_handcrafted_group_ablation_runs.csv", "A5b"),
        (OUTPUTS / "a5c_alpha_handcrafted_gated_seed42_runs.csv", "A5c"),
        (OUTPUTS / "a5d_alpha_handcrafted_gated_3seed_runs.csv", "A5d"),
        (OUTPUTS / "a7_final_gmu_main5feat_gated_3seed_runs.csv", "A7"),
        (OUTPUTS / "b2_proton_c7_handcrafted_lowcorr_seed42_runs.csv", "B2a"),
        (OUTPUTS / "b2_proton_c7_handcrafted_gated_seed42_runs.csv", "B2b"),
    ]:
        if not path.exists():
            continue
        for src in read_csv(path):
            json_metrics: dict[str, object] = {}
            metric_path = metrics_path_from_source(src)
            if metric_path:
                metric_data = read_json(metric_path)
                json_metrics.update(split_metrics(metric_data, "validation", src.get("dataset", "")))
                json_metrics.update(split_metrics(metric_data, "test", src.get("dataset", "")))
            row = {
                "row_type": "cnn_feature_fusion",
                "experiment_id": experiment_id,
                "method_id": method_id_from_features(src.get("handcrafted_features", "")),
                "method_label": method_label_from_features(src.get("handcrafted_features", "")),
                "dataset": src.get("dataset", ""),
                "particle": src.get("particle", ""),
                "modalities": src.get("modalities", ""),
                "model": src.get("model", ""),
                "fusion_mode": src.get("fusion_mode", ""),
                "loss": src.get("loss", ""),
                "label_encoding": src.get("label_encoding", ""),
                "seed": src.get("seed", ""),
                "handcrafted_features": src.get("handcrafted_features", ""),
                "handcrafted_source_modalities": src.get("handcrafted_source_modalities", ""),
                "comparison_role": "final_gmu_component_check" if experiment_id == "A7" else (
                    "three_seed_feature_verification" if experiment_id == "A5d" else "seed42_feature_screening_or_transfer"
                ),
                "val_accuracy": src.get("val_accuracy", ""),
                "val_mae_argmax": src.get("val_mae_argmax", ""),
                "val_macro_f1": src.get("val_macro_f1", "") or json_metrics.get("val_macro_f1", ""),
                "test_accuracy": src.get("test_accuracy", ""),
                "test_mae_argmax": src.get("test_mae_argmax", ""),
                "test_macro_f1": src.get("test_macro_f1", ""),
                "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "notes": "A7 is final GMU component check; test is report-only" if experiment_id == "A7" else "test is report-only; use validation for any selection",
            }
            rows.append(row)
    return rows


def build_loss_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add_row(
        src: dict[str, object],
        experiment_id: str,
        phase: str,
        comparison_family: str,
        selection_status: str,
        reference_baseline_id: str,
        source_file: str,
        notes: str,
    ) -> None:
        json_metrics: dict[str, object] = {}
        metric_path = metrics_path_from_source(src)
        if metric_path:
            try:
                metric_data = read_json(metric_path)
                dataset = str(src.get("dataset", ""))
                json_metrics.update(split_metrics(metric_data, "validation", dataset))
                json_metrics.update(split_metrics(metric_data, "test", dataset))
            except (OSError, json.JSONDecodeError):
                json_metrics = {}

        def metric_value(field: str) -> object:
            return src.get(field, "") or json_metrics.get(field, "")

        rows.append({
            "experiment_id": experiment_id,
            "phase": phase,
            "comparison_family": comparison_family,
            "method_id": loss_method_id(src),
            "method_label": loss_method_label(src),
            "dataset": src.get("dataset", ""),
            "particle": src.get("particle", ""),
            "modalities": src.get("modalities", ""),
            "seed": src.get("seed", ""),
            "loss": src.get("loss", ""),
            "label_encoding": src.get("label_encoding", ""),
            "gaussian_sigma": src.get("gaussian_sigma", ""),
            "expected_mae_weight": src.get("expected_mae_weight", ""),
            "emd_weight": src.get("emd_weight", ""),
            "emd_p": src.get("emd_p", ""),
            "emd_angle_weighted": src.get("emd_angle_weighted", ""),
            "selection_status": selection_status,
            "reference_baseline_id": reference_baseline_id,
            "val_accuracy": metric_value("val_accuracy"),
            "val_mae_argmax": metric_value("val_mae_argmax"),
            "val_macro_f1": metric_value("val_macro_f1"),
            "val_high_angle_macro_f1": metric_value("val_high_angle_macro_f1"),
            "val_far_error_rate_abs_ge_20": metric_value("val_far_error_rate_abs_ge_20"),
            "test_accuracy": metric_value("test_accuracy"),
            "test_mae_argmax": metric_value("test_mae_argmax"),
            "test_macro_f1": metric_value("test_macro_f1"),
            "test_high_angle_macro_f1": metric_value("test_high_angle_macro_f1"),
            "test_far_error_rate_abs_ge_20": metric_value("test_far_error_rate_abs_ge_20"),
            "source_file": source_file,
            "notes": notes,
        })

    for src in a2_runs():
        add_row(
            src,
            "A2-best",
            "alpha_reference_baseline_three_seed",
            "Alpha A6 ordered-loss comparison",
            "reference_baseline",
            "A2-best:cross_entropy_onehot",
            str(src.get("source_file", "")),
            "Alpha CE one-hot baseline reused for A6; model selection must use validation only",
        )

    for path, experiment_id, phase in [
        (OUTPUTS / "a6a_alpha_tot_ordinal_loss_seed42_runs.csv", "A6a", "seed42_screening"),
        (OUTPUTS / "a6b_alpha_tot_ce_emd_0p02_3seed_runs.csv", "A6b", "three_seed_verification"),
    ]:
        if not path.exists():
            continue
        for src in read_csv(path):
            method = loss_method_id(src)
            if experiment_id == "A6a":
                status = "validation_selected_seed42_tie_break" if method == "ce_emd_lambda_0.02" else "screening_candidate"
                note = "A6a seed42 screening; CE+EMD lambda=0.02 was only a tie-break-level validation candidate"
            else:
                status = "three_seed_rejected"
                note = "A6b shows CE+EMD lambda=0.02 is weaker than A2 CE baseline on validation"
            add_row(
                src,
                experiment_id,
                phase,
                "Alpha A6 ordered-loss comparison",
                status,
                "A2-best:cross_entropy_onehot",
                str(path.relative_to(ROOT)).replace("\\", "/"),
                note,
            )

    b1_path = OUTPUTS / "b1_proton_c7_resnet18_tot_best_patience8_3seed_runs.csv"
    if b1_path.exists():
        for src in read_csv(b1_path):
            add_row(
                src,
                "B1-best",
                "proton_reference_baseline_three_seed",
                "Proton B3 ordered-loss comparison",
                "reference_baseline",
                "B1-best:cross_entropy_onehot",
                str(b1_path.relative_to(ROOT)).replace("\\", "/"),
                "Proton CE one-hot baseline reused for B3 ordered-loss comparison",
            )

    for path, experiment_id, phase in [
        (OUTPUTS / "b3a_proton_c7_ordinal_loss_seed42_runs.csv", "B3a", "seed42_screening"),
        (OUTPUTS / "b3b_proton_c7_expected_mae_3seed_runs.csv", "B3b-main", "three_seed_verification"),
        (OUTPUTS / "b3b_proton_c7_ce_emd_optional_3seed_runs.csv", "B3b-optional", "three_seed_verification"),
    ]:
        if not path.exists():
            continue
        for src in read_csv(path):
            method = loss_method_id(src)
            if experiment_id == "B3a":
                if method == "ce_expected_mae_lambda_0.05":
                    status = "validation_selected_seed42"
                elif method == "ce_emd_lambda_0.05":
                    status = "optional_close_candidate"
                else:
                    status = "screening_candidate"
                note = "B3a selects by validation; test is report-only"
            elif experiment_id == "B3b-main":
                status = "three_seed_selected"
                note = "B3b ExpectedMAE is the current recommended Proton ordered loss"
            else:
                status = "three_seed_optional_control"
                note = "B3b CE+EMD is a strong optional ordered-loss control"
            add_row(
                src,
                experiment_id,
                phase,
                "Proton B3 ordered-loss comparison",
                status,
                "B1-best:cross_entropy_onehot",
                str(path.relative_to(ROOT)).replace("\\", "/"),
                note,
            )
    return rows


def build_decision_rows(main_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (str(row.get("experiment_id", "")), str(row.get("method_id", ""))): row
        for row in main_rows
    }

    def delta(candidate: dict[str, object], baseline: dict[str, object], field: str, scale: float = 1.0) -> str:
        cand = to_float(candidate.get(field))
        base = to_float(baseline.get(field))
        if cand is None or base is None:
            return ""
        return fmt((cand - base) * scale)

    def append_decision(
        out: list[dict[str, object]],
        decision_id: str,
        topic: str,
        baseline_key: tuple[str, str],
        candidate_key: tuple[str, str],
        validation_rule: str,
        decision: str,
        final_status: str,
        paper_interpretation: str,
        notes: str,
    ) -> None:
        baseline = by_key.get(baseline_key)
        candidate = by_key.get(candidate_key)
        if not baseline or not candidate:
            out.append({
                "decision_id": decision_id,
                "decision_topic": topic,
                "baseline_experiment_id": baseline_key[0],
                "baseline_method_id": baseline_key[1],
                "candidate_experiment_id": candidate_key[0],
                "candidate_method_id": candidate_key[1],
                "decision": "source_missing",
                "notes": f"Could not find baseline or candidate in 01_main_results_summary. {notes}",
            })
            return
        out.append({
            "decision_id": decision_id,
            "dataset": candidate.get("dataset", ""),
            "particle": candidate.get("particle", ""),
            "decision_topic": topic,
            "baseline_experiment_id": baseline_key[0],
            "baseline_method_id": baseline_key[1],
            "baseline_label": baseline.get("method_label", ""),
            "candidate_experiment_id": candidate_key[0],
            "candidate_method_id": candidate_key[1],
            "candidate_label": candidate.get("method_label", ""),
            "validation_rule": validation_rule,
            "val_accuracy_delta_pp": delta(candidate, baseline, "val_accuracy_mean", 100.0),
            "val_mae_delta_deg": delta(candidate, baseline, "val_mae_argmax_mean"),
            "val_macro_f1_delta": delta(candidate, baseline, "val_macro_f1_mean"),
            "val_high_angle_macro_f1_delta": delta(candidate, baseline, "val_high_angle_macro_f1_mean"),
            "test_accuracy_delta_pp": delta(candidate, baseline, "test_accuracy_mean", 100.0),
            "test_mae_delta_deg": delta(candidate, baseline, "test_mae_argmax_mean"),
            "test_macro_f1_delta": delta(candidate, baseline, "test_macro_f1_mean"),
            "test_high_angle_macro_f1_delta": delta(candidate, baseline, "test_high_angle_macro_f1_mean"),
            "decision": decision,
            "final_status": final_status,
            "paper_interpretation": paper_interpretation,
            "source_files": f"{baseline.get('source_file', '')};{candidate.get('source_file', '')}",
            "notes": notes,
        })

    rows: list[dict[str, object]] = []
    append_decision(
        rows,
        "alpha_loss_a6b",
        "Alpha ToT ordered-loss strategy",
        ("A2-best", "alpha_tot_resnet18_no_maxpool_baseline"),
        ("A6b", "ce_emd_lambda_0.02"),
        "A6a candidate must survive three-seed validation comparison; test is report-only",
        "reject_candidate_keep_baseline",
        "Alpha final loss remains CE one-hot",
        "CE+EMD lambda=0.02 had only a seed42 tie-break signal and became weaker than A2 CE after three seeds.",
        "Do not migrate A6 loss to GMU or final Alpha model.",
    )
    append_decision(
        rows,
        "alpha_gmu_handcrafted_a7",
        "Final Alpha GMU handcrafted-feature component",
        ("A4c-1-3", "dual_stream_gmu_aux"),
        ("A7", "gmu_aux_main_5feat_gated"),
        "A7 must improve validation metrics over A4c GMU; test is report-only",
        "reject_candidate_keep_gmu_no_handcrafted",
        "Alpha final end-to-end model uses GMU without handcrafted scalars",
        "main_5feat does not provide stable validation gain on top of GMU; Val Acc ties but Val MAE worsens.",
        "A7 gate_tot/gate_toa are GMU image-branch gates, not handcrafted fusion weights.",
    )
    append_decision(
        rows,
        "alpha_final_end_to_end_gmu",
        "Alpha final end-to-end multimodal architecture",
        ("A2-best", "alpha_tot_resnet18_no_maxpool_baseline"),
        ("A4c-1-3", "dual_stream_gmu_aux"),
        "Select end-to-end multimodal architecture by validation balance and physical interpretability; test reports generalization only",
        "adopt_candidate",
        "Alpha final end-to-end multimodal model is dual_stream_gmu_aux + ToT/relative_minmax ToA + CE + no handcrafted",
        "GMU is not chosen because of test back-selection; it has competitive validation accuracy, better A4c Val MAE, and gate behavior consistent with ToT-main/ToA-auxiliary physics.",
        "A4b post-hoc fusion remains expert-level reference, not the final deployable end-to-end architecture.",
    )
    append_decision(
        rows,
        "proton_loss_b3b_expected_mae",
        "Proton_C_7 ordered-loss strategy",
        ("B1-best", "proton_tot_resnet18_no_maxpool_baseline"),
        ("B3b-main", "ce_expected_mae_lambda_0.05"),
        "B3a selects by validation accuracy; B3b verifies by three seeds",
        "adopt_candidate",
        "Proton recommended loss is CE + ExpectedMAE lambda=0.05",
        "ExpectedMAE improves validation accuracy, MAE, macro-F1, and high-angle behavior relative to the CE baseline.",
        "CE+EMD lambda=0.05 remains a strong optional ordered-loss control.",
    )
    return rows


def build_excluded_rows(index: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for exp_id, src in index.items():
        if src.get("role") in {"excluded", "diagnostic"} or src.get("include_level") in {"exclude", "discussion", "discussion_or_appendix"}:
            rows.append(src)
    return rows


def split_handcrafted_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    classical = [row for row in rows if row.get("row_type") == "classical_model_metrics"]
    importance = [row for row in rows if row.get("row_type") == "feature_group_importance"]
    cnn = [row for row in rows if row.get("row_type") == "cnn_feature_fusion"]
    return classical, importance, cnn


def categorize_missing(table_name: str, field: str, missing_rows: list[dict[str, object]]) -> tuple[str, str, str]:
    experiments = {str(row.get("experiment_id", "")) for row in missing_rows}
    if experiments == {"A6"}:
        return "pending", "A6 尚未完成或尚未拉取结果。", "等待 A6 完成后重新生成。"
    if table_name == "06_handcrafted_feature_results.csv":
        return "heterogeneous_combined_table", "该表合并了经典模型指标、特征重要性和 CNN 融合三种行类型。", "论文分析优先使用 06a/06b/06c 拆分表。"
    if field == "handcrafted_features":
        return "not_applicable", "该实验没有启用手工特征。", "保持空值或在展示时标为 NA。"
    if field in {"best_epoch", "stopped_epoch", "max_epochs", "early_stopped"} and any(exp.startswith("A4b") for exp in experiments):
        return "not_applicable", "A4b 是基于已有预测的 post-hoc fusion，不重新训练。", "保持空值或在展示时标为 NA。"
    if any(token in field for token in ["gate_", "film_", "selection_rate", "selected_strategies", "expert_gate"]):
        return "not_applicable", "该字段只对特定 gate/FiLM/selector 方法存在。", "保持空值或在展示时标为 NA。"
    if field in {"gaussian_sigma", "expected_mae_weight", "emd_weight", "emd_p", "emd_angle_weighted"}:
        return "not_applicable", "该字段只对对应损失函数存在。", "保持空值或在展示时标为 NA。"
    if field in {"test_seconds_mean", "test_seconds_std"} and any(exp.startswith("A4b") for exp in experiments):
        return "not_applicable", "A4b 是后处理融合诊断，不是标准训练/测试计时实验。", "保持空值或在展示时标为 NA。"
    if "far_error_rate" in field and any(exp.startswith("A4b") for exp in experiments):
        return "source_missing", "A4b 后处理汇总未保存混淆矩阵或逐样本预测，不能稳定派生 far-error。", "如论文需要该指标，应扩展 A4b 脚本保存预测或混淆矩阵后重算。"
    if "high_angle_macro_f1" in field and any(exp.startswith("A4b") for exp in experiments):
        return "partially_recoverable", "A4b 可从 by_class 派生 high-angle F1；若仍为空通常说明该行没有 by_class 对应项。", "检查具体 strategy 是否在 by_class 中。"
    if field in {"model", "fusion_mode", "loss", "label_encoding"}:
        return "source_missing", "源汇总表未携带该配置字段，或该行不是训练模型。", "必要时从实验索引或 metadata 回填。"
    return "needs_review", "未被规则自动归类。", "人工抽查源文件。"


def build_missing_audit(tables: dict[str, tuple[list[dict[str, object]], list[str]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for table_name, (table_rows, fields) in tables.items():
        total = len(table_rows)
        for field in fields:
            missing = [row for row in table_rows if str(row.get(field, "")).strip() == ""]
            if not missing:
                continue
            category, reason, action = categorize_missing(table_name, field, missing)
            experiments = sorted({str(row.get("experiment_id", "")) for row in missing if str(row.get("experiment_id", ""))})
            rows.append({
                "table_name": table_name,
                "field": field,
                "empty_cells": len(missing),
                "total_rows": total,
                "empty_rate": len(missing) / total if total else "",
                "category": category,
                "affected_experiments": ";".join(experiments),
                "reason": reason,
                "recommended_action": action,
            })
    return rows


def main() -> int:
    index = index_rows()
    run_rows = build_run_rows()
    main_rows = build_main_summary(index, run_rows)
    per_class_rows = build_per_class_rows()
    error_rows = build_error_rows(main_rows)
    modality_rows = build_modality_gate_rows()
    handcrafted_rows = build_handcrafted_rows()
    handcrafted_classical_rows, handcrafted_importance_rows, handcrafted_cnn_rows = split_handcrafted_rows(handcrafted_rows)
    loss_rows = build_loss_rows()
    decision_rows = build_decision_rows(main_rows)
    excluded_rows = build_excluded_rows(index)
    error_fields = list(error_rows[0].keys()) if error_rows else []
    modality_fields = list({key: None for row in modality_rows for key in row}.keys())
    handcrafted_fields = list({key: None for row in handcrafted_rows for key in row}.keys())
    handcrafted_classical_fields = list({key: None for row in handcrafted_classical_rows for key in row}.keys())
    handcrafted_importance_fields = list({key: None for row in handcrafted_importance_rows for key in row}.keys())
    handcrafted_cnn_fields = list({key: None for row in handcrafted_cnn_rows for key in row}.keys())
    loss_fields = list({key: None for row in loss_rows for key in row}.keys())
    decision_fields = list({key: None for row in decision_rows for key in row}.keys())
    excluded_fields = list(excluded_rows[0].keys()) if excluded_rows else []
    audit_rows = build_missing_audit({
        "01_main_results_summary.csv": (main_rows, MAIN_FIELDS),
        "02_run_level_results.csv": (run_rows, RUN_FIELDS),
        "03_per_class_results.csv": (per_class_rows, PER_CLASS_FIELDS),
        "04_error_structure.csv": (error_rows, error_fields),
        "05_modality_and_gate_diagnostics.csv": (modality_rows, modality_fields),
        "06_handcrafted_feature_results.csv": (handcrafted_rows, handcrafted_fields),
        "06a_handcrafted_classical_metrics.csv": (handcrafted_classical_rows, handcrafted_classical_fields),
        "06b_handcrafted_feature_importance.csv": (handcrafted_importance_rows, handcrafted_importance_fields),
        "06c_handcrafted_cnn_fusion.csv": (handcrafted_cnn_rows, handcrafted_cnn_fields),
        "07_loss_strategy_results.csv": (loss_rows, loss_fields),
        "08_excluded_or_diagnostic_runs.csv": (excluded_rows, excluded_fields),
        "10_final_decision_summary.csv": (decision_rows, decision_fields),
    })

    write_csv(PKG / "01_main_results_summary.csv", main_rows, MAIN_FIELDS)
    write_csv(PKG / "02_run_level_results.csv", run_rows, RUN_FIELDS)
    write_csv(PKG / "03_per_class_results.csv", per_class_rows, PER_CLASS_FIELDS)
    write_dynamic_csv(PKG / "04_error_structure.csv", error_rows)
    write_dynamic_csv(PKG / "05_modality_and_gate_diagnostics.csv", modality_rows)
    write_dynamic_csv(PKG / "06_handcrafted_feature_results.csv", handcrafted_rows)
    write_dynamic_csv(PKG / "06a_handcrafted_classical_metrics.csv", handcrafted_classical_rows)
    write_dynamic_csv(PKG / "06b_handcrafted_feature_importance.csv", handcrafted_importance_rows)
    write_dynamic_csv(PKG / "06c_handcrafted_cnn_fusion.csv", handcrafted_cnn_rows)
    write_dynamic_csv(PKG / "07_loss_strategy_results.csv", loss_rows)
    write_dynamic_csv(PKG / "08_excluded_or_diagnostic_runs.csv", excluded_rows)
    write_dynamic_csv(PKG / "10_final_decision_summary.csv", decision_rows)
    write_dynamic_csv(PKG / "09_missing_value_audit.csv", audit_rows)

    print("paper_data_package tables generated")
    for path in sorted(PKG.glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            count = max(sum(1 for _ in f) - 1, 0)
        print(f"{path.name}: {count} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
