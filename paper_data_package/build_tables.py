from __future__ import annotations

import csv
import json
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


def split_metrics(metrics: dict, split: str) -> dict[str, object]:
    data = metrics.get(split, {})
    prefix = "val" if split == "validation" else split
    return {
        f"{prefix}_accuracy": data.get("accuracy", ""),
        f"{prefix}_mae_argmax": data.get("mae_argmax", ""),
        f"{prefix}_p90_error": data.get("p90_error", ""),
        f"{prefix}_macro_f1": data.get("macro_f1", ""),
    }


def a2_runs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((EXPERIMENTS / "a2_best_3seed").glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
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
        row.update(split_metrics(metrics, "validation"))
        row.update(split_metrics(metrics, "test"))
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
            metric_data = json.loads(metric_path.read_text(encoding="utf-8"))
            json_metrics.update(split_metrics(metric_data, "validation"))
            json_metrics.update(split_metrics(metric_data, "test"))
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
    return path if path.exists() else None


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


def a4b_run_rows(path: Path, experiment_id: str, keep: set[str]) -> list[dict[str, object]]:
    rows = []
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
    rows.extend(runs_from_csv_group(
        OUTPUTS / "a5d_alpha_handcrafted_gated_3seed_runs.csv",
        "A5d",
        "handcrafted_features",
        notes="handcrafted feature three-seed verification",
    ))
    rows.extend(runs_from_csv_group(
        OUTPUTS / "b1_proton_c7_resnet18_tot_best_patience8_3seed_runs.csv",
        "B1-best",
        "model",
        label_prefix="Proton baseline ",
        notes="formal Proton_C_7 baseline",
    ))
    rows.extend(runs_from_csv_group(
        OUTPUTS / "b3b_proton_c7_expected_mae_3seed_runs.csv",
        "B3b-main",
        "loss",
        label_prefix="",
        notes="validation-selected ordered loss",
    ))
    rows.extend(runs_from_csv_group(
        OUTPUTS / "b3b_proton_c7_ce_emd_optional_3seed_runs.csv",
        "B3b-optional",
        "loss",
        label_prefix="",
        notes="optional ordered-loss control",
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

    for path, experiment_id, method_id, label, basis, note in [
        (
            OUTPUTS / "a4b_5_gated_late_fusion_mean_std.csv",
            "A4b-5",
            "validation_selected_selector",
            "A4b-5 validation-selected late fusion selector",
            "validation-selected selector within A4b-5",
            "best selector chosen by validation, not by test",
        ),
        (
            OUTPUTS / "a4b_6_residual_gated_fusion_mean_std.csv",
            "A4b-6",
            "validation_selected_residual_validation_grid",
            "A4b-6 validation-selected residual fusion",
            "validation-selected residual rule within A4b-6",
            "best residual selector chosen by validation, not by test",
        ),
    ]:
        for src in read_csv(path):
            if src.get("method") == method_id:
                rows.append(summary_row_from_mean_std(index, path, src, experiment_id, method_id, label, basis, note))

    a4c_path = OUTPUTS / "a4c_end_to_end_bimodal_fusion_mean_std.csv"
    for src in read_csv(a4c_path):
        model = src.get("model", "")
        rows.append(summary_row_from_mean_std(
            index,
            a4c_path,
            src,
            "A4c-1-3",
            model,
            f"A4c {model}",
            "three-seed end-to-end bimodal comparison; selection by validation",
            "A4c formal end-to-end bimodal comparison; do not choose by test",
            extra={"val_macro_f1_mean": summarize_runs(by_exp_method[("A4c-1-3", model)]).get("val_macro_f1_mean", ""),
                   "val_macro_f1_std": summarize_runs(by_exp_method[("A4c-1-3", model)]).get("val_macro_f1_std", "")},
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
            extra={"val_macro_f1_mean": extra.get("val_macro_f1_mean", ""), "val_macro_f1_std": extra.get("val_macro_f1_std", "")},
        ))

    b1_path = OUTPUTS / "b1_proton_c7_resnet18_tot_best_patience8_3seed_mean_std.csv"
    b1_src = read_csv(b1_path)[0]
    b1_runs = by_exp_method[("B1-best", "resnet18_no_maxpool")]
    rows.append(summary_row_from_mean_std(
        index,
        b1_path,
        b1_src,
        "B1-best",
        "proton_tot_resnet18_no_maxpool_baseline",
        "Proton_C_7 ToT baseline: ResNet18 no maxpool",
        "formal three-seed baseline",
        "baseline reference for Proton_C_7 experiments",
        extra={"val_macro_f1_mean": summarize_runs(b1_runs).get("val_macro_f1_mean", ""),
               "val_macro_f1_std": summarize_runs(b1_runs).get("val_macro_f1_std", "")},
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
    metrics = json.loads(path.read_text(encoding="utf-8"))
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
    return path if path.exists() else None


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
                "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "notes": "A4b prediction-level fusion / oracle / selector diagnostics",
            }
            for key, value in src.items():
                if key.startswith(("val_", "test_")) or key in {"n_runs", "seeds", "selected_strategies"}:
                    row[key] = value
            rows.append(row)

    for path, experiment_id in [
        (OUTPUTS / "a4c_end_to_end_bimodal_fusion_mean_std.csv", "A4c-1-3"),
        (OUTPUTS / "a4c_warm_started_expert_gate_mean_std.csv", "A4c-4"),
    ]:
        if not path.exists():
            continue
        for src in read_csv(path):
            row = {
                "experiment_id": experiment_id,
                "method_id": src.get("model", ""),
                "method_label": src.get("model", ""),
                "dataset": src.get("dataset", "Alpha_100"),
                "particle": src.get("particle", "alpha"),
                "modalities": src.get("modalities", "ToT+ToA"),
                "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "notes": "A4c end-to-end fusion gate/FiLM diagnostics",
            }
            for key, value in src.items():
                if key.startswith(("val_", "test_")) or key in {"n_runs", "seeds", "model", "fusion_mode", "expert_gate_freeze_experts"}:
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


def build_handcrafted_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    a5a_dir = OUTPUTS / "a5a_alpha_handcrafted_screening" / "a5a_alpha_handcrafted_screening"
    for src in read_csv(a5a_dir / "model_metrics.csv"):
        rows.append({
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
        (OUTPUTS / "b2_proton_c7_handcrafted_lowcorr_seed42_runs.csv", "B2a"),
        (OUTPUTS / "b2_proton_c7_handcrafted_gated_seed42_runs.csv", "B2b"),
    ]:
        if not path.exists():
            continue
        for src in read_csv(path):
            row = {
                "experiment_id": experiment_id,
                "method_id": method_id_from_features(src.get("handcrafted_features", "")),
                "method_label": method_label_from_features(src.get("handcrafted_features", "")),
                "dataset": src.get("dataset", ""),
                "particle": src.get("particle", ""),
                "modalities": src.get("modalities", ""),
                "fusion_mode": src.get("fusion_mode", ""),
                "seed": src.get("seed", ""),
                "handcrafted_features": src.get("handcrafted_features", ""),
                "handcrafted_source_modalities": src.get("handcrafted_source_modalities", ""),
                "val_accuracy": src.get("val_accuracy", ""),
                "val_mae_argmax": src.get("val_mae_argmax", ""),
                "val_macro_f1": src.get("val_macro_f1", ""),
                "test_accuracy": src.get("test_accuracy", ""),
                "test_mae_argmax": src.get("test_mae_argmax", ""),
                "test_macro_f1": src.get("test_macro_f1", ""),
                "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "notes": "test is report-only; use validation for any selection",
            }
            rows.append(row)
    return rows


def build_loss_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path, experiment_id, phase in [
        (OUTPUTS / "b3a_proton_c7_ordinal_loss_seed42_runs.csv", "B3a", "seed42_screening"),
        (OUTPUTS / "b3b_proton_c7_expected_mae_3seed_runs.csv", "B3b-main", "three_seed_verification"),
        (OUTPUTS / "b3b_proton_c7_ce_emd_optional_3seed_runs.csv", "B3b-optional", "three_seed_verification"),
    ]:
        if not path.exists():
            continue
        for src in read_csv(path):
            rows.append({
                "experiment_id": experiment_id,
                "phase": phase,
                "method_id": src.get("loss", ""),
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
                "val_accuracy": src.get("val_accuracy", ""),
                "val_mae_argmax": src.get("val_mae_argmax", ""),
                "val_macro_f1": src.get("val_macro_f1", ""),
                "val_high_angle_macro_f1": src.get("val_high_angle_macro_f1", ""),
                "val_far_error_rate_abs_ge_20": src.get("val_far_error_rate_abs_ge_20", ""),
                "test_accuracy": src.get("test_accuracy", ""),
                "test_mae_argmax": src.get("test_mae_argmax", ""),
                "test_macro_f1": src.get("test_macro_f1", ""),
                "test_high_angle_macro_f1": src.get("test_high_angle_macro_f1", ""),
                "test_far_error_rate_abs_ge_20": src.get("test_far_error_rate_abs_ge_20", ""),
                "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "notes": "B3a selects by validation; B3b reports final test only after selection",
            })
    rows.append({
        "experiment_id": "A6",
        "phase": "planned_or_running",
        "dataset": "Alpha_100",
        "particle": "alpha",
        "modalities": "ToT",
        "notes": "A6 Alpha ordered-loss results were pending in the latest experiment log",
    })
    return rows


def build_excluded_rows(index: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for exp_id, src in index.items():
        if src.get("role") in {"excluded", "diagnostic"} or src.get("include_level") in {"exclude", "discussion", "discussion_or_appendix"}:
            rows.append(src)
    return rows


def main() -> int:
    index = index_rows()
    run_rows = build_run_rows()
    main_rows = build_main_summary(index, run_rows)
    per_class_rows = build_per_class_rows()
    error_rows = build_error_rows(main_rows)
    modality_rows = build_modality_gate_rows()
    handcrafted_rows = build_handcrafted_rows()
    loss_rows = build_loss_rows()
    excluded_rows = build_excluded_rows(index)

    write_csv(PKG / "01_main_results_summary.csv", main_rows, MAIN_FIELDS)
    write_csv(PKG / "02_run_level_results.csv", run_rows, RUN_FIELDS)
    write_csv(PKG / "03_per_class_results.csv", per_class_rows, PER_CLASS_FIELDS)
    write_dynamic_csv(PKG / "04_error_structure.csv", error_rows)
    write_dynamic_csv(PKG / "05_modality_and_gate_diagnostics.csv", modality_rows)
    write_dynamic_csv(PKG / "06_handcrafted_feature_results.csv", handcrafted_rows)
    write_dynamic_csv(PKG / "07_loss_strategy_results.csv", loss_rows)
    write_dynamic_csv(PKG / "08_excluded_or_diagnostic_runs.csv", excluded_rows)

    print("paper_data_package tables generated")
    for path in sorted(PKG.glob("0*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            count = max(sum(1 for _ in f) - 1, 0)
        print(f"{path.name}: {count} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
