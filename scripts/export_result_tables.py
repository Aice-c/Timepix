"""Export normalized experiment result tables for paper analysis.

This script is intentionally read-only with respect to project outputs. It scans
standard training runs under outputs/experiments and writes normalized CSV
snapshots to an analysis directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any


SCALAR_METRIC_KEYS = {
    "loss",
    "accuracy",
    "mae",
    "mae_argmax",
    "mae_weighted",
    "p90_error",
    "p90_error_weighted",
    "macro_f1",
    "high_angle_macro_f1",
    "confusion_45_50",
    "confusion_60_70",
    "far_error_rate_abs_ge_20",
}


def _long_path(path: str | Path) -> str:
    resolved = str(Path(path).resolve())
    if os.name != "nt":
        return resolved
    if resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


def _display_path(path: str | Path) -> str:
    text = str(path)
    if text.startswith("\\\\?\\UNC\\"):
        return "\\\\" + text[8:]
    if text.startswith("\\\\?\\"):
        return text[4:]
    return text


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(_long_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(_long_path(path), "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _exists(path: str | Path) -> bool:
    return os.path.exists(_long_path(path))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with open(_long_path(path), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_joined(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    if isinstance(value, dict):
        return ";".join(str(k) for k in value.keys())
    return str(value)


def _angle_values(metadata: dict[str, Any]) -> list[float]:
    data_info = metadata.get("data_info", {}) or {}
    label_map = data_info.get("label_map", {}) or {}
    if label_map:
        items = sorted(((int(k), v) for k, v in label_map.items()), key=lambda item: item[0])
        values: list[float] = []
        for _, raw in items:
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(float("nan"))
        return values
    num_classes = int(data_info.get("num_classes") or 0)
    return [float(i) for i in range(num_classes)]


def _stage_from_group(group: str) -> str:
    lowered = group.lower()
    special_map = {
        "a4b": "A4b",
        "a4c": "A4c",
        "a5a": "A5",
        "a5b": "A5",
        "a5c": "A5",
        "a5d": "A5",
        "b3a": "B3",
        "b3b": "B3",
    }
    for special, stage in special_map.items():
        if lowered.startswith(special):
            return stage
    match = re.match(r"([ab]\d+)", lowered)
    if not match:
        return "other"
    return match.group(1).upper()


def _formal_dataset(metadata: dict[str, Any], group: str) -> str:
    dataset = metadata.get("dataset", {}) or {}
    data_info = metadata.get("data_info", {}) or {}
    raw = str(dataset.get("name", ""))
    particle = str(dataset.get("particle", ""))
    data_root = str(data_info.get("data_root", ""))
    text = " ".join([raw, particle, data_root, group]).lower()
    if "proton_c_7" in text:
        return "Proton_C_7"
    if "proton_c" in text:
        return "Proton_C"
    if "alpha" in text:
        if "_50" in group.lower() or "50x50" in text or "alpha_50" in text:
            return "Alpha_50"
        return "Alpha_100"
    return raw or "unknown"


def _result_role(group: str, experiment_name: str, formal_dataset: str) -> str:
    text = f"{group} {experiment_name}".lower()
    if "template" in text:
        return "template_only"
    if "oracle" in text or "complementarity" in text:
        return "oracle_upper_bound"
    if formal_dataset == "Alpha_50" or "_50" in text:
        return "deprecated"
    if group == "b1_proton_c7_resnet18_tot_best_3seed":
        return "deprecated"
    if "seed42" in text or text.endswith("seed42"):
        return "pilot_seed42"
    if "screen" in text or "a5a" in text:
        return "screening"
    if "diagnostic" in text:
        return "diagnostic"
    if any(token in text for token in ["3seed", "best", "b3b", "a2_best"]):
        return "official_or_candidate"
    return "grid_or_single_run"


def _deprecated_reason(group: str, formal_dataset: str) -> str:
    if formal_dataset == "Alpha_50":
        return "Alpha_50 historical run; formal line returned to Alpha_100"
    if group == "b1_proton_c7_resnet18_tot_best_3seed":
        return "Older B1-best with patience=5; patience=8 is formal"
    return ""


def _find_best_log_row(rows: list[dict[str, str]], best_epoch: int | None) -> dict[str, str] | None:
    if not rows:
        return None
    if best_epoch is not None:
        for row in rows:
            if _safe_int(row.get("epoch")) == best_epoch:
                return row
    return rows[-1]


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key in SCALAR_METRIC_KEYS and not isinstance(value, (list, dict))}


def _setting_key(row: dict[str, Any]) -> str:
    parts = [
        row.get("experiment_group", ""),
        row.get("dataset", ""),
        row.get("modalities", ""),
        row.get("model", ""),
        row.get("fusion_mode", ""),
        row.get("loss", ""),
        row.get("label_encoding", ""),
        f"conv{row.get('conv1_kernel_size','')}-{row.get('conv1_stride','')}-{row.get('conv1_padding','')}",
        f"dropout{row.get('dropout','')}",
        f"lr{row.get('learning_rate','')}",
        f"batch{row.get('batch_size','')}",
        f"wd{row.get('weight_decay','')}",
        f"features={row.get('handcrafted_features','')}",
        f"toa={row.get('toa_transform','')}",
        f"mask={row.get('add_hit_mask','')}",
        f"loss_extra={row.get('loss_extra','')}",
    ]
    return "|".join(str(part) for part in parts)


def _scan_standard_runs(outputs_root: Path) -> list[Path]:
    experiments_root = outputs_root / "experiments"
    if not _exists(experiments_root):
        return []
    metrics_paths: list[Path] = []
    for dirpath, _, filenames in os.walk(_long_path(experiments_root)):
        if "metrics.json" in filenames:
            metrics_paths.append(Path(_display_path(os.path.join(dirpath, "metrics.json"))))
    return sorted(metrics_paths, key=lambda p: str(p).lower())


def _extract_run(
    metrics_path: Path,
    project_root: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    exp_dir = metrics_path.parent
    group = exp_dir.parent.name
    metrics = _read_json(metrics_path)
    metadata_path = exp_dir / "metadata.json"
    metadata = _read_json(metadata_path) if _exists(metadata_path) else {}
    experiment_name = metadata.get("experiment_name") or exp_dir.name
    stage = _stage_from_group(group)
    formal_dataset = _formal_dataset(metadata, group)
    role = _result_role(group, str(experiment_name), formal_dataset)
    angle_values = _angle_values(metadata)

    dataset_cfg = metadata.get("dataset", {}) or {}
    data_info = metadata.get("data_info", {}) or {}
    model_cfg = metadata.get("model", {}) or {}
    loss_cfg = metadata.get("loss", {}) or {}
    training_cfg = metadata.get("training", {}) or {}
    data_cfg = metadata.get("data", {}) or {}

    training_log_path = exp_dir / "training_log.csv"
    training_rows = _read_csv(training_log_path) if _exists(training_log_path) else []
    best_epoch = _safe_int(metrics.get("best_epoch") or metadata.get("best_epoch"))
    best_log_row = _find_best_log_row(training_rows, best_epoch)

    handcrafted_features = data_info.get("handcrafted_features")
    if not handcrafted_features:
        handcrafted_features = (metadata.get("handcrafted_features", {}) or {}).get("features")
    loss_extra = {
        key: value
        for key, value in loss_cfg.items()
        if key not in {"name", "label_encoding"}
    }

    row: dict[str, Any] = {
        "stage": stage,
        "experiment_group": group,
        "experiment_name": experiment_name,
        "experiment_dir": str(exp_dir),
        "formal_dataset": formal_dataset,
        "dataset": dataset_cfg.get("name", ""),
        "particle": dataset_cfg.get("particle", ""),
        "modalities": _as_joined(data_info.get("modalities") or dataset_cfg.get("modalities")),
        "input_channels": len(data_info.get("modalities", []) or dataset_cfg.get("modalities", []) or []),
        "toa_transform": data_cfg.get("toa_transform", "none"),
        "add_hit_mask": data_cfg.get("add_hit_mask", False),
        "handcrafted_enabled": bool(data_info.get("handcrafted_dim", 0)),
        "handcrafted_dim": data_info.get("handcrafted_dim", 0),
        "handcrafted_features": _as_joined(handcrafted_features),
        "task": metadata.get("task", ""),
        "model": model_cfg.get("name", ""),
        "fusion_mode": model_cfg.get("fusion_mode", ""),
        "conv1_kernel_size": model_cfg.get("conv1_kernel_size", ""),
        "conv1_stride": model_cfg.get("conv1_stride", ""),
        "conv1_padding": model_cfg.get("conv1_padding", ""),
        "dropout": model_cfg.get("dropout", ""),
        "feature_dim": model_cfg.get("feature_dim", ""),
        "hidden_dim": model_cfg.get("hidden_dim", ""),
        "loss": loss_cfg.get("name", ""),
        "label_encoding": loss_cfg.get("label_encoding", ""),
        "loss_extra": json.dumps(loss_extra, sort_keys=True, ensure_ascii=False),
        "seed": training_cfg.get("seed", ""),
        "split_seed": "",
        "split_path": data_info.get("split_path", ""),
        "best_epoch": best_epoch if best_epoch is not None else "",
        "stopped_epoch": metrics.get("stopped_epoch", ""),
        "max_epochs": metrics.get("max_epochs") or training_cfg.get("epochs", ""),
        "early_stopped": metrics.get("early_stopped", ""),
        "learning_rate": training_cfg.get("learning_rate", ""),
        "batch_size": training_cfg.get("batch_size", ""),
        "weight_decay": training_cfg.get("weight_decay", ""),
        "scheduler": training_cfg.get("scheduler", ""),
        "early_stopping_patience": training_cfg.get("early_stopping_patience", ""),
        "mixed_precision_enabled": (metadata.get("mixed_precision", {}) or {}).get("enabled", ""),
        "params_total": (metadata.get("param_count", {}) or {}).get("total", ""),
        "fit_seconds": metrics.get("fit_seconds", ""),
        "test_seconds": metrics.get("test_seconds", ""),
        "total_seconds": metrics.get("total_seconds", ""),
        "result_role": role,
        "deprecated_reason": _deprecated_reason(group, formal_dataset),
        "source_quality": "metrics_json+training_log" if best_log_row else "metrics_json",
        "config_path": metadata.get("config_path", ""),
    }

    for split_name in ("validation", "test"):
        split_metrics = metrics.get(split_name, {}) or {}
        for metric_name, value in _scalar_metrics(split_metrics).items():
            row[f"{split_name}_{metric_name}"] = value

    if best_log_row:
        row.update(
            {
                "train_loss_at_best_epoch": best_log_row.get("train_loss", ""),
                "train_accuracy_at_best_epoch": best_log_row.get("train_accuracy", ""),
                "train_mae_argmax_at_best_epoch": best_log_row.get("train_mae_argmax", ""),
                "train_p90_error_at_best_epoch": best_log_row.get("train_p90_error", ""),
                "train_macro_f1_at_best_epoch": best_log_row.get("train_macro_f1", ""),
                "val_loss_at_best_epoch": best_log_row.get("val_loss", ""),
            }
        )

    row["setting_key"] = _setting_key(row)

    epoch_rows: list[dict[str, Any]] = []
    for log_row in training_rows:
        epoch = _safe_int(log_row.get("epoch"))
        item = {
            "stage": stage,
            "experiment_group": group,
            "experiment_name": experiment_name,
            "experiment_dir": str(exp_dir),
            "formal_dataset": formal_dataset,
            "seed": row["seed"],
            "epoch": epoch if epoch is not None else log_row.get("epoch", ""),
            "is_best_epoch": bool(epoch is not None and best_epoch is not None and epoch == best_epoch),
        }
        item.update(log_row)
        epoch_rows.append(item)

    split_rows: list[dict[str, Any]] = []
    if best_log_row:
        train_map = {
            "loss": best_log_row.get("train_loss"),
            "accuracy": best_log_row.get("train_accuracy"),
            "mae_argmax": best_log_row.get("train_mae_argmax"),
            "p90_error": best_log_row.get("train_p90_error"),
            "macro_f1": best_log_row.get("train_macro_f1"),
        }
        for metric_name, value in train_map.items():
            split_rows.append(
                {
                    "stage": stage,
                    "experiment_group": group,
                    "experiment_name": experiment_name,
                    "experiment_dir": str(exp_dir),
                    "formal_dataset": formal_dataset,
                    "seed": row["seed"],
                    "split": "train",
                    "metric": metric_name,
                    "value": value,
                    "source": "training_log_best_epoch",
                    "availability": "derived_from_training_log",
                }
            )
    for split_name in ("validation", "test"):
        split_metrics = metrics.get(split_name, {}) or {}
        for metric_name, value in _scalar_metrics(split_metrics).items():
            split_rows.append(
                {
                    "stage": stage,
                    "experiment_group": group,
                    "experiment_name": experiment_name,
                    "experiment_dir": str(exp_dir),
                    "formal_dataset": formal_dataset,
                    "seed": row["seed"],
                    "split": split_name,
                    "metric": metric_name,
                    "value": value,
                    "source": "metrics_json",
                    "availability": "recorded",
                }
            )

    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for split_name in ("validation", "test"):
        split_metrics = metrics.get(split_name, {}) or {}
        cm = split_metrics.get("confusion_matrix") or []
        per_class = split_metrics.get("per_class") or []
        supports = [sum(int(v) for v in row_cm) for row_cm in cm] if cm else []
        for item in per_class:
            cls = int(item.get("class_index", 0))
            per_class_rows.append(
                {
                    "stage": stage,
                    "experiment_group": group,
                    "experiment_name": experiment_name,
                    "experiment_dir": str(exp_dir),
                    "formal_dataset": formal_dataset,
                    "seed": row["seed"],
                    "split": split_name,
                    "class_index": cls,
                    "class_angle": angle_values[cls] if cls < len(angle_values) else "",
                    "precision": item.get("precision", ""),
                    "recall": item.get("recall", ""),
                    "f1": item.get("f1", ""),
                    "support": supports[cls] if cls < len(supports) else "",
                    "source": "metrics_json",
                    "availability": "recorded",
                }
            )
        for i, row_cm in enumerate(cm):
            row_sum = sum(int(v) for v in row_cm)
            for j, count in enumerate(row_cm):
                count_int = int(count)
                confusion_rows.append(
                    {
                        "stage": stage,
                        "experiment_group": group,
                        "experiment_name": experiment_name,
                        "experiment_dir": str(exp_dir),
                        "formal_dataset": formal_dataset,
                        "seed": row["seed"],
                        "split": split_name,
                        "true_class_index": i,
                        "true_angle": angle_values[i] if i < len(angle_values) else "",
                        "pred_class_index": j,
                        "pred_angle": angle_values[j] if j < len(angle_values) else "",
                        "count": count_int,
                        "row_normalized": count_int / row_sum if row_sum else 0.0,
                        "source": "metrics_json",
                        "availability": "recorded",
                    }
                )

    prediction_rows = []
    for split_name in ("test",):
        pred_path = exp_dir / "predictions.csv"
        prediction_rows.append(
            {
                "stage": stage,
                "experiment_group": group,
                "experiment_name": experiment_name,
                "experiment_dir": str(exp_dir),
                "formal_dataset": formal_dataset,
                "seed": row["seed"],
                "split": split_name,
                "predictions_path": str(pred_path) if _exists(pred_path) else "",
                "has_predictions": _exists(pred_path),
                "source": "standard_training_run",
                "notes": "standard runner saves test predictions only",
            }
        )

    completeness = {
        "stage": stage,
        "experiment_group": group,
        "experiment_name": experiment_name,
        "experiment_dir": str(exp_dir),
        "formal_dataset": formal_dataset,
        "has_metrics_json": _exists(metrics_path),
        "has_metadata_json": _exists(metadata_path),
        "has_config_yaml": _exists(exp_dir / "config.yaml"),
        "has_training_log_csv": _exists(training_log_path),
        "has_best_model_pth": _exists(exp_dir / "best_model.pth"),
        "has_last_checkpoint_pth": _exists(exp_dir / "last_checkpoint.pth"),
        "has_test_predictions_csv": _exists(exp_dir / "predictions.csv"),
        "has_test_confusion_matrix_csv": _exists(exp_dir / "confusion_matrix.csv"),
        "train_per_class_available": False,
        "train_confusion_available": False,
        "validation_per_class_available": bool((metrics.get("validation") or {}).get("per_class")),
        "test_per_class_available": bool((metrics.get("test") or {}).get("per_class")),
        "can_recompute_train_val_test": _exists(exp_dir / "best_model.pth") and _exists(exp_dir / "config.yaml"),
    }
    return row, epoch_rows, split_rows, per_class_rows, confusion_rows, prediction_rows, completeness


def _catalog_nonstandard_outputs(outputs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(outputs_root.glob("*")):
        if path.is_dir():
            continue
        if path.suffix.lower() not in {".csv", ".json"}:
            continue
        name = path.name
        stage = _stage_from_group(name)
        kind = "summary_or_table"
        if "by_class" in name:
            kind = "by_class"
        elif "mean_std" in name:
            kind = "mean_std"
        elif "summary" in name:
            kind = "summary"
        elif "oracle" in name:
            kind = "oracle_or_diagnostic"
        rows.append(
            {
                "stage": stage,
                "file_name": name,
                "path": str(path),
                "suffix": path.suffix.lower(),
                "kind": kind,
                "last_write_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "size_bytes": path.stat().st_size,
                "result_role": "diagnostic_or_aggregate",
            }
        )
    return rows


def _copy_nonstandard_outputs(nonstandard_rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in nonstandard_rows:
        src = Path(str(row.get("path", "")))
        if not _exists(src):
            continue
        dst = out_dir / src.name
        shutil.copy2(_long_path(src), _long_path(dst))


def _aggregate_seed(rows_all: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "train_accuracy_at_best_epoch",
        "train_mae_argmax_at_best_epoch",
        "train_macro_f1_at_best_epoch",
        "validation_accuracy",
        "validation_mae_argmax",
        "validation_p90_error",
        "validation_macro_f1",
        "validation_high_angle_macro_f1",
        "test_accuracy",
        "test_mae_argmax",
        "test_p90_error",
        "test_macro_f1",
        "test_high_angle_macro_f1",
        "test_far_error_rate_abs_ge_20",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_all:
        grouped[str(row.get("setting_key", ""))].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for _, items in sorted(grouped.items(), key=lambda pair: pair[0]):
        first = items[0]
        out = {
            "stage": first.get("stage", ""),
            "experiment_group": first.get("experiment_group", ""),
            "formal_dataset": first.get("formal_dataset", ""),
            "dataset": first.get("dataset", ""),
            "modalities": first.get("modalities", ""),
            "model": first.get("model", ""),
            "fusion_mode": first.get("fusion_mode", ""),
            "handcrafted_features": first.get("handcrafted_features", ""),
            "loss": first.get("loss", ""),
            "label_encoding": first.get("label_encoding", ""),
            "loss_extra": first.get("loss_extra", ""),
            "result_role": first.get("result_role", ""),
            "n_runs": len(items),
            "seeds": ";".join(str(item.get("seed", "")) for item in items),
            "setting_key": first.get("setting_key", ""),
        }
        for metric in metrics:
            values = [_safe_float(item.get(metric)) for item in items]
            values = [v for v in values if v is not None]
            if values:
                out[f"{metric}_mean"] = mean(values)
                out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
                out[f"{metric}_min"] = min(values)
                out[f"{metric}_max"] = max(values)
            else:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_std"] = ""
                out[f"{metric}_min"] = ""
                out[f"{metric}_max"] = ""
        aggregate_rows.append(out)
    return aggregate_rows


def _copy_stage_tables(rows_all: list[dict[str, Any]], out_dir: Path) -> None:
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_all:
        by_stage[str(row.get("stage", "other"))].append(row)
    for stage, rows in sorted(by_stage.items()):
        _write_csv(out_dir / f"{stage}_run_summary.csv", rows)


def _write_report(
    report_path: Path,
    rows_all: list[dict[str, Any]],
    completeness_rows: list[dict[str, Any]],
    nonstandard_rows: list[dict[str, Any]],
) -> None:
    by_stage = defaultdict(int)
    by_dataset = defaultdict(int)
    for row in rows_all:
        by_stage[row.get("stage", "other")] += 1
        by_dataset[row.get("formal_dataset", "unknown")] += 1

    missing_training = [row for row in completeness_rows if not row.get("has_training_log_csv")]
    missing_best_model = [row for row in completeness_rows if not row.get("has_best_model_pth")]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Timepix result export completeness report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Standard training runs exported: {len(rows_all)}",
        f"Nonstandard root output files indexed: {len(nonstandard_rows)}",
        "",
        "## Runs by stage",
        "",
    ]
    for key, value in sorted(by_stage.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Runs by formal dataset", ""])
    for key, value in sorted(by_dataset.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Availability notes",
            "",
            "- Validation/test per-class metrics and confusion matrices are read from metrics.json when present.",
            "- Train split overall metrics are derived from training_log.csv at best_epoch.",
            "- Historical standard runs do not save train split per-class metrics or train confusion matrices.",
            "- Recompute train/validation/test predictions from best_model.pth only for selected paper-critical runs if needed.",
            "",
            f"Runs missing training_log.csv: {len(missing_training)}",
            f"Runs missing best_model.pth: {len(missing_best_model)}",
        ]
    )
    if missing_training:
        lines.extend(["", "### Missing training_log examples"])
        for row in missing_training[:20]:
            lines.append(f"- {row.get('experiment_group')} / {row.get('experiment_name')}")
    if missing_best_model:
        lines.extend(["", "### Missing best_model examples"])
        for row in missing_best_model[:20]:
            lines.append(f"- {row.get('experiment_group')} / {row.get('experiment_name')}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export(project_root: Path, analysis_root: Path) -> None:
    outputs_root = project_root / "outputs"
    timestamp = datetime.now().isoformat(timespec="seconds")

    generated_dirs = [
        "00_inventory",
        "01_run_level",
        "02_epoch_history",
        "03_split_metrics",
        "04_per_class",
        "05_confusion_matrix",
        "06_predictions_index",
        "07_seed_aggregate",
        "08_experiment_stage_tables",
        "09_paper_ready_tables",
        "10_nonstandard_outputs",
        "99_reports",
    ]
    for dirname in generated_dirs:
        directory = analysis_root / dirname
        directory.mkdir(parents=True, exist_ok=True)
        for old_file in directory.glob("*"):
            if old_file.is_file() and old_file.suffix.lower() in {".csv", ".md", ".txt"}:
                old_file.unlink()

    standard_metrics = _scan_standard_runs(outputs_root)
    runs_all: list[dict[str, Any]] = []
    epoch_rows_all: list[dict[str, Any]] = []
    split_rows_all: list[dict[str, Any]] = []
    per_class_rows_all: list[dict[str, Any]] = []
    confusion_rows_all: list[dict[str, Any]] = []
    prediction_rows_all: list[dict[str, Any]] = []
    completeness_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []

    for metrics_path in standard_metrics:
        (
            run_row,
            epoch_rows,
            split_rows,
            per_class_rows,
            confusion_rows,
            prediction_rows,
            completeness,
        ) = _extract_run(metrics_path, project_root)
        runs_all.append(run_row)
        epoch_rows_all.extend(epoch_rows)
        split_rows_all.extend(split_rows)
        per_class_rows_all.extend(per_class_rows)
        confusion_rows_all.extend(confusion_rows)
        prediction_rows_all.extend(prediction_rows)
        completeness_rows.append(completeness)
        inventory_rows.append(
            {
                "stage": run_row.get("stage", ""),
                "experiment_group": run_row.get("experiment_group", ""),
                "experiment_name": run_row.get("experiment_name", ""),
                "formal_dataset": run_row.get("formal_dataset", ""),
                "purpose": "",
                "result_role": run_row.get("result_role", ""),
                "status": "exported_current_snapshot",
                "config_path": run_row.get("config_path", ""),
                "output_source": run_row.get("experiment_dir", ""),
                "is_official": run_row.get("result_role") in {"official_or_candidate"},
                "is_deprecated": run_row.get("result_role") == "deprecated",
                "notes": run_row.get("deprecated_reason", ""),
            }
        )

    seed_aggregate = _aggregate_seed(runs_all)
    nonstandard_rows = _catalog_nonstandard_outputs(outputs_root)

    _write_csv(analysis_root / "00_inventory" / "experiment_inventory.csv", inventory_rows)
    _write_csv(analysis_root / "00_inventory" / "nonstandard_outputs_index.csv", nonstandard_rows)
    _write_csv(analysis_root / "00_inventory" / "data_completeness.csv", completeness_rows)
    _copy_nonstandard_outputs(nonstandard_rows, analysis_root / "10_nonstandard_outputs")
    _write_csv(analysis_root / "01_run_level" / "runs_all.csv", runs_all)
    _write_csv(analysis_root / "02_epoch_history" / "training_log_all.csv", epoch_rows_all)
    _write_csv(analysis_root / "03_split_metrics" / "split_metrics_long.csv", split_rows_all)
    _write_csv(analysis_root / "04_per_class" / "per_class_metrics_all.csv", per_class_rows_all)
    _write_csv(analysis_root / "05_confusion_matrix" / "confusion_matrix_long.csv", confusion_rows_all)
    _write_csv(analysis_root / "06_predictions_index" / "predictions_index.csv", prediction_rows_all)
    _write_csv(analysis_root / "07_seed_aggregate" / "seed_aggregate_all.csv", seed_aggregate)
    _copy_stage_tables(runs_all, analysis_root / "08_experiment_stage_tables")
    paper_overview_fields = [
        "stage",
        "experiment_group",
        "formal_dataset",
        "model",
        "modalities",
        "fusion_mode",
        "handcrafted_features",
        "loss",
        "label_encoding",
        "loss_extra",
        "result_role",
        "n_runs",
        "seeds",
        "train_accuracy_at_best_epoch_mean",
        "validation_accuracy_mean",
        "validation_accuracy_std",
        "test_accuracy_mean",
        "test_accuracy_std",
        "validation_mae_argmax_mean",
        "test_mae_argmax_mean",
        "validation_macro_f1_mean",
        "test_macro_f1_mean",
        "test_high_angle_macro_f1_mean",
        "test_far_error_rate_abs_ge_20_mean",
    ]
    _write_csv(
        analysis_root / "09_paper_ready_tables" / "stage_metric_overview.csv",
        seed_aggregate,
        fieldnames=paper_overview_fields,
    )
    _write_report(
        analysis_root / "99_reports" / "data_completeness_report.md",
        runs_all,
        completeness_rows,
        nonstandard_rows,
    )
    (analysis_root / "99_reports" / "EXPORT_TIMESTAMP.txt").write_text(timestamp + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export normalized Timepix result tables.")
    parser.add_argument("--project-root", default=".", help="Project root containing outputs/.")
    parser.add_argument("--out-dir", default=r"E:\Analysis\Timepix_results", help="Analysis output directory.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    analysis_root = Path(args.out_dir).resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)
    export(project_root, analysis_root)
    print(f"Exported result tables to {analysis_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
