#!/usr/bin/env python
"""Summarize Timepix experiment metadata into a CSV table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_EXPERIMENT_ROOT = Path("outputs/experiments")


SUMMARY_FIELDS = [
    "experiment_group",
    "experiment_name",
    "experiment_dir",
    "dataset",
    "particle",
    "modalities",
    "input_channels",
    "toa_transform",
    "add_hit_mask",
    "handcrafted_enabled",
    "handcrafted_dim",
    "handcrafted_feature_count",
    "handcrafted_features",
    "handcrafted_source_modalities",
    "task",
    "model",
    "fusion_mode",
    "expert_gate_freeze_experts",
    "expert_gate_init_bias_to_candidate",
    "conv1_kernel_size",
    "conv1_stride",
    "conv1_padding",
    "dropout",
    "feature_dim",
    "hidden_dim",
    "image_size",
    "patch_size",
    "loss",
    "label_encoding",
    "gaussian_sigma",
    "expected_mae_weight",
    "emd_weight",
    "emd_p",
    "emd_angle_weighted",
    "best_epoch",
    "stopped_epoch",
    "max_epochs",
    "early_stopped",
    "seed",
    "split_seed",
    "split_path",
    "split_manifest_hash",
    "learning_rate",
    "batch_size",
    "scheduler",
    "mixed_precision",
    "mixed_precision_dtype",
    "mixed_precision_enabled",
    "fit_seconds",
    "test_seconds",
    "total_seconds",
    "val_accuracy",
    "test_accuracy",
    "val_mae_argmax",
    "test_mae_argmax",
    "val_p90_error",
    "test_p90_error",
    "val_macro_f1",
    "test_macro_f1",
    "val_high_angle_macro_f1",
    "test_high_angle_macro_f1",
    "val_confusion_45_50",
    "test_confusion_45_50",
    "val_confusion_60_70",
    "test_confusion_60_70",
    "val_far_error_rate_abs_ge_20",
    "test_far_error_rate_abs_ge_20",
    "val_gate_tot_mean",
    "test_gate_tot_mean",
    "val_gate_toa_mean",
    "test_gate_toa_mean",
    "val_gate_primary_mean",
    "test_gate_primary_mean",
    "val_gate_candidate_mean",
    "test_gate_candidate_mean",
    "val_film_gamma_abs_mean",
    "test_film_gamma_abs_mean",
    "val_film_beta_abs_mean",
    "test_film_beta_abs_mean",
    "params_total",
    "git_commit",
    "git_dirty",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment outputs")
    parser.add_argument("--root", default=None, help="Experiment output root or group directory")
    parser.add_argument("--out", default=None, help="Output CSV path")
    parser.add_argument("--group", default=None, help="Summarize outputs/experiments/<group>")
    parser.add_argument("--all", action="store_true", help="Recursively summarize all experiment groups")
    return parser.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _metadata_paths(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/metadata.json" if recursive else "*/metadata.json"
    return sorted(root.glob(pattern))


def _infer_group(metadata: dict, metadata_path: Path, root: Path, recursive: bool) -> str:
    group = metadata.get("experiment_group")
    if group:
        return str(group)
    if recursive:
        try:
            relative = metadata_path.relative_to(root)
            if len(relative.parts) >= 3:
                return relative.parts[0]
        except ValueError:
            pass
    return "default"


def _row_from_metadata(metadata: dict, metadata_path: Path, root: Path, recursive: bool) -> dict:
    dataset = metadata.get("dataset", {})
    metrics = metadata.get("metrics", {})
    val = metrics.get("validation", {})
    test = metrics.get("test", {})
    model = metadata.get("model", {})
    loss = metadata.get("loss", {})
    training = metadata.get("training", {})
    data = metadata.get("data", {})
    mixed_precision = metadata.get("mixed_precision", {})
    timing = metadata.get("timing", {})
    git = metadata.get("git", {})
    data_info = metadata.get("data_info", {})
    handcrafted_features = data_info.get("handcrafted_features", [])
    feature_source_modalities = data_info.get("feature_source_modalities", [])
    val_diag = metrics.get("validation_diagnostics", {})
    test_diag = metrics.get("test_diagnostics", {})

    def diag_mean(source: dict, key: str):
        value = source.get(key, {})
        if isinstance(value, dict):
            return value.get("mean")
        return None

    return {
        "experiment_group": _infer_group(metadata, metadata_path, root, recursive),
        "experiment_name": metadata.get("experiment_name"),
        "experiment_dir": metadata.get("experiment_dir"),
        "dataset": dataset.get("name"),
        "particle": dataset.get("particle"),
        "modalities": "+".join(dataset.get("modalities", [])),
        "input_channels": data_info.get("input_channels"),
        "toa_transform": data_info.get("toa_transform", data.get("toa_transform", "none")),
        "add_hit_mask": data_info.get("add_hit_mask", data.get("add_hit_mask", False)),
        "handcrafted_enabled": bool(data_info.get("handcrafted_dim", 0)),
        "handcrafted_dim": data_info.get("handcrafted_dim"),
        "handcrafted_feature_count": len(handcrafted_features) if isinstance(handcrafted_features, list) else None,
        "handcrafted_features": ";".join(handcrafted_features) if isinstance(handcrafted_features, list) else "",
        "handcrafted_source_modalities": "+".join(feature_source_modalities)
        if isinstance(feature_source_modalities, list)
        else "",
        "task": metadata.get("task"),
        "model": model.get("name"),
        "fusion_mode": model.get("fusion_mode"),
        "expert_gate_freeze_experts": (model.get("expert_gate") or {}).get("freeze_experts"),
        "expert_gate_init_bias_to_candidate": (model.get("expert_gate") or {}).get("init_bias_to_candidate"),
        "conv1_kernel_size": model.get("conv1_kernel_size", model.get("kernel_size")),
        "conv1_stride": model.get("conv1_stride", model.get("stride")),
        "conv1_padding": model.get("conv1_padding", model.get("padding")),
        "dropout": model.get("dropout"),
        "feature_dim": model.get("feature_dim"),
        "hidden_dim": model.get("hidden_dim"),
        "image_size": model.get("image_size"),
        "patch_size": model.get("patch_size"),
        "loss": loss.get("name"),
        "label_encoding": loss.get("label_encoding"),
        "gaussian_sigma": loss.get("gaussian_sigma"),
        "expected_mae_weight": loss.get("expected_mae_weight"),
        "emd_weight": loss.get("emd_weight"),
        "emd_p": loss.get("emd_p"),
        "emd_angle_weighted": loss.get("emd_angle_weighted"),
        "best_epoch": metrics.get("best_epoch"),
        "stopped_epoch": metrics.get("stopped_epoch"),
        "max_epochs": metrics.get("max_epochs"),
        "early_stopped": metrics.get("early_stopped"),
        "seed": training.get("seed"),
        "split_seed": data_info.get("split_seed"),
        "split_path": data_info.get("split_path"),
        "split_manifest_hash": metadata.get("split_manifest_hash"),
        "learning_rate": training.get("learning_rate"),
        "batch_size": training.get("batch_size"),
        "scheduler": training.get("scheduler"),
        "mixed_precision": training.get("mixed_precision", False),
        "mixed_precision_dtype": training.get("mixed_precision_dtype", "float16"),
        "mixed_precision_enabled": mixed_precision.get("enabled"),
        "fit_seconds": timing.get("fit_seconds", metrics.get("fit_seconds")),
        "test_seconds": timing.get("test_seconds", metrics.get("test_seconds")),
        "total_seconds": timing.get("total_seconds", metrics.get("total_seconds")),
        "val_accuracy": val.get("accuracy"),
        "test_accuracy": test.get("accuracy"),
        "val_mae_argmax": val.get("mae_argmax", val.get("mae")),
        "test_mae_argmax": test.get("mae_argmax", test.get("mae")),
        "val_p90_error": val.get("p90_error"),
        "test_p90_error": test.get("p90_error"),
        "val_macro_f1": val.get("macro_f1"),
        "test_macro_f1": test.get("macro_f1"),
        "val_high_angle_macro_f1": val.get("high_angle_macro_f1"),
        "test_high_angle_macro_f1": test.get("high_angle_macro_f1"),
        "val_confusion_45_50": val.get("confusion_45_50"),
        "test_confusion_45_50": test.get("confusion_45_50"),
        "val_confusion_60_70": val.get("confusion_60_70"),
        "test_confusion_60_70": test.get("confusion_60_70"),
        "val_far_error_rate_abs_ge_20": val.get("far_error_rate_abs_ge_20"),
        "test_far_error_rate_abs_ge_20": test.get("far_error_rate_abs_ge_20"),
        "val_gate_tot_mean": diag_mean(val_diag, "gate_tot"),
        "test_gate_tot_mean": diag_mean(test_diag, "gate_tot"),
        "val_gate_toa_mean": diag_mean(val_diag, "gate_toa"),
        "test_gate_toa_mean": diag_mean(test_diag, "gate_toa"),
        "val_gate_primary_mean": diag_mean(val_diag, "gate_primary"),
        "test_gate_primary_mean": diag_mean(test_diag, "gate_primary"),
        "val_gate_candidate_mean": diag_mean(val_diag, "gate_candidate"),
        "test_gate_candidate_mean": diag_mean(test_diag, "gate_candidate"),
        "val_film_gamma_abs_mean": diag_mean(val_diag, "film_gamma_abs"),
        "test_film_gamma_abs_mean": diag_mean(test_diag, "film_gamma_abs"),
        "val_film_beta_abs_mean": diag_mean(val_diag, "film_beta_abs"),
        "test_film_beta_abs_mean": diag_mean(test_diag, "film_beta_abs"),
        "params_total": metadata.get("param_count", {}).get("total"),
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
    }


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, bool]:
    if args.group and args.all:
        raise ValueError("--group and --all cannot be used together")

    if args.group:
        root = DEFAULT_EXPERIMENT_ROOT / args.group
        out = Path(args.out) if args.out else Path("outputs") / f"{args.group}_summary.csv"
        return root, out, False

    if args.all:
        root = DEFAULT_EXPERIMENT_ROOT
        out = Path(args.out) if args.out else Path("outputs/experiment_summary.csv")
        return root, out, True

    if args.root:
        root = Path(args.root)
        out = Path(args.out) if args.out else Path("outputs/experiment_summary.csv")
        return root, out, False

    root = DEFAULT_EXPERIMENT_ROOT
    out = Path(args.out) if args.out else Path("outputs/experiment_summary.csv")
    return root, out, True


def main() -> int:
    args = parse_args()
    root, out, recursive = _resolve_inputs(args)
    rows = [
        _row_from_metadata(_load_json(metadata_path), metadata_path, root, recursive)
        for metadata_path in _metadata_paths(root, recursive)
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
