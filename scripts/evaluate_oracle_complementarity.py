#!/usr/bin/env python
"""Evaluate oracle complementarity from trained Timepix checkpoints.

This diagnostic reloads existing runs and recomputes logits on deterministic
train/val/test loaders. It is intended for A4b-3a/b:

- ToT-vs-ToT seed oracle control.
- Validation/test oracle checks between ToT and a relative-ToA candidate.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timepix.config import load_yaml, resolve_project_path
from timepix.config_validation import validate_experiment_config
from timepix.training.metrics import classification_metrics


DEFAULT_EXPERIMENT_ROOT = Path("outputs/experiments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A4b oracle complementarity diagnostics")
    parser.add_argument("--root", default=str(DEFAULT_EXPERIMENT_ROOT), help="Experiment output root")
    parser.add_argument(
        "--mode",
        choices=["both", "tot-seed-control", "tot-vs-candidate"],
        default="both",
        help="Which diagnostic to run",
    )
    parser.add_argument(
        "--tot-group",
        action="append",
        default=None,
        help="Experiment group containing ToT runs. May be repeated.",
    )
    parser.add_argument(
        "--candidate-group",
        action="append",
        default=None,
        help="Experiment group containing ToT+ToA candidate runs. May be repeated.",
    )
    parser.add_argument("--splits", default="val,test", help="Comma-separated splits: train,val,test")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional training seed filter")
    parser.add_argument("--data-root", default=None, help="Override dataset.root for this machine")
    parser.add_argument("--num-workers", type=int, default=0, help="Override dataloader num_workers for inference")
    parser.add_argument(
        "--candidate-toa-transform",
        default="relative_minmax",
        help="Candidate data.toa_transform filter, or 'any'",
    )
    parser.add_argument(
        "--candidate-add-hit-mask",
        choices=["false", "true", "any"],
        default="false",
        help="Candidate data.add_hit_mask filter",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["auto", "true", "false"],
        default="auto",
        help="Use CUDA autocast for inference. auto follows each source config.",
    )
    parser.add_argument("--output-json", default=None, help="Output JSON path")
    parser.add_argument("--output-summary", default=None, help="Output summary CSV path")
    parser.add_argument("--output-by-class", default=None, help="Output per-class CSV path")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _metadata_path(run_dir: Path) -> Path:
    path = run_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {run_dir}")
    return path


def _config_path(run_dir: Path) -> Path:
    path = run_dir / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config.yaml: {run_dir}")
    return path


def _checkpoint_path(run_dir: Path) -> Path:
    path = run_dir / "best_model.pth"
    if not path.exists():
        raise FileNotFoundError(f"Missing best_model.pth: {run_dir}")
    return path


def _load_config(run_dir: Path) -> dict[str, Any]:
    cfg = load_yaml(_config_path(run_dir))
    validate_experiment_config(cfg)
    return cfg


def _modalities(metadata: dict[str, Any]) -> tuple[str, ...]:
    dataset = metadata.get("dataset", {})
    data_info = metadata.get("data_info", {})
    return tuple(dataset.get("modalities") or data_info.get("modalities") or [])


def _training_seed(metadata: dict[str, Any]) -> int | None:
    seed = metadata.get("training", {}).get("seed")
    return int(seed) if seed is not None else None


def _toa_transform(metadata: dict[str, Any]) -> str:
    data_info = metadata.get("data_info", {})
    data_cfg = metadata.get("data", {})
    return str(data_info.get("toa_transform", data_cfg.get("toa_transform", "none")))


def _add_hit_mask(metadata: dict[str, Any]) -> bool:
    data_info = metadata.get("data_info", {})
    data_cfg = metadata.get("data", {})
    return bool(data_info.get("add_hit_mask", data_cfg.get("add_hit_mask", False)))


def _discover_group_runs(root: Path, groups: list[str], seeds: set[int] | None) -> list[Path]:
    runs: list[Path] = []
    seen: set[Path] = set()
    for group in groups:
        group_root = resolve_project_path(root) / group
        if not group_root.is_dir():
            continue
        for metadata_path in sorted(group_root.glob("*/metadata.json")):
            run_dir = metadata_path.parent
            if run_dir in seen:
                continue
            metadata = _load_json(metadata_path)
            seed = _training_seed(metadata)
            if seeds is not None and seed not in seeds:
                continue
            seen.add(run_dir)
            runs.append(run_dir)
    return runs


def _discover_tot_runs(root: Path, groups: list[str], seeds: set[int] | None) -> dict[int, Path]:
    by_seed: dict[int, Path] = {}
    for run_dir in _discover_group_runs(root, groups, seeds):
        metadata = _load_json(_metadata_path(run_dir))
        seed = _training_seed(metadata)
        if seed is None or _modalities(metadata) != ("ToT",):
            continue
        by_seed.setdefault(seed, run_dir)
    return by_seed


def _candidate_matches(metadata: dict[str, Any], transform: str, mask_filter: str) -> bool:
    if _modalities(metadata) != ("ToT", "ToA"):
        return False
    if transform != "any" and _toa_transform(metadata) != transform:
        return False
    if mask_filter != "any":
        expected = mask_filter == "true"
        if _add_hit_mask(metadata) != expected:
            return False
    return True


def _discover_candidate_runs(
    root: Path,
    groups: list[str],
    seeds: set[int] | None,
    transform: str,
    mask_filter: str,
) -> dict[int, list[Path]]:
    by_seed: dict[int, list[Path]] = {}
    for run_dir in _discover_group_runs(root, groups, seeds):
        metadata = _load_json(_metadata_path(run_dir))
        seed = _training_seed(metadata)
        if seed is None or not _candidate_matches(metadata, transform, mask_filter):
            continue
        by_seed.setdefault(seed, []).append(run_dir)
    return by_seed


def _dtype_from_config(cfg: dict[str, Any]):
    import torch

    name = str(cfg.get("training", {}).get("mixed_precision_dtype", "float16")).lower().replace("torch.", "")
    return torch.bfloat16 if name in {"bf16", "bfloat16"} else torch.float16


def _autocast_factory(enabled: bool, dtype, device):
    if not enabled or device.type != "cuda":
        return None

    import torch

    def factory():
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            try:
                return torch.amp.autocast("cuda", dtype=dtype)
            except TypeError:
                pass
        return torch.cuda.amp.autocast(dtype=dtype)

    return factory


def _effective_amp(args: argparse.Namespace, cfg: dict[str, Any], device) -> tuple[bool, Any | None]:
    if args.mixed_precision == "false":
        return False, None
    if args.mixed_precision == "true":
        return device.type == "cuda", _dtype_from_config(cfg)
    requested = bool(cfg.get("training", {}).get("mixed_precision", False))
    return requested and device.type == "cuda", _dtype_from_config(cfg)


def _dataset_keys(loader) -> list[str]:
    dataset = loader.dataset
    if hasattr(dataset, "records"):
        return [record.key for record in dataset.records]
    raise TypeError("Expected TimepixDataset with a records attribute")


def _load_run_payload(
    run_dir: Path,
    splits: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    from timepix.data import build_dataloaders
    from timepix.losses import build_loss
    from timepix.models import build_model
    from timepix.training.trainer import evaluate

    cfg = _load_config(run_dir)
    if cfg.get("task", {}).get("type", "classification") != "classification":
        raise ValueError(f"Only classification runs are supported: {run_dir}")
    cfg = deepcopy(cfg)
    cfg.setdefault("training", {})["num_workers"] = int(args.num_workers)
    cfg["training"]["progress_bar"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp, dtype = _effective_amp(args, cfg, device)
    autocast_factory = _autocast_factory(use_amp, dtype, device)

    loaders, data_info = build_dataloaders(cfg, data_root_override=args.data_root, eval_mode=True)
    label_map = data_info["label_map"]
    num_classes = int(data_info["num_classes"])
    angle_values = [float(label_map[i]) for i in range(num_classes)]
    model = build_model(
        cfg,
        input_channels=int(data_info.get("input_channels", len(data_info["modalities"]))),
        num_classes=num_classes,
        task="classification",
        handcrafted_dim=int(data_info["handcrafted_dim"]),
    ).to(device)
    state = torch.load(_checkpoint_path(run_dir), map_location=device)
    model.load_state_dict(state)
    criterion = build_loss(cfg, num_classes, label_map).to(device)

    split_payloads: dict[str, Any] = {}
    for split in splits:
        if split not in loaders:
            raise ValueError(f"Unknown split '{split}'")
        payload = evaluate(
            model,
            loaders[split],
            criterion,
            device,
            "classification",
            progress_bar=False,
            autocast_factory=autocast_factory,
        )
        keys = _dataset_keys(loaders[split])
        labels = payload["labels"].astype(int)
        if len(keys) != len(labels):
            raise ValueError(f"Key/label length mismatch for {run_dir} split={split}")
        split_payloads[split] = {
            "keys": keys,
            "labels": labels,
            "logits": payload["logits"],
        }

    return {
        "run_dir": str(run_dir),
        "metadata": _load_json(_metadata_path(run_dir)),
        "data_info": data_info,
        "angle_values": angle_values,
        "splits": split_payloads,
        "device": str(device),
        "mixed_precision_enabled": use_amp,
    }


def _run_label(run_dir: Path) -> str:
    metadata = _load_json(_metadata_path(run_dir))
    modalities = "+".join(_modalities(metadata))
    seed = _training_seed(metadata)
    transform = _toa_transform(metadata)
    mask = _add_hit_mask(metadata)
    if transform and transform != "none":
        suffix = "mask" if mask else "no_mask"
        return f"{modalities}_{transform}_{suffix}_seed{seed}"
    return f"{modalities}_seed{seed}"


def _validate_pair(base: dict[str, Any], other: dict[str, Any], split: str) -> None:
    base_split = base["splits"][split]
    other_split = other["splits"][split]
    if base["angle_values"] != other["angle_values"]:
        raise ValueError("Label maps/angle values do not match")
    if base_split["keys"] != other_split["keys"]:
        raise ValueError(f"Sample keys are not aligned for split={split}")
    if not np.array_equal(base_split["labels"], other_split["labels"]):
        raise ValueError(f"Labels are not aligned for split={split}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _pair_summary(
    comparator_type: str,
    split: str,
    base_run: Path,
    base: dict[str, Any],
    other_run: Path,
    other: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_pair(base, other, split)
    angle_values = np.asarray(base["angle_values"], dtype=float)
    labels = base["splits"][split]["labels"]
    base_logits = base["splits"][split]["logits"]
    other_logits = other["splits"][split]["logits"]

    base_metrics = classification_metrics(base_logits, labels, base["angle_values"])
    other_metrics = classification_metrics(other_logits, labels, base["angle_values"])
    base_preds = _softmax(base_logits).argmax(axis=1)
    other_preds = _softmax(other_logits).argmax(axis=1)
    base_correct = base_preds == labels
    other_correct = other_preds == labels
    base_errors = np.abs(angle_values[base_preds] - angle_values[labels])
    other_errors = np.abs(angle_values[other_preds] - angle_values[labels])
    oracle_correct = np.logical_or(base_correct, other_correct)
    oracle_errors = np.minimum(base_errors, other_errors)
    base_wrong = ~base_correct

    base_acc = float(base_metrics["accuracy"])
    other_acc = float(other_metrics["accuracy"])
    oracle_acc = float(oracle_correct.mean()) if len(labels) else 0.0
    base_mae = float(base_metrics["mae_argmax"])
    other_mae = float(other_metrics["mae_argmax"])
    oracle_mae = float(oracle_errors.mean()) if len(labels) else 0.0

    base_metadata = base["metadata"]
    other_metadata = other["metadata"]
    summary = {
        "comparator_type": comparator_type,
        "split": split,
        "base": _run_label(base_run),
        "other": _run_label(other_run),
        "base_seed": _training_seed(base_metadata),
        "other_seed": _training_seed(other_metadata),
        "base_modalities": "+".join(_modalities(base_metadata)),
        "other_modalities": "+".join(_modalities(other_metadata)),
        "other_toa_transform": _toa_transform(other_metadata),
        "other_add_hit_mask": _add_hit_mask(other_metadata),
        "n": int(len(labels)),
        "base_accuracy": base_acc,
        "other_accuracy": other_acc,
        "base_mae": base_mae,
        "other_mae": other_mae,
        "base_macro_f1": float(base_metrics["macro_f1"]),
        "other_macro_f1": float(other_metrics["macro_f1"]),
        "both_correct": int(np.logical_and(base_correct, other_correct).sum()),
        "base_correct_other_wrong": int(np.logical_and(base_correct, ~other_correct).sum()),
        "base_wrong_other_correct": int(np.logical_and(~base_correct, other_correct).sum()),
        "both_wrong": int(np.logical_and(~base_correct, ~other_correct).sum()),
        "base_wrong_count": int(base_wrong.sum()),
        "other_better_when_base_wrong": int((other_errors[base_wrong] < base_errors[base_wrong]).sum()),
        "other_equal_when_base_wrong": int((other_errors[base_wrong] == base_errors[base_wrong]).sum()),
        "other_worse_when_base_wrong": int((other_errors[base_wrong] > base_errors[base_wrong]).sum()),
        "other_better_when_base_wrong_rate": float(
            (other_errors[base_wrong] < base_errors[base_wrong]).mean()
        )
        if base_wrong.any()
        else 0.0,
        "oracle_accuracy": oracle_acc,
        "oracle_accuracy_gain_vs_base": oracle_acc - base_acc,
        "oracle_mae": oracle_mae,
        "oracle_mae_gain_vs_base": base_mae - oracle_mae,
        "base_run": str(base_run),
        "other_run": str(other_run),
    }

    by_class_rows = []
    for class_index, angle in enumerate(angle_values):
        mask = labels == class_index
        if not mask.any():
            continue
        local_base_correct = base_correct[mask]
        local_other_correct = other_correct[mask]
        local_base_errors = base_errors[mask]
        local_other_errors = other_errors[mask]
        local_base_wrong = ~local_base_correct
        local_oracle_correct = np.logical_or(local_base_correct, local_other_correct)
        by_class_rows.append(
            {
                "comparator_type": comparator_type,
                "split": split,
                "class_index": int(class_index),
                "class_angle": float(angle),
                "base": summary["base"],
                "other": summary["other"],
                "base_seed": summary["base_seed"],
                "other_seed": summary["other_seed"],
                "n": int(mask.sum()),
                "base_accuracy": float(local_base_correct.mean()),
                "other_accuracy": float(local_other_correct.mean()),
                "both_correct": int(np.logical_and(local_base_correct, local_other_correct).sum()),
                "base_correct_other_wrong": int(np.logical_and(local_base_correct, ~local_other_correct).sum()),
                "base_wrong_other_correct": int(np.logical_and(~local_base_correct, local_other_correct).sum()),
                "both_wrong": int(np.logical_and(~local_base_correct, ~local_other_correct).sum()),
                "other_better_when_base_wrong": int(
                    (local_other_errors[local_base_wrong] < local_base_errors[local_base_wrong]).sum()
                ),
                "oracle_accuracy": float(local_oracle_correct.mean()),
                "oracle_accuracy_gain_vs_base": float(local_oracle_correct.mean() - local_base_correct.mean()),
                "oracle_mae": float(np.minimum(local_base_errors, local_other_errors).mean()),
            }
        )
    return summary, by_class_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _default_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    stem = "a4b_oracle_complementarity"
    json_path = Path(args.output_json) if args.output_json else Path("outputs") / f"{stem}.json"
    summary_path = Path(args.output_summary) if args.output_summary else Path("outputs") / f"{stem}_summary.csv"
    by_class_path = Path(args.output_by_class) if args.output_by_class else Path("outputs") / f"{stem}_by_class.csv"
    return json_path, summary_path, by_class_path


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not splits:
        raise ValueError("--splits must contain at least one split")
    bad_splits = [split for split in splits if split not in {"train", "val", "test"}]
    if bad_splits:
        raise ValueError(f"Unsupported splits: {bad_splits}")

    tot_groups = args.tot_group or ["a4_modality_comparison", "a4_modality_comparison_seed42"]
    candidate_groups = args.candidate_group or ["a4b_toa_transform", "a4b_toa_transform_seed42"]
    seed_filter = set(args.seeds) if args.seeds else None

    tot_runs = _discover_tot_runs(root, tot_groups, seed_filter)
    candidate_runs = _discover_candidate_runs(
        root,
        candidate_groups,
        seed_filter,
        args.candidate_toa_transform,
        args.candidate_add_hit_mask,
    )

    warnings: list[str] = []
    comparisons: list[tuple[str, Path, Path]] = []
    if args.mode in {"both", "tot-seed-control"}:
        if len(tot_runs) < 2:
            warnings.append("Need at least two ToT seeds for tot-seed-control; skipping that comparison.")
        for left_seed, right_seed in itertools.combinations(sorted(tot_runs), 2):
            comparisons.append(("tot_seed_control", tot_runs[left_seed], tot_runs[right_seed]))

    if args.mode in {"both", "tot-vs-candidate"}:
        shared_seeds = sorted(set(tot_runs) & set(candidate_runs))
        if not shared_seeds:
            warnings.append("No matching ToT/candidate seeds found for tot-vs-candidate.")
        for seed in shared_seeds:
            for candidate_run in candidate_runs[seed]:
                comparisons.append(("tot_vs_candidate", tot_runs[seed], candidate_run))

    if not comparisons:
        for item in warnings:
            print(f"Warning: {item}")
        raise RuntimeError("No comparisons to evaluate")

    cache: dict[Path, dict[str, Any]] = {}

    def payload(run_dir: Path) -> dict[str, Any]:
        if run_dir not in cache:
            cache[run_dir] = _load_run_payload(run_dir, splits, args)
        return cache[run_dir]

    summary_rows: list[dict[str, Any]] = []
    by_class_rows: list[dict[str, Any]] = []
    json_payload: dict[str, Any] = {
        "analysis": "a4b_oracle_complementarity",
        "mode": args.mode,
        "splits": splits,
        "tot_groups": tot_groups,
        "candidate_groups": candidate_groups,
        "candidate_toa_transform": args.candidate_toa_transform,
        "candidate_add_hit_mask": args.candidate_add_hit_mask,
        "warnings": warnings,
        "comparisons": [],
    }

    for comparator_type, base_run, other_run in comparisons:
        base_payload = payload(base_run)
        other_payload = payload(other_run)
        for split in splits:
            summary, by_class = _pair_summary(comparator_type, split, base_run, base_payload, other_run, other_payload)
            summary_rows.append(summary)
            by_class_rows.extend(by_class)
            json_payload["comparisons"].append(summary)
            print(
                "Oracle | "
                f"type={comparator_type} split={split} "
                f"{summary['base']} vs {summary['other']} "
                f"base_acc={summary['base_accuracy']:.4f} "
                f"other_acc={summary['other_accuracy']:.4f} "
                f"oracle_acc={summary['oracle_accuracy']:.4f} "
                f"gain={summary['oracle_accuracy_gain_vs_base']:.4f}"
            )

    json_path, summary_path, by_class_path = _default_outputs(args)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary_path, summary_rows)
    _write_csv(by_class_path, by_class_rows)

    for item in warnings:
        print(f"Warning: {item}")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote per-class CSV: {by_class_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

