#!/usr/bin/env python
"""Dump logits/probabilities for saved Timepix experiment runs.

This utility is intended for post-hoc diagnostics that need validation logits
from already-trained checkpoints. The standard runner saves test predictions
only; this script reloads the best checkpoint and writes deterministic
train/val/test predictions without retraining.
"""

from __future__ import annotations

import argparse
import csv
import glob
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump split predictions from saved Timepix runs")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH_OR_GLOB",
        help="Run directory, metadata.json, or glob to dump. Can be repeated.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for dumped CSV files")
    parser.add_argument("--data-root", default=None, help="Override dataset.root for this machine")
    parser.add_argument("--num-workers", type=int, default=0, help="Override dataloader workers")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to dump",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["auto", "true", "false"],
        default="auto",
        help="Use CUDA autocast. auto follows source config.",
    )
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


def _run_dir(path: str | Path) -> Path:
    candidate = resolve_project_path(path)
    if candidate.name == "metadata.json":
        candidate = candidate.parent
    if not candidate.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {candidate}")
    return candidate


def _parse_run_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"--run must be LABEL=PATH_OR_GLOB, got: {value}")
    label, pattern = value.split("=", 1)
    label = label.strip()
    pattern = pattern.strip()
    if not label or not pattern:
        raise ValueError(f"--run must be LABEL=PATH_OR_GLOB, got: {value}")
    return label, pattern


def _discover_runs(specs: list[str]) -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    seen: set[tuple[str, Path]] = set()
    for item in specs:
        label, pattern = _parse_run_arg(item)
        raw_matches = glob.glob(str(resolve_project_path(pattern)))
        if not raw_matches:
            raw_matches = glob.glob(pattern)
        if not raw_matches:
            raise FileNotFoundError(f"No run matches pattern for {label}: {pattern}")
        for match in sorted(raw_matches):
            run_dir = _run_dir(match)
            key = (label, run_dir)
            if key not in seen:
                seen.add(key)
                discovered.append(key)
    return discovered


def _training_seed(metadata: dict[str, Any]) -> int | None:
    seed = metadata.get("training", {}).get("seed")
    return int(seed) if seed is not None else None


def _dtype_from_config(cfg: dict[str, Any]):
    import torch

    name = str(cfg.get("training", {}).get("mixed_precision_dtype", "float16")).lower().replace("torch.", "")
    return torch.bfloat16 if name in {"bf16", "bfloat16"} else torch.float16


def _autocast_factory(enabled: bool, dtype, device):
    if not enabled or device.type != "cuda":
        return None

    import torch

    def factory():
        return torch.autocast(device_type="cuda", dtype=dtype)

    return factory


def _effective_amp(args: argparse.Namespace, cfg: dict[str, Any], device):
    if args.mixed_precision == "true":
        return device.type == "cuda", _dtype_from_config(cfg)
    if args.mixed_precision == "false":
        return False, _dtype_from_config(cfg)
    return bool(cfg.get("training", {}).get("mixed_precision", False)) and device.type == "cuda", _dtype_from_config(cfg)


def _dataset_keys(loader) -> list[str]:
    dataset = loader.dataset
    records = getattr(dataset, "records", None)
    if records is None:
        return [str(i) for i in range(len(dataset))]
    keys: list[str] = []
    for record in records:
        key = getattr(record, "key", None)
        if key is None:
            key = getattr(record, "sample_key", None)
        if key is None:
            key = getattr(record, "path", None)
        keys.append(str(key))
    return keys


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _write_predictions(
    path: Path,
    *,
    label: str,
    run_dir: Path,
    split: str,
    keys: list[str],
    labels: np.ndarray,
    logits: np.ndarray,
    class_names: list[str],
) -> None:
    probs = _softmax(logits)
    pred_ids = probs.argmax(axis=1)
    headers = [
        "label",
        "run_dir",
        "split",
        "row",
        "sample_key",
        "true_label",
        "pred_label",
        "true_class",
        "pred_class",
    ]
    headers.extend(f"logit_{name}" for name in class_names)
    headers.extend(f"prob_{name}" for name in class_names)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for idx, key in enumerate(keys):
            row = {
                "label": label,
                "run_dir": str(run_dir),
                "split": split,
                "row": idx,
                "sample_key": key,
                "true_label": int(labels[idx]),
                "pred_label": int(pred_ids[idx]),
                "true_class": class_names[int(labels[idx])],
                "pred_class": class_names[int(pred_ids[idx])],
            }
            for cls_idx, name in enumerate(class_names):
                row[f"logit_{name}"] = f"{float(logits[idx, cls_idx]):.10g}"
                row[f"prob_{name}"] = f"{float(probs[idx, cls_idx]):.10g}"
            writer.writerow(row)


def _dump_run(label: str, run_dir: Path, args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    import torch

    from timepix.data import build_dataloaders
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
    class_names = [str(label_map[i]) for i in range(int(data_info["num_classes"]))]
    model = build_model(
        cfg,
        input_channels=int(data_info.get("input_channels", len(data_info["modalities"]))),
        num_classes=int(data_info["num_classes"]),
        task="classification",
        handcrafted_dim=int(data_info["handcrafted_dim"]),
    ).to(device)
    state = torch.load(_checkpoint_path(run_dir), map_location=device)
    model.load_state_dict(state)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    metadata = _load_json(_metadata_path(run_dir))
    seed = _training_seed(metadata)
    safe_label = _safe_name(label)
    manifest_rows: list[dict[str, Any]] = []
    for split in args.splits:
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
        logits = payload["logits"]
        if len(keys) != len(labels):
            raise ValueError(f"Key/label length mismatch for {run_dir} split={split}")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(f"Logit/label length mismatch for {run_dir} split={split}")
        seed_part = f"seed{seed}" if seed is not None else "seedNA"
        output_path = out_dir / f"{safe_label}_{seed_part}_{split}_predictions.csv"
        _write_predictions(
            output_path,
            label=label,
            run_dir=run_dir,
            split=split,
            keys=keys,
            labels=labels,
            logits=logits,
            class_names=class_names,
        )
        manifest_rows.append(
            {
                "label": label,
                "seed": "" if seed is None else seed,
                "split": split,
                "run_dir": str(run_dir),
                "output": str(output_path),
                "n": int(labels.shape[0]),
                "num_classes": int(data_info["num_classes"]),
                "class_names": ";".join(class_names),
                "mixed_precision_enabled": bool(use_amp),
            }
        )
    return manifest_rows


def main() -> int:
    args = parse_args()
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = _discover_runs(args.run)
    rows: list[dict[str, Any]] = []
    for label, run_dir in runs:
        rows.extend(_dump_run(label, run_dir, args, out_dir))

    manifest_path = out_dir / "dump_manifest.csv"
    if rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} dumped split files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
