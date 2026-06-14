#!/usr/bin/env python
"""Dump and summarize per-sample GMU gate diagnostics from saved runs."""

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
    parser = argparse.ArgumentParser(description="Analyze GMU gate behavior from saved Timepix runs")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH_OR_GLOB",
        help="Run directory, metadata.json, or glob. Can be repeated.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory")
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


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


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


def _training_seed(metadata: dict[str, Any]) -> int | None:
    seed = metadata.get("training", {}).get("seed")
    return int(seed) if seed is not None else None


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


def _dtype_from_config(cfg: dict[str, Any]):
    import torch

    name = str(cfg.get("training", {}).get("mixed_precision_dtype", "float16")).lower().replace("torch.", "")
    return torch.bfloat16 if name in {"bf16", "bfloat16"} else torch.float16


def _effective_amp(args: argparse.Namespace, cfg: dict[str, Any], device):
    if args.mixed_precision == "true":
        return device.type == "cuda", _dtype_from_config(cfg)
    if args.mixed_precision == "false":
        return False, _dtype_from_config(cfg)
    return bool(cfg.get("training", {}).get("mixed_precision", False)) and device.type == "cuda", _dtype_from_config(cfg)


def _autocast_factory(enabled: bool, dtype, device):
    if not enabled or device.type != "cuda":
        return None

    import torch

    def factory():
        return torch.autocast(device_type="cuda", dtype=dtype)

    return factory


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _bool_group(value: Any) -> str:
    return "correct" if bool(int(value)) else "wrong"


def _cosr_case(row: dict[str, Any]) -> str | None:
    true_cls = str(row["true_class"])
    pred_cls = str(row["pred_class"])
    if true_cls == "Co" and pred_cls == "Co":
        return "Co_correct"
    if true_cls == "Sr" and pred_cls == "Sr":
        return "Sr_correct"
    if true_cls == "Co" and pred_cls == "Sr":
        return "Co_to_Sr"
    if true_cls == "Sr" and pred_cls == "Co":
        return "Sr_to_Co"
    if true_cls in {"Co", "Sr"}:
        return f"{true_cls}_to_other"
    return None


def _summarize_group(rows: list[dict[str, Any]], *, split: str, group_type: str, group: str) -> dict[str, Any]:
    gate_tot = np.asarray([float(row["gate_tot"]) for row in rows], dtype=float)
    gate_toa = np.asarray([float(row["gate_toa"]) for row in rows], dtype=float)
    confidence = np.asarray([float(row.get("confidence", 0.0)) for row in rows], dtype=float)
    return {
        "split": split,
        "group_type": group_type,
        "group": group,
        "n": int(len(rows)),
        "gate_tot_mean": float(gate_tot.mean()) if len(rows) else 0.0,
        "gate_tot_std": float(gate_tot.std(ddof=0)) if len(rows) else 0.0,
        "gate_toa_mean": float(gate_toa.mean()) if len(rows) else 0.0,
        "gate_toa_std": float(gate_toa.std(ddof=0)) if len(rows) else 0.0,
        "confidence_mean": float(confidence.mean()) if len(rows) else 0.0,
    }


def summarize_gate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-sample gate rows by split, class, correctness, and Co/Sr case."""

    summary: list[dict[str, Any]] = []
    splits = sorted({str(row["split"]) for row in rows})
    for split in splits:
        split_rows = [row for row in rows if str(row["split"]) == split]
        summary.append(_summarize_group(split_rows, split=split, group_type="overall", group="all"))

        for cls in sorted({str(row["true_class"]) for row in split_rows}):
            group_rows = [row for row in split_rows if str(row["true_class"]) == cls]
            summary.append(_summarize_group(group_rows, split=split, group_type="true_class", group=cls))

        for cls in sorted({str(row["pred_class"]) for row in split_rows}):
            group_rows = [row for row in split_rows if str(row["pred_class"]) == cls]
            summary.append(_summarize_group(group_rows, split=split, group_type="pred_class", group=cls))

        for correctness in ["correct", "wrong"]:
            group_rows = [row for row in split_rows if _bool_group(row["correct"]) == correctness]
            if group_rows:
                summary.append(_summarize_group(group_rows, split=split, group_type="correctness", group=correctness))

        cosr_pairs: dict[str, list[dict[str, Any]]] = {}
        for row in split_rows:
            case = _cosr_case(row)
            if case is not None:
                cosr_pairs.setdefault(case, []).append(row)
        for case in sorted(cosr_pairs):
            summary.append(_summarize_group(cosr_pairs[case], split=split, group_type="cosr_case", group=case))

    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _dump_run(label: str, run_dir: Path, args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    from timepix.data import build_dataloaders
    from timepix.models import build_model
    from timepix.training.trainer import _unpack_batch

    cfg = _load_config(run_dir)
    if cfg.get("task", {}).get("type", "classification") != "classification":
        raise ValueError(f"Only classification runs are supported: {run_dir}")
    if cfg.get("model", {}).get("name") != "dual_stream_gmu_aux":
        raise ValueError(f"Only dual_stream_gmu_aux runs expose GMU gate diagnostics: {run_dir}")
    cfg = deepcopy(cfg)
    cfg.setdefault("training", {})["num_workers"] = int(args.num_workers)
    cfg["training"]["progress_bar"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp, dtype = _effective_amp(args, cfg, device)
    autocast_factory = _autocast_factory(use_amp, dtype, device)

    loaders, data_info = build_dataloaders(cfg, data_root_override=args.data_root, eval_mode=True)
    class_names = [str(data_info["label_map"][i]) for i in range(int(data_info["num_classes"]))]
    model = build_model(
        cfg,
        input_channels=int(data_info.get("input_channels", len(data_info["modalities"]))),
        num_classes=int(data_info["num_classes"]),
        task="classification",
        handcrafted_dim=int(data_info["handcrafted_dim"]),
    ).to(device)
    state = torch.load(_checkpoint_path(run_dir), map_location=device)
    model.load_state_dict(state)
    model.eval()

    metadata = _load_json(_metadata_path(run_dir))
    seed = _training_seed(metadata)
    safe_label = _safe_name(label)
    all_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for split in args.splits:
        keys = _dataset_keys(loaders[split])
        split_rows: list[dict[str, Any]] = []
        offset = 0
        with torch.no_grad():
            for batch in loaders[split]:
                images, labels, handcrafted = _unpack_batch(batch)
                images = images.to(device)
                labels = labels.to(device)
                handcrafted = handcrafted.to(device) if handcrafted is not None else None
                if autocast_factory is None:
                    output = model(images, handcrafted)
                else:
                    with autocast_factory():
                        output = model(images, handcrafted)
                diagnostics = output.diagnostics or {}
                if "gate_tot" not in diagnostics or "gate_toa" not in diagnostics:
                    raise ValueError(f"Run does not emit gate_tot/gate_toa diagnostics: {run_dir}")
                logits = output.logits.detach().float().cpu().numpy()
                probs = _softmax(logits)
                labels_np = labels.detach().cpu().numpy().astype(int)
                pred_ids = probs.argmax(axis=1).astype(int)
                gate_tot = diagnostics["gate_tot"].detach().float().cpu().numpy().reshape(-1)
                gate_toa = diagnostics["gate_toa"].detach().float().cpu().numpy().reshape(-1)
                for batch_idx in range(labels_np.shape[0]):
                    row_idx = offset + batch_idx
                    true_id = int(labels_np[batch_idx])
                    pred_id = int(pred_ids[batch_idx])
                    row = {
                        "label": label,
                        "seed": "" if seed is None else int(seed),
                        "run_dir": str(run_dir),
                        "split": split,
                        "row": row_idx,
                        "sample_key": keys[row_idx],
                        "true_label": true_id,
                        "pred_label": pred_id,
                        "true_class": class_names[true_id],
                        "pred_class": class_names[pred_id],
                        "correct": int(true_id == pred_id),
                        "confidence": float(probs[batch_idx, pred_id]),
                        "prob_true": float(probs[batch_idx, true_id]),
                        "gate_tot": float(gate_tot[batch_idx]),
                        "gate_toa": float(gate_toa[batch_idx]),
                    }
                    split_rows.append(row)
                offset += labels_np.shape[0]
        if len(split_rows) != len(keys):
            raise ValueError(f"Dumped row count mismatch for {run_dir} split={split}: {len(split_rows)} vs {len(keys)}")
        seed_part = f"seed{seed}" if seed is not None else "seedNA"
        split_path = out_dir / f"{safe_label}_{seed_part}_{split}_gates.csv"
        _write_csv(split_path, split_rows)
        all_rows.extend(split_rows)
        manifest_rows.append(
            {
                "label": label,
                "seed": "" if seed is None else seed,
                "split": split,
                "run_dir": str(run_dir),
                "output": str(split_path),
                "n": len(split_rows),
                "class_names": ";".join(class_names),
                "mixed_precision_enabled": bool(use_amp),
            }
        )
    return all_rows, manifest_rows


def _add_seed_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    seeds = sorted({str(row.get("seed", "")) for row in rows})
    for seed in seeds:
        seed_rows = [row for row in rows if str(row.get("seed", "")) == seed]
        for row in summarize_gate_rows(seed_rows):
            row = dict(row)
            row["seed"] = seed
            summary.append(row)
    return summary


def main() -> int:
    args = parse_args()
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = _discover_runs(args.run)

    all_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for label, run_dir in runs:
        run_rows, run_manifest = _dump_run(label, run_dir, args, out_dir)
        all_rows.extend(run_rows)
        manifest_rows.extend(run_manifest)

    _write_csv(out_dir / "gate_samples.csv", all_rows)
    _write_csv(out_dir / "gate_summary_by_seed.csv", _add_seed_summary(all_rows))
    _write_csv(out_dir / "gate_summary_all_runs.csv", summarize_gate_rows(all_rows))
    _write_csv(out_dir / "gate_manifest.csv", manifest_rows)
    print(f"Wrote {len(all_rows)} gate sample rows from {len(runs)} runs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
