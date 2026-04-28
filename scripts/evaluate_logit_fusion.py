#!/usr/bin/env python
"""Evaluate late logit fusion from trained single-modality checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
DEFAULT_ALPHAS = "0,0.05,0.10,0.20,0.30,0.50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse logits from trained ToT and ToA single-modality runs. "
            "The fusion weight is selected on validation data only, then reported on test."
        )
    )
    parser.add_argument("--group", default="a4_modality_comparison_seed42", help="Experiment group to scan")
    parser.add_argument("--root", default=str(DEFAULT_EXPERIMENT_ROOT), help="Experiment output root")
    parser.add_argument("--tot-run", default=None, help="Explicit ToT run directory or metadata.json")
    parser.add_argument("--toa-run", default=None, help="Explicit ToA run directory or metadata.json")
    parser.add_argument("--seed", type=int, default=None, help="Restrict automatic group matching to one training seed")
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS, help="Comma-separated ToA weights")
    parser.add_argument("--data-root", default=None, help="Override dataset.root for this machine")
    parser.add_argument("--output-csv", default=None, help="CSV output path")
    parser.add_argument("--output-json", default=None, help="JSON output path")
    parser.add_argument(
        "--mixed-precision",
        choices=["auto", "true", "false"],
        default="auto",
        help="Use CUDA autocast for inference. auto follows the source training configs.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _run_dir(path: str | Path) -> Path:
    candidate = resolve_project_path(path)
    if candidate.name == "metadata.json":
        candidate = candidate.parent
    if not candidate.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {candidate}")
    return candidate


def _metadata_path(run_dir: Path) -> Path:
    path = run_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {run_dir}")
    return path


def _load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml: {run_dir}")
    cfg = load_yaml(config_path)
    validate_experiment_config(cfg)
    return cfg


def _checkpoint_path(run_dir: Path) -> Path:
    path = run_dir / "best_model.pth"
    if not path.exists():
        raise FileNotFoundError(f"Missing best_model.pth: {run_dir}")
    return path


def _parse_alphas(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"alpha must be within [0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("--alphas must contain at least one value")
    return values


def _modalities(metadata: dict[str, Any]) -> tuple[str, ...]:
    dataset = metadata.get("dataset", {})
    return tuple(dataset.get("modalities") or metadata.get("data_info", {}).get("modalities") or [])


def _training_seed(metadata: dict[str, Any]) -> int | None:
    training = metadata.get("training", {})
    seed = training.get("seed")
    return int(seed) if seed is not None else None


def _discover_pairs(root: Path, group: str, seed: int | None) -> list[tuple[int, Path, Path]]:
    group_root = resolve_project_path(root) / group
    if not group_root.is_dir():
        raise FileNotFoundError(f"Experiment group does not exist: {group_root}")

    by_seed: dict[int, dict[str, Path]] = {}
    for metadata_path in sorted(group_root.glob("*/metadata.json")):
        metadata = _load_json(metadata_path)
        modalities = _modalities(metadata)
        run_seed = _training_seed(metadata)
        if run_seed is None:
            continue
        if seed is not None and run_seed != seed:
            continue
        key = None
        if modalities == ("ToT",):
            key = "tot"
        elif modalities == ("ToA",):
            key = "toa"
        if key is not None:
            by_seed.setdefault(run_seed, {})[key] = metadata_path.parent

    pairs = []
    for run_seed in sorted(by_seed):
        item = by_seed[run_seed]
        if "tot" in item and "toa" in item:
            pairs.append((run_seed, item["tot"], item["toa"]))
    if not pairs:
        raise RuntimeError(f"No ToT/ToA single-modality pairs found in {group_root}")
    return pairs


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


def _effective_amp(args: argparse.Namespace, tot_cfg: dict[str, Any], toa_cfg: dict[str, Any], device) -> tuple[bool, Any | None]:
    if args.mixed_precision == "false":
        return False, None
    if args.mixed_precision == "true":
        return device.type == "cuda", _dtype_from_config(tot_cfg)
    requested = bool(tot_cfg.get("training", {}).get("mixed_precision", False)) or bool(
        toa_cfg.get("training", {}).get("mixed_precision", False)
    )
    return requested and device.type == "cuda", _dtype_from_config(tot_cfg)


def _load_model_logits_for_splits(
    run_dir: Path,
    cfg: dict[str, Any],
    splits: list[str],
    data_root: str | None,
    device,
    autocast_factory,
) -> tuple[dict[str, dict[str, np.ndarray]], list[float], dict[str, Any]]:
    import torch

    from timepix.data import build_dataloaders
    from timepix.losses import build_loss
    from timepix.models import build_model
    from timepix.training.trainer import evaluate

    if cfg.get("task", {}).get("type", "classification") != "classification":
        raise ValueError("Late logit fusion currently supports classification runs only")

    loaders, data_info = build_dataloaders(cfg, data_root_override=data_root)
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
    payloads = {}
    for split in splits:
        payload = evaluate(
            model,
            loaders[split],
            criterion,
            device,
            "classification",
            progress_bar=False,
            autocast_factory=autocast_factory,
        )
        payloads[split] = {
            "logits": payload["logits"],
            "labels": payload["labels"].astype(int),
        }
    return payloads, angle_values, data_info


def _score(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics.get("accuracy", 0.0)),
        -float(metrics.get("mae_argmax", metrics.get("mae", 0.0))),
        float(metrics.get("macro_f1", 0.0)),
    )


def _evaluate_pair(
    seed: int,
    tot_run: Path,
    toa_run: Path,
    alphas: list[float],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    tot_cfg = _load_config(tot_run)
    toa_cfg = _load_config(toa_run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp, dtype = _effective_amp(args, tot_cfg, toa_cfg, device)
    autocast_factory = _autocast_factory(use_amp, dtype, device)

    tot_payloads, angle_values, tot_data = _load_model_logits_for_splits(
        tot_run, tot_cfg, ["val", "test"], args.data_root, device, autocast_factory
    )
    toa_payloads, angle_values_toa, toa_data = _load_model_logits_for_splits(
        toa_run, toa_cfg, ["val", "test"], args.data_root, device, autocast_factory
    )
    val_tot = tot_payloads["val"]["logits"]
    val_toa = toa_payloads["val"]["logits"]
    val_labels_tot = tot_payloads["val"]["labels"]
    val_labels_toa = toa_payloads["val"]["labels"]
    test_tot = tot_payloads["test"]["logits"]
    test_toa = toa_payloads["test"]["logits"]
    test_labels_tot = tot_payloads["test"]["labels"]
    test_labels_toa = toa_payloads["test"]["labels"]

    if angle_values != angle_values_toa:
        raise ValueError("ToT and ToA label maps do not match")
    if not np.array_equal(val_labels_tot, val_labels_toa):
        raise ValueError("Validation labels are not aligned between ToT and ToA runs")
    if not np.array_equal(test_labels_tot, test_labels_toa):
        raise ValueError("Test labels are not aligned between ToT and ToA runs")

    rows = []
    selected_row = None
    selected_score = None
    for alpha in alphas:
        val_logits = (1.0 - alpha) * val_tot + alpha * val_toa
        test_logits = (1.0 - alpha) * test_tot + alpha * test_toa
        val_metrics = classification_metrics(val_logits, val_labels_tot, angle_values)
        test_metrics = classification_metrics(test_logits, test_labels_tot, angle_values)
        row = {
            "seed": seed,
            "alpha_toa": alpha,
            "alpha_tot": 1.0 - alpha,
            "selected": False,
            "val_accuracy": val_metrics.get("accuracy"),
            "val_mae_argmax": val_metrics.get("mae_argmax"),
            "val_p90_error": val_metrics.get("p90_error"),
            "val_macro_f1": val_metrics.get("macro_f1"),
            "test_accuracy": test_metrics.get("accuracy"),
            "test_mae_argmax": test_metrics.get("mae_argmax"),
            "test_p90_error": test_metrics.get("p90_error"),
            "test_macro_f1": test_metrics.get("macro_f1"),
            "tot_run": str(tot_run),
            "toa_run": str(toa_run),
            "split_path": tot_data.get("split_path"),
            "device": str(device),
            "mixed_precision_enabled": use_amp,
        }
        score = _score(val_metrics)
        if selected_score is None or score > selected_score:
            selected_score = score
            selected_row = row
        rows.append(row)

    assert selected_row is not None
    selected_row["selected"] = True
    summary = {
        "seed": seed,
        "selected_alpha_toa": selected_row["alpha_toa"],
        "selected_alpha_tot": selected_row["alpha_tot"],
        "selection_rule": "max validation accuracy, then lower validation MAE, then higher validation macro-F1",
        "selected": selected_row,
        "tot_run": str(tot_run),
        "toa_run": str(toa_run),
        "tot_data_info": tot_data,
        "toa_data_info": toa_data,
        "device": str(device),
        "mixed_precision_enabled": use_amp,
    }
    return rows, summary


def _default_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    stem = f"a4b_late_logit_fusion_{args.group}" if args.group else "a4b_late_logit_fusion"
    csv_path = Path(args.output_csv) if args.output_csv else Path("outputs") / f"{stem}.csv"
    json_path = Path(args.output_json) if args.output_json else Path("outputs") / f"{stem}.json"
    return csv_path, json_path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "alpha_toa",
        "alpha_tot",
        "selected",
        "val_accuracy",
        "val_mae_argmax",
        "val_p90_error",
        "val_macro_f1",
        "test_accuracy",
        "test_mae_argmax",
        "test_p90_error",
        "test_macro_f1",
        "tot_run",
        "toa_run",
        "split_path",
        "device",
        "mixed_precision_enabled",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    alphas = _parse_alphas(args.alphas)
    if bool(args.tot_run) != bool(args.toa_run):
        raise ValueError("--tot-run and --toa-run must be provided together")

    if args.tot_run and args.toa_run:
        tot_run = _run_dir(args.tot_run)
        toa_run = _run_dir(args.toa_run)
        seed = args.seed
        if seed is None:
            metadata = _load_json(_metadata_path(tot_run))
            seed = _training_seed(metadata) or 0
        pairs = [(seed, tot_run, toa_run)]
    else:
        pairs = _discover_pairs(Path(args.root), args.group, args.seed)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for seed, tot_run, toa_run in pairs:
        rows, summary = _evaluate_pair(seed, tot_run, toa_run, alphas, args)
        all_rows.extend(rows)
        summaries.append(summary)
        selected = summary["selected"]
        print(
            "Selected late fusion | "
            f"seed={seed} alpha_toa={selected['alpha_toa']:.3f} "
            f"val_acc={selected['val_accuracy']:.4f} "
            f"test_acc={selected['test_accuracy']:.4f} "
            f"test_mae={selected['test_mae_argmax']:.3f} "
            f"test_macro_f1={selected['test_macro_f1']:.4f}"
        )

    csv_path, json_path = _default_output_paths(args)
    _write_csv(csv_path, all_rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "method": "late_logit_fusion",
                "formula": "logits = (1 - alpha_toa) * logits_tot + alpha_toa * logits_toa",
                "alphas": alphas,
                "selection_data": "validation",
                "pairs": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
