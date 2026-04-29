#!/usr/bin/env python
"""Evaluate rule-based or lightweight selectors over frozen logits for A4b-4.

The script reloads two trained checkpoints, recomputes deterministic logits on
train/val/test, selects all thresholds/rules on validation, and reports the
final test metrics. ResNet experts stay frozen.
"""

from __future__ import annotations

import argparse
import csv
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
from timepix.training.metrics import confusion_matrix, p90_error


DEFAULT_EXPERIMENT_ROOT = Path("outputs/experiments")
DEFAULT_THRESHOLDS = ",".join(f"{value / 100:.2f}" for value in range(5, 100, 5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A4b-4 selector fusion over frozen logits")
    parser.add_argument("--root", default=str(DEFAULT_EXPERIMENT_ROOT), help="Experiment output root")
    parser.add_argument("--tot-group", action="append", default=None, help="Group containing ToT runs")
    parser.add_argument("--candidate-group", action="append", default=None, help="Group containing candidate runs")
    parser.add_argument("--tot-run", default=None, help="Explicit ToT run directory or metadata.json")
    parser.add_argument("--candidate-run", default=None, help="Explicit candidate run directory or metadata.json")
    parser.add_argument("--seed", type=int, default=42, help="Training seed to match when scanning groups")
    parser.add_argument("--data-root", default=None, help="Override dataset.root for this machine")
    parser.add_argument("--num-workers", type=int, default=0, help="Override dataloader workers for inference")
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
    parser.add_argument(
        "--selector-mode",
        choices=["trained", "rule"],
        default="trained",
        help="Use a trained selector or validation-selected hand rules",
    )
    parser.add_argument(
        "--selector-fit",
        choices=["train", "val-cv"],
        default="train",
        help="For trained mode: fit selector on train logits or use validation cross-fitting",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of validation folds for --selector-fit val-cv")
    parser.add_argument(
        "--selector-target",
        choices=["lower-error", "candidate-correct-primary-wrong"],
        default="lower-error",
        help="Training target for the selector",
    )
    parser.add_argument("--selector-hidden-dim", type=int, default=0, help="Optional MLP hidden size; 0 means logistic")
    parser.add_argument("--selector-epochs", type=int, default=500, help="Selector training epochs")
    parser.add_argument("--selector-lr", type=float, default=0.01, help="Selector learning rate")
    parser.add_argument("--selector-weight-decay", type=float, default=0.0001, help="Selector weight decay")
    parser.add_argument("--selector-seed", type=int, default=42, help="Selector initialization seed")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="Comma-separated validation thresholds")
    parser.add_argument("--output-json", default=None, help="Output JSON path")
    parser.add_argument("--output-summary", default=None, help="Output summary CSV path")
    parser.add_argument("--output-by-class", default=None, help="Output per-class CSV path")
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


def _discover_group_runs(root: Path, groups: list[str], seed: int | None) -> list[Path]:
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
            run_seed = _training_seed(metadata)
            if seed is not None and run_seed != seed:
                continue
            seen.add(run_dir)
            runs.append(run_dir)
    return runs


def _discover_tot_run(root: Path, groups: list[str], seed: int | None) -> Path:
    matches = []
    for run_dir in _discover_group_runs(root, groups, seed):
        metadata = _load_json(_metadata_path(run_dir))
        if _modalities(metadata) == ("ToT",):
            matches.append(run_dir)
    if not matches:
        raise RuntimeError("No matching ToT run found")
    return matches[0]


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


def _discover_candidate_run(root: Path, groups: list[str], seed: int | None, transform: str, mask_filter: str) -> Path:
    matches = []
    for run_dir in _discover_group_runs(root, groups, seed):
        metadata = _load_json(_metadata_path(run_dir))
        if _candidate_matches(metadata, transform, mask_filter):
            matches.append(run_dir)
    if not matches:
        raise RuntimeError("No matching candidate run found")
    return matches[0]


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


def _load_run_payload(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
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
    for split in ["train", "val", "test"]:
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


def _validate_pair(primary: dict[str, Any], candidate: dict[str, Any], split: str) -> None:
    primary_split = primary["splits"][split]
    candidate_split = candidate["splits"][split]
    if primary["angle_values"] != candidate["angle_values"]:
        raise ValueError("Label maps/angle values do not match")
    if primary_split["keys"] != candidate_split["keys"]:
        raise ValueError(f"Sample keys are not aligned for split={split}")
    if not np.array_equal(primary_split["labels"], candidate_split["labels"]):
        raise ValueError(f"Labels are not aligned for split={split}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(probs + 1e-12)).sum(axis=1, keepdims=True)


def _margin(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    if sorted_probs.shape[1] == 1:
        return sorted_probs[:, :1]
    return (sorted_probs[:, :1] - sorted_probs[:, 1:2])


def _one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    result = np.zeros((indices.shape[0], num_classes), dtype=np.float32)
    result[np.arange(indices.shape[0]), indices.astype(int)] = 1.0
    return result


def _selector_features(primary_logits: np.ndarray, candidate_logits: np.ndarray, angle_values: list[float]) -> np.ndarray:
    primary_probs = _softmax(primary_logits)
    candidate_probs = _softmax(candidate_logits)
    primary_preds = primary_probs.argmax(axis=1)
    candidate_preds = candidate_probs.argmax(axis=1)
    angles = np.asarray(angle_values, dtype=np.float32)
    max_span = float(max(angles.max() - angles.min(), 1.0))
    pred_angle_diff = np.abs(angles[primary_preds] - angles[candidate_preds])[:, None] / max_span
    disagreement = (primary_preds != candidate_preds).astype(np.float32)[:, None]
    num_classes = primary_logits.shape[1]
    return np.concatenate(
        [
            primary_logits.astype(np.float32),
            candidate_logits.astype(np.float32),
            (candidate_logits - primary_logits).astype(np.float32),
            primary_probs.astype(np.float32),
            candidate_probs.astype(np.float32),
            (candidate_probs - primary_probs).astype(np.float32),
            primary_probs.max(axis=1, keepdims=True).astype(np.float32),
            candidate_probs.max(axis=1, keepdims=True).astype(np.float32),
            _margin(primary_probs).astype(np.float32),
            _margin(candidate_probs).astype(np.float32),
            _entropy(primary_probs).astype(np.float32),
            _entropy(candidate_probs).astype(np.float32),
            disagreement,
            pred_angle_diff.astype(np.float32),
            _one_hot(primary_preds, num_classes),
            _one_hot(candidate_preds, num_classes),
        ],
        axis=1,
    )


def _errors(logits: np.ndarray, labels: np.ndarray, angle_values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    probs = _softmax(logits)
    preds = probs.argmax(axis=1)
    angles = np.asarray(angle_values, dtype=float)
    return np.abs(angles[preds] - angles[labels.astype(int)]), preds


def _selector_target(
    primary_logits: np.ndarray,
    candidate_logits: np.ndarray,
    labels: np.ndarray,
    angle_values: list[float],
    mode: str,
) -> np.ndarray:
    primary_errors, primary_preds = _errors(primary_logits, labels, angle_values)
    candidate_errors, candidate_preds = _errors(candidate_logits, labels, angle_values)
    if mode == "candidate-correct-primary-wrong":
        return np.logical_and(candidate_preds == labels, primary_preds != labels).astype(np.float32)
    return (candidate_errors < primary_errors).astype(np.float32)


def _standardize(train_features: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (
        (train_features - mean) / std,
        [(item - mean) / std for item in others],
        {"mean": mean.squeeze(0).tolist(), "std": std.squeeze(0).tolist()},
    )


def _train_selector(
    features: np.ndarray,
    target: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch import nn

    torch.manual_seed(int(args.selector_seed))
    x = torch.from_numpy(features.astype(np.float32))
    y = torch.from_numpy(target.astype(np.float32)).view(-1, 1)
    input_dim = x.shape[1]
    if int(args.selector_hidden_dim) > 0:
        model = nn.Sequential(
            nn.Linear(input_dim, int(args.selector_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(args.selector_hidden_dim), 1),
        )
    else:
        model = nn.Linear(input_dim, 1)

    positive = float(y.sum().item())
    negative = float(y.numel() - positive)
    pos_weight_value = negative / max(positive, 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.selector_lr),
        weight_decay=float(args.selector_weight_decay),
    )
    model.train()
    final_loss = 0.0
    for _ in range(int(args.selector_epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    model.eval()
    info = {
        "input_dim": int(input_dim),
        "positive_count": int(positive),
        "negative_count": int(negative),
        "positive_rate": float(positive / max(y.numel(), 1)),
        "pos_weight": float(pos_weight_value),
        "final_loss": final_loss,
    }
    return model, info


def _predict_selector(model, features: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        logits = model(torch.from_numpy(features.astype(np.float32))).squeeze(1)
        return torch.sigmoid(logits).cpu().numpy()


def _make_stratified_folds(target: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    n_folds = max(2, min(int(n_folds), int(target.shape[0])))
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for value in [0.0, 1.0]:
        indices = np.where(target == value)[0]
        rng.shuffle(indices)
        for offset, index in enumerate(indices.tolist()):
            folds[offset % n_folds].append(index)
    return [np.asarray(sorted(fold), dtype=int) for fold in folds if fold]


def _fit_train_selector_probs(
    raw_features: dict[str, np.ndarray],
    raw_targets: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    train_features, [val_features, test_features], feature_scaler = _standardize(
        raw_features["train"],
        raw_features["val"],
        raw_features["test"],
    )
    selector, selector_info = _train_selector(train_features, raw_targets["train"], args)
    selector_info.update(
        {
            "mode": "trained",
            "fit": "train",
            "target": args.selector_target,
            "hidden_dim": int(args.selector_hidden_dim),
            "epochs": int(args.selector_epochs),
            "lr": float(args.selector_lr),
            "weight_decay": float(args.selector_weight_decay),
        }
    )
    return (
        {
            "train": _predict_selector(selector, train_features),
            "val": _predict_selector(selector, val_features),
            "test": _predict_selector(selector, test_features),
        },
        selector_info,
        feature_scaler,
    )


def _fit_val_cv_selector_probs(
    raw_features: dict[str, np.ndarray],
    raw_targets: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    val_features = raw_features["val"]
    val_target = raw_targets["val"]
    folds = _make_stratified_folds(val_target, int(args.cv_folds), int(args.selector_seed))
    oof_probs = np.zeros(val_target.shape[0], dtype=np.float32)
    fold_infos = []
    for fold_index, heldout_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(val_target.shape[0]), heldout_idx, assume_unique=True)
        fold_train, [fold_heldout], _ = _standardize(val_features[train_idx], val_features[heldout_idx])
        selector, fold_info = _train_selector(fold_train, val_target[train_idx], args)
        oof_probs[heldout_idx] = _predict_selector(selector, fold_heldout)
        fold_infos.append(
            {
                "fold": fold_index,
                "train_count": int(train_idx.shape[0]),
                "heldout_count": int(heldout_idx.shape[0]),
                "heldout_positive_rate": float(val_target[heldout_idx].mean()) if heldout_idx.size else 0.0,
                "final_loss": fold_info.get("final_loss"),
            }
        )

    full_val_features, [train_features, test_features], feature_scaler = _standardize(
        raw_features["val"],
        raw_features["train"],
        raw_features["test"],
    )
    selector, selector_info = _train_selector(full_val_features, val_target, args)
    selector_info.update(
        {
            "mode": "trained",
            "fit": "val-cv",
            "target": args.selector_target,
            "hidden_dim": int(args.selector_hidden_dim),
            "epochs": int(args.selector_epochs),
            "lr": float(args.selector_lr),
            "weight_decay": float(args.selector_weight_decay),
            "cv_folds": int(len(folds)),
            "cv_fold_info": fold_infos,
            "validation_probabilities": "out_of_fold",
            "final_fit": "full_validation",
        }
    )
    return (
        {
            "train": _predict_selector(selector, train_features),
            "val": oof_probs,
            "test": _predict_selector(selector, test_features),
        },
        selector_info,
        feature_scaler,
    )


def _rule_masks(
    primary_logits: np.ndarray,
    candidate_logits: np.ndarray,
    angle_values: list[float],
) -> dict[str, np.ndarray]:
    primary_probs = _softmax(primary_logits)
    candidate_probs = _softmax(candidate_logits)
    primary_preds = primary_probs.argmax(axis=1)
    candidate_preds = candidate_probs.argmax(axis=1)
    angles = np.asarray(angle_values, dtype=float)
    primary_conf = primary_probs.max(axis=1)
    candidate_conf = candidate_probs.max(axis=1)
    primary_margin = _margin(primary_probs).squeeze(1)
    candidate_margin = _margin(candidate_probs).squeeze(1)
    primary_entropy = _entropy(primary_probs).squeeze(1)
    candidate_entropy = _entropy(candidate_probs).squeeze(1)
    disagree = primary_preds != candidate_preds
    candidate_is_30 = np.isclose(angles[candidate_preds], 30.0)
    primary_low_grid = [0.40, 0.45, 0.50, 0.55, 0.60]
    advantage_grid = [0.00, 0.03, 0.05, 0.10, 0.15]
    masks: dict[str, np.ndarray] = {}
    for delta in advantage_grid:
        tag = str(delta).replace(".", "p")
        masks[f"rule_conf_adv_{tag}"] = disagree & (candidate_conf >= primary_conf + delta)
        masks[f"rule_margin_adv_{tag}"] = disagree & (candidate_margin >= primary_margin + delta)
        masks[f"rule_entropy_adv_{tag}"] = disagree & (candidate_entropy <= primary_entropy - delta)
        masks[f"rule_30_conf_adv_{tag}"] = candidate_is_30 & disagree & (candidate_conf >= primary_conf + delta)
        masks[f"rule_30_margin_adv_{tag}"] = candidate_is_30 & disagree & (candidate_margin >= primary_margin + delta)
    for threshold in primary_low_grid:
        tag = str(threshold).replace(".", "p")
        masks[f"rule_primary_low_{tag}_cand_higher"] = disagree & (primary_conf <= threshold) & (
            candidate_conf >= primary_conf
        )
        masks[f"rule_30_primary_low_{tag}_cand_higher"] = candidate_is_30 & disagree & (
            primary_conf <= threshold
        ) & (candidate_conf >= primary_conf)
    return {name: mask.astype(bool) for name, mask in masks.items()}


def _metrics_from_preds(preds: np.ndarray, labels: np.ndarray, angle_values: list[float]) -> dict[str, Any]:
    num_classes = len(angle_values)
    if labels.size == 0:
        return {"accuracy": 0.0, "mae_argmax": 0.0, "p90_error": 0.0, "macro_f1": 0.0, "per_class": []}
    angles = np.asarray(angle_values, dtype=float)
    true_angles = angles[labels.astype(int)]
    pred_angles = angles[preds.astype(int)]
    errors = np.abs(pred_angles - true_angles)
    cm = confusion_matrix(labels, preds, num_classes)
    per_class = []
    f1_values = []
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
        class_mask = labels == cls
        per_class.append(
            {
                "class_index": cls,
                "class_angle": float(angles[cls]),
                "accuracy": float(np.mean(preds[class_mask] == labels[class_mask])) if class_mask.any() else 0.0,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n": int(class_mask.sum()),
            }
        )
        f1_values.append(f1)
    return {
        "accuracy": float(np.mean(preds == labels)),
        "mae_argmax": float(errors.mean()),
        "p90_error": p90_error(errors),
        "macro_f1": float(np.mean(f1_values)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def _preds_from_logits(logits: np.ndarray) -> np.ndarray:
    return _softmax(logits).argmax(axis=1)


def _strategy_preds(
    strategy: str,
    primary_logits: np.ndarray,
    candidate_logits: np.ndarray,
    labels: np.ndarray,
    angle_values: list[float],
    selector_probs: np.ndarray | None = None,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    primary_preds = _preds_from_logits(primary_logits)
    candidate_preds = _preds_from_logits(candidate_logits)
    if strategy == "primary_only":
        select = np.zeros(labels.shape[0], dtype=bool)
        return primary_preds, select
    if strategy == "candidate_only":
        select = np.ones(labels.shape[0], dtype=bool)
        return candidate_preds, select
    if strategy == "oracle":
        primary_errors, _ = _errors(primary_logits, labels, angle_values)
        candidate_errors, _ = _errors(candidate_logits, labels, angle_values)
        select = candidate_errors < primary_errors
        return np.where(select, candidate_preds, primary_preds), select
    if strategy == "selector":
        if selector_probs is None or threshold is None:
            raise ValueError("selector strategy requires selector probabilities and a threshold")
        select = selector_probs >= threshold
        return np.where(select, candidate_preds, primary_preds), select
    if strategy.startswith("rule:"):
        if selector_probs is None:
            raise ValueError("rule strategy requires a precomputed selection mask")
        select = selector_probs.astype(bool)
        return np.where(select, candidate_preds, primary_preds), select
    raise ValueError(f"Unknown strategy: {strategy}")


def _score(metrics: dict[str, Any], selection_rate: float) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("accuracy", 0.0)),
        -float(metrics.get("mae_argmax", 0.0)),
        float(metrics.get("macro_f1", 0.0)),
        -float(selection_rate),
    )


def _evaluate_strategy(
    split: str,
    strategy: str,
    primary_logits: np.ndarray,
    candidate_logits: np.ndarray,
    labels: np.ndarray,
    angle_values: list[float],
    selector_probs: np.ndarray | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    preds, select = _strategy_preds(
        strategy,
        primary_logits,
        candidate_logits,
        labels,
        angle_values,
        selector_probs=selector_probs,
        threshold=threshold,
    )
    metrics = _metrics_from_preds(preds, labels, angle_values)
    metrics.update(
        {
            "split": split,
            "strategy": strategy,
            "threshold": threshold,
            "selection_rate": float(select.mean()) if select.size else 0.0,
            "selected_count": int(select.sum()),
            "n": int(labels.shape[0]),
        }
    )
    return metrics


def _parse_thresholds(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"threshold must be within [0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("--thresholds must contain at least one value")
    return sorted(set(values))


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


def _row_from_metrics(
    seed: int | None,
    strategy: str,
    threshold: float | None,
    selected: bool,
    metrics_by_split: dict[str, dict[str, Any]],
    selector_info: dict[str, Any],
    primary_run: Path,
    candidate_run: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": seed,
        "strategy": strategy,
        "threshold": threshold,
        "selected_by_val": selected,
        "selector_mode": selector_info.get("mode"),
        "selector_fit": selector_info.get("fit"),
        "selector_target": selector_info.get("target"),
        "selector_positive_rate": selector_info.get("positive_rate"),
        "selector_final_loss": selector_info.get("final_loss"),
        "primary": _run_label(primary_run),
        "candidate": _run_label(candidate_run),
        "primary_run": str(primary_run),
        "candidate_run": str(candidate_run),
    }
    for split, metrics in metrics_by_split.items():
        prefix = f"{split}_"
        row[prefix + "accuracy"] = metrics["accuracy"]
        row[prefix + "mae_argmax"] = metrics["mae_argmax"]
        row[prefix + "p90_error"] = metrics["p90_error"]
        row[prefix + "macro_f1"] = metrics["macro_f1"]
        row[prefix + "selection_rate"] = metrics["selection_rate"]
        row[prefix + "selected_count"] = metrics["selected_count"]
    return row


def _by_class_rows(
    seed: int | None,
    selected_strategy_key: str,
    strategies: list[tuple[str, float | None]],
    metrics_by_strategy: dict[tuple[str, float | None], dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for strategy, threshold in strategies:
        key = (strategy, threshold)
        strategy_key = f"{strategy}:{threshold}" if threshold is not None else strategy
        include = strategy in {"primary_only", "candidate_only", "oracle"} or strategy_key == selected_strategy_key
        if not include:
            continue
        for split, metrics in metrics_by_strategy[key].items():
            for item in metrics["per_class"]:
                rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "threshold": threshold,
                        "selected_strategy": strategy_key == selected_strategy_key,
                        "split": split,
                        **item,
                    }
                )
    return rows


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
    suffix = args.selector_mode if args.selector_mode == "rule" else f"trained_{args.selector_fit}"
    stem = f"a4b_4_selector_fusion_{suffix}_seed{args.seed}"
    json_path = Path(args.output_json) if args.output_json else Path("outputs") / f"{stem}.json"
    summary_path = Path(args.output_summary) if args.output_summary else Path("outputs") / f"{stem}_summary.csv"
    by_class_path = Path(args.output_by_class) if args.output_by_class else Path("outputs") / f"{stem}_by_class.csv"
    return json_path, summary_path, by_class_path


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if bool(args.tot_run) != bool(args.candidate_run):
        raise ValueError("--tot-run and --candidate-run must be provided together")

    if args.tot_run and args.candidate_run:
        primary_run = _run_dir(args.tot_run)
        candidate_run = _run_dir(args.candidate_run)
    else:
        tot_groups = args.tot_group or ["a2_best_3seed"]
        candidate_groups = args.candidate_group or ["a4b_toa_transform_seed42"]
        primary_run = _discover_tot_run(root, tot_groups, args.seed)
        candidate_run = _discover_candidate_run(
            root,
            candidate_groups,
            args.seed,
            args.candidate_toa_transform,
            args.candidate_add_hit_mask,
        )

    primary = _load_run_payload(primary_run, args)
    candidate = _load_run_payload(candidate_run, args)
    for split in ["train", "val", "test"]:
        _validate_pair(primary, candidate, split)

    angle_values = primary["angle_values"]
    split_data: dict[str, dict[str, Any]] = {}
    for split in ["train", "val", "test"]:
        split_data[split] = {
            "primary_logits": primary["splits"][split]["logits"],
            "candidate_logits": candidate["splits"][split]["logits"],
            "labels": primary["splits"][split]["labels"],
        }

    raw_features = {
        split: _selector_features(
            split_data[split]["primary_logits"],
            split_data[split]["candidate_logits"],
            angle_values,
        )
        for split in ["train", "val", "test"]
    }
    raw_targets = {
        split: _selector_target(
            split_data[split]["primary_logits"],
            split_data[split]["candidate_logits"],
            split_data[split]["labels"],
            angle_values,
            args.selector_target,
        )
        for split in ["train", "val", "test"]
    }

    selector_probs: dict[str, np.ndarray] = {}
    rule_masks_by_split: dict[str, dict[str, np.ndarray]] = {}
    feature_scaler: dict[str, Any] = {}
    if args.selector_mode == "trained":
        if args.selector_fit == "train":
            selector_probs, selector_info, feature_scaler = _fit_train_selector_probs(raw_features, raw_targets, args)
        else:
            selector_probs, selector_info, feature_scaler = _fit_val_cv_selector_probs(raw_features, raw_targets, args)
        thresholds = _parse_thresholds(args.thresholds)
        strategies: list[tuple[str, float | None]] = [("primary_only", None), ("candidate_only", None)]
        strategies.extend(("selector", threshold) for threshold in thresholds)
    else:
        selector_info = {
            "mode": "rule",
            "fit": "validation_rule_selection",
            "target": "none",
            "positive_rate": None,
            "final_loss": None,
        }
        for split in ["train", "val", "test"]:
            rule_masks_by_split[split] = _rule_masks(
                split_data[split]["primary_logits"],
                split_data[split]["candidate_logits"],
                angle_values,
            )
        thresholds = []
        rule_names = sorted(rule_masks_by_split["val"])
        strategies = [("primary_only", None), ("candidate_only", None)]
        strategies.extend((f"rule:{name}", None) for name in rule_names)
    strategies.append(("oracle", None))

    metrics_by_strategy: dict[tuple[str, float | None], dict[str, dict[str, Any]]] = {}
    for strategy, threshold in strategies:
        key = (strategy, threshold)
        metrics_by_strategy[key] = {}
        for split in ["train", "val", "test"]:
            strategy_probs = None
            if strategy == "selector":
                strategy_probs = selector_probs[split]
            elif strategy.startswith("rule:"):
                rule_name = strategy.split(":", 1)[1]
                strategy_probs = rule_masks_by_split[split][rule_name].astype(np.float32)
            metrics_by_strategy[key][split] = _evaluate_strategy(
                split,
                strategy,
                split_data[split]["primary_logits"],
                split_data[split]["candidate_logits"],
                split_data[split]["labels"],
                angle_values,
                selector_probs=strategy_probs,
                threshold=threshold,
            )

    selectable = [("primary_only", None), ("candidate_only", None)]
    if args.selector_mode == "trained":
        selectable.extend(("selector", threshold) for threshold in thresholds)
    else:
        selectable.extend((strategy, threshold) for strategy, threshold in strategies if strategy.startswith("rule:"))
    selected_key = max(
        selectable,
        key=lambda item: _score(
            metrics_by_strategy[item]["val"],
            metrics_by_strategy[item]["val"]["selection_rate"],
        ),
    )
    selected_strategy_key = f"{selected_key[0]}:{selected_key[1]}" if selected_key[1] is not None else selected_key[0]

    summary_rows = []
    for strategy, threshold in strategies:
        if strategy == "oracle":
            selected = False
        else:
            selected = (strategy, threshold) == selected_key
        summary_rows.append(
            _row_from_metrics(
                args.seed,
                strategy,
                threshold,
                selected,
                metrics_by_strategy[(strategy, threshold)],
                selector_info,
                primary_run,
                candidate_run,
            )
        )

    by_class_rows = _by_class_rows(args.seed, selected_strategy_key, strategies, metrics_by_strategy)
    json_path, summary_path, by_class_path = _default_outputs(args)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "analysis": "a4b_4_selector_fusion",
        "seed": args.seed,
        "primary_run": str(primary_run),
        "candidate_run": str(candidate_run),
        "primary": _run_label(primary_run),
        "candidate": _run_label(candidate_run),
        "selector_mode": args.selector_mode,
        "selector_fit": args.selector_fit if args.selector_mode == "trained" else "validation_rule_selection",
        "selector": selector_info,
        "feature_scaler": feature_scaler,
        "thresholds": thresholds,
        "rules": sorted(rule_masks_by_split["val"]) if args.selector_mode == "rule" else [],
        "selected_strategy": selected_strategy_key,
        "selection_rule": "max validation accuracy, then lower validation MAE, then higher macro-F1, then lower candidate selection rate",
        "summary": summary_rows,
        "data_info": {
            "primary": primary["data_info"],
            "candidate": candidate["data_info"],
        },
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary_path, summary_rows)
    _write_csv(by_class_path, by_class_rows)

    selected_metrics = metrics_by_strategy[selected_key]
    print(
        "Selected selector fusion | "
        f"strategy={selected_strategy_key} "
        f"val_acc={selected_metrics['val']['accuracy']:.4f} "
        f"test_acc={selected_metrics['test']['accuracy']:.4f} "
        f"test_mae={selected_metrics['test']['mae_argmax']:.3f} "
        f"test_macro_f1={selected_metrics['test']['macro_f1']:.4f} "
        f"test_selection_rate={selected_metrics['test']['selection_rate']:.4f}"
    )
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote per-class CSV: {by_class_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
