#!/usr/bin/env python
"""Evaluate A4b-5 sample-wise gated late fusion over frozen logits.

The script reloads two frozen experts, recomputes deterministic train/val/test
logits, then compares entropy-soft and learned sample-wise gates. ResNet experts
are not retrained. Validation selects the A4b-5 variant; test is only reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_selector_fusion import (  # noqa: E402
    DEFAULT_EXPERIMENT_ROOT,
    _discover_candidate_run,
    _discover_tot_run,
    _errors,
    _load_json,
    _load_run_payload,
    _margin,
    _metadata_path,
    _metrics_from_preds,
    _modalities,
    _preds_from_logits,
    _run_dir,
    _run_label,
    _rule_masks,
    _selector_features,
    _softmax,
    _standardize,
    _strategy_preds,
    _toa_transform,
    _training_seed,
    _validate_pair,
    _write_csv,
)


DEFAULT_THRESHOLDS = "-0.10,-0.05,0.00,0.03,0.05,0.10,0.15,0.20"
DEFAULT_SLOPES = "5,10,20,40"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A4b-5 gated late fusion over frozen logits")
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
        "--fit-modes",
        default="train,val-cv",
        help="Comma-separated learned-gate fit modes: train,val-cv",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Validation folds for val-cv learned gates")
    parser.add_argument("--gate-hidden-dim", type=int, default=0, help="Optional gate MLP hidden size; 0 means linear")
    parser.add_argument("--gate-epochs", type=int, default=500, help="Gate training epochs")
    parser.add_argument("--gate-lr", type=float, default=0.01, help="Gate learning rate")
    parser.add_argument("--gate-weight-decay", type=float, default=0.0001, help="Gate weight decay")
    parser.add_argument("--gate-seed", type=int, default=42, help="Gate initialization seed")
    parser.add_argument("--conservative-init-bias", type=float, default=-2.0, help="Initial bias for conservative gate")
    parser.add_argument("--conservative-gate-l1", type=float, default=0.01, help="Mean-gate penalty for conservative gate")
    parser.add_argument("--entropy-thresholds", default=DEFAULT_THRESHOLDS, help="Comma-separated entropy thresholds")
    parser.add_argument("--entropy-slopes", default=DEFAULT_SLOPES, help="Comma-separated entropy sigmoid slopes")
    parser.add_argument("--output-json", default=None, help="Output JSON path")
    parser.add_argument("--output-summary", default=None, help="Output summary CSV path")
    parser.add_argument("--output-by-class", default=None, help="Output per-class CSV path")
    return parser.parse_args()


def _parse_csv_floats(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("Expected at least one numeric value")
    return sorted(set(values))


def _parse_fit_modes(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"train", "val-cv"}
    bad = sorted(set(values) - allowed)
    if bad:
        raise ValueError(f"Unsupported fit modes: {bad}")
    return values or ["train", "val-cv"]


def _gate_features(primary_logits: np.ndarray, candidate_logits: np.ndarray, angle_values: list[float]) -> np.ndarray:
    base = _selector_features(primary_logits, candidate_logits, angle_values)
    primary_probs = _softmax(primary_logits)
    candidate_probs = _softmax(candidate_logits)
    primary_preds = primary_probs.argmax(axis=1)
    candidate_preds = candidate_probs.argmax(axis=1)
    angles = np.asarray(angle_values, dtype=float)
    primary_is_30 = np.isclose(angles[primary_preds], 30.0).astype(np.float32)[:, None]
    candidate_is_30 = np.isclose(angles[candidate_preds], 30.0).astype(np.float32)[:, None]
    return np.concatenate([base, primary_is_30, candidate_is_30], axis=1).astype(np.float32)


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


def _candidate_better_target(
    primary_logits: np.ndarray,
    candidate_logits: np.ndarray,
    labels: np.ndarray,
    angle_values: list[float],
) -> np.ndarray:
    primary_errors, _ = _errors(primary_logits, labels, angle_values)
    candidate_errors, _ = _errors(candidate_logits, labels, angle_values)
    return (candidate_errors < primary_errors).astype(np.float32)


class GateSpec:
    def __init__(
        self,
        *,
        name: str,
        fusion_space: str,
        gate_shape: str,
        fit_mode: str,
        init_bias: float = 0.0,
        gate_l1: float = 0.0,
    ) -> None:
        self.name = name
        self.fusion_space = fusion_space
        self.gate_shape = gate_shape
        self.fit_mode = fit_mode
        self.init_bias = float(init_bias)
        self.gate_l1 = float(gate_l1)

    @property
    def output_dim(self) -> int | None:
        return None if self.gate_shape == "class" else 1


def _torch_train_gate(
    features: np.ndarray,
    primary_logits: np.ndarray,
    candidate_logits: np.ndarray,
    labels: np.ndarray,
    spec: GateSpec,
    args: argparse.Namespace,
    num_classes: int,
):
    import torch
    import torch.nn.functional as F
    from torch import nn

    torch.manual_seed(int(args.gate_seed))
    output_dim = num_classes if spec.gate_shape == "class" else 1
    if int(args.gate_hidden_dim) > 0:
        model = nn.Sequential(
            nn.Linear(features.shape[1], int(args.gate_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(args.gate_hidden_dim), output_dim),
        )
        final_layer = model[-1]
    else:
        model = nn.Linear(features.shape[1], output_dim)
        final_layer = model
    if hasattr(final_layer, "bias") and final_layer.bias is not None:
        nn.init.constant_(final_layer.bias, spec.init_bias)

    x = torch.from_numpy(features.astype(np.float32))
    p_logits = torch.from_numpy(primary_logits.astype(np.float32))
    c_logits = torch.from_numpy(candidate_logits.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.int64))
    p_probs = F.softmax(p_logits, dim=1)
    c_probs = F.softmax(c_logits, dim=1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.gate_lr),
        weight_decay=float(args.gate_weight_decay),
    )
    final_loss = 0.0
    model.train()
    for _ in range(int(args.gate_epochs)):
        optimizer.zero_grad(set_to_none=True)
        gate = torch.sigmoid(model(x))
        if spec.fusion_space == "prob":
            mixed = (1.0 - gate) * p_probs + gate * c_probs
            mixed = mixed / mixed.sum(dim=1, keepdim=True).clamp_min(1e-12)
            loss = F.nll_loss(torch.log(mixed.clamp_min(1e-12)), y)
        elif spec.fusion_space == "logit":
            mixed = (1.0 - gate) * p_logits + gate * c_logits
            loss = F.cross_entropy(mixed, y)
        else:
            raise ValueError(f"Unknown fusion space: {spec.fusion_space}")
        if spec.gate_l1 > 0:
            loss = loss + float(spec.gate_l1) * gate.mean()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    model.eval()
    info = {
        "input_dim": int(features.shape[1]),
        "output_dim": int(output_dim),
        "fusion_space": spec.fusion_space,
        "gate_shape": spec.gate_shape,
        "fit_mode": spec.fit_mode,
        "init_bias": spec.init_bias,
        "gate_l1": spec.gate_l1,
        "hidden_dim": int(args.gate_hidden_dim),
        "epochs": int(args.gate_epochs),
        "lr": float(args.gate_lr),
        "weight_decay": float(args.gate_weight_decay),
        "final_loss": final_loss,
    }
    return model, info


def _predict_gate(model, features: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        values = torch.sigmoid(model(torch.from_numpy(features.astype(np.float32)))).cpu().numpy()
    return values.astype(np.float32)


def _fit_train_gate(
    raw_features: dict[str, np.ndarray],
    split_data: dict[str, dict[str, Any]],
    angle_values: list[float],
    spec: GateSpec,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train_features, [val_features, test_features], _ = _standardize(
        raw_features["train"],
        raw_features["val"],
        raw_features["test"],
    )
    model, info = _torch_train_gate(
        train_features,
        split_data["train"]["primary_logits"],
        split_data["train"]["candidate_logits"],
        split_data["train"]["labels"],
        spec,
        args,
        len(angle_values),
    )
    return (
        {
            "train": _predict_gate(model, train_features),
            "val": _predict_gate(model, val_features),
            "test": _predict_gate(model, test_features),
        },
        info,
    )


def _fit_val_cv_gate(
    raw_features: dict[str, np.ndarray],
    split_data: dict[str, dict[str, Any]],
    angle_values: list[float],
    spec: GateSpec,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    val_features = raw_features["val"]
    val_target = _candidate_better_target(
        split_data["val"]["primary_logits"],
        split_data["val"]["candidate_logits"],
        split_data["val"]["labels"],
        angle_values,
    )
    folds = _make_stratified_folds(val_target, int(args.cv_folds), int(args.gate_seed))
    output_dim = len(angle_values) if spec.gate_shape == "class" else 1
    oof_gates = np.zeros((val_target.shape[0], output_dim), dtype=np.float32)
    fold_infos = []
    for fold_index, heldout_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(val_target.shape[0]), heldout_idx, assume_unique=True)
        fold_train, [fold_heldout], _ = _standardize(val_features[train_idx], val_features[heldout_idx])
        fold_split_data = {
            "primary_logits": split_data["val"]["primary_logits"][train_idx],
            "candidate_logits": split_data["val"]["candidate_logits"][train_idx],
            "labels": split_data["val"]["labels"][train_idx],
        }
        model, fold_info = _torch_train_gate(
            fold_train,
            fold_split_data["primary_logits"],
            fold_split_data["candidate_logits"],
            fold_split_data["labels"],
            spec,
            args,
            len(angle_values),
        )
        oof_gates[heldout_idx] = _predict_gate(model, fold_heldout)
        fold_infos.append(
            {
                "fold": fold_index,
                "train_count": int(train_idx.shape[0]),
                "heldout_count": int(heldout_idx.shape[0]),
                "heldout_positive_rate": float(val_target[heldout_idx].mean()) if heldout_idx.size else 0.0,
                "final_loss": fold_info.get("final_loss"),
            }
        )

    full_val_features, [train_features, test_features], _ = _standardize(
        raw_features["val"],
        raw_features["train"],
        raw_features["test"],
    )
    model, info = _torch_train_gate(
        full_val_features,
        split_data["val"]["primary_logits"],
        split_data["val"]["candidate_logits"],
        split_data["val"]["labels"],
        spec,
        args,
        len(angle_values),
    )
    info["cv_folds"] = int(len(folds))
    info["cv_fold_info"] = fold_infos
    info["validation_gates"] = "out_of_fold"
    info["final_fit"] = "full_validation"
    return (
        {
            "train": _predict_gate(model, train_features),
            "val": oof_gates,
            "test": _predict_gate(model, test_features),
        },
        info,
    )


def _entropy_soft_gates(
    primary_logits: np.ndarray,
    candidate_logits: np.ndarray,
    threshold: float,
    slope: float,
) -> np.ndarray:
    primary_entropy = -(_softmax(primary_logits) * np.log(_softmax(primary_logits) + 1e-12)).sum(axis=1, keepdims=True)
    candidate_entropy = -(_softmax(candidate_logits) * np.log(_softmax(candidate_logits) + 1e-12)).sum(axis=1, keepdims=True)
    entropy_adv = primary_entropy - candidate_entropy
    gate = 1.0 / (1.0 + np.exp(-float(slope) * (entropy_adv - float(threshold))))
    return gate.astype(np.float32)


def _fused_preds_and_scores(
    primary_logits: np.ndarray,
    candidate_logits: np.ndarray,
    gates: np.ndarray,
    fusion_space: str,
) -> tuple[np.ndarray, np.ndarray]:
    if gates.ndim == 1:
        gates = gates[:, None]
    if fusion_space == "prob":
        primary_probs = _softmax(primary_logits)
        candidate_probs = _softmax(candidate_logits)
        scores = (1.0 - gates) * primary_probs + gates * candidate_probs
        scores = scores / np.clip(scores.sum(axis=1, keepdims=True), 1e-12, None)
    elif fusion_space == "logit":
        scores = (1.0 - gates) * primary_logits + gates * candidate_logits
    else:
        raise ValueError(f"Unknown fusion space: {fusion_space}")
    return scores.argmax(axis=1), scores


def _effective_gate(gates: np.ndarray, preds: np.ndarray | None = None) -> np.ndarray:
    if gates.ndim == 1 or gates.shape[1] == 1:
        return gates.reshape(-1)
    if preds is None:
        return gates.mean(axis=1)
    return gates[np.arange(gates.shape[0]), preds.astype(int)]


def _gate_stats(
    gates: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    primary_logits: np.ndarray,
    candidate_logits: np.ndarray,
    angle_values: list[float],
) -> dict[str, Any]:
    effective = _effective_gate(gates, preds)
    primary_errors, _ = _errors(primary_logits, labels, angle_values)
    candidate_errors, _ = _errors(candidate_logits, labels, angle_values)
    high = effective >= 0.5
    beneficial = candidate_errors < primary_errors
    harmful = candidate_errors > primary_errors
    angles = np.asarray(angle_values, dtype=float)
    true_30 = np.isclose(angles[labels.astype(int)], 30.0)
    return {
        "gate_mean": float(effective.mean()) if effective.size else 0.0,
        "gate_std": float(effective.std()) if effective.size else 0.0,
        "gate_p90": float(np.quantile(effective, 0.90)) if effective.size else 0.0,
        "gate_high_rate": float(high.mean()) if high.size else 0.0,
        "gate_mean_true_30": float(effective[true_30].mean()) if true_30.any() else 0.0,
        "beneficial_high_gate_count": int(np.logical_and(high, beneficial).sum()),
        "harmful_high_gate_count": int(np.logical_and(high, harmful).sum()),
        "neutral_high_gate_count": int(np.logical_and.reduce((high, ~beneficial, ~harmful)).sum()),
    }


def _evaluate_fusion(
    split: str,
    strategy: str,
    primary_logits: np.ndarray,
    candidate_logits: np.ndarray,
    labels: np.ndarray,
    angle_values: list[float],
    *,
    gates: np.ndarray | None = None,
    fusion_space: str = "prob",
) -> dict[str, Any]:
    if strategy in {"primary_only", "candidate_only", "oracle"} or strategy.startswith("rule:"):
        rule_masks = None
        selector_probs = None
        if strategy.startswith("rule:"):
            rule_name = strategy.split(":", 1)[1]
            rule_masks = _rule_masks(primary_logits, candidate_logits, angle_values)
            selector_probs = rule_masks[rule_name].astype(np.float32)
        preds, select = _strategy_preds(
            strategy,
            primary_logits,
            candidate_logits,
            labels,
            angle_values,
            selector_probs=selector_probs,
            threshold=None,
        )
        metrics = _metrics_from_preds(preds, labels, angle_values)
        gate_stats = {
            "gate_mean": float(select.mean()) if select.size else 0.0,
            "gate_std": 0.0,
            "gate_p90": 1.0 if select.any() else 0.0,
            "gate_high_rate": float(select.mean()) if select.size else 0.0,
            "gate_mean_true_30": 0.0,
            "beneficial_high_gate_count": 0,
            "harmful_high_gate_count": 0,
            "neutral_high_gate_count": 0,
        }
    else:
        if gates is None:
            raise ValueError("A gated strategy requires gates")
        preds, _ = _fused_preds_and_scores(primary_logits, candidate_logits, gates, fusion_space)
        metrics = _metrics_from_preds(preds, labels, angle_values)
        gate_stats = _gate_stats(gates, preds, labels, primary_logits, candidate_logits, angle_values)

    metrics.update(
        {
            "split": split,
            "strategy": strategy,
            "fusion_space": fusion_space,
            "n": int(labels.shape[0]),
            **gate_stats,
        }
    )
    return metrics


def _score(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("accuracy", 0.0)),
        -float(metrics.get("mae_argmax", 0.0)),
        float(metrics.get("macro_f1", 0.0)),
        -float(metrics.get("gate_mean", 0.0)),
    )


def _select_runs(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.root)
    if bool(args.tot_run) != bool(args.candidate_run):
        raise ValueError("--tot-run and --candidate-run must be provided together")
    if args.tot_run and args.candidate_run:
        return _run_dir(args.tot_run), _run_dir(args.candidate_run)
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
    return primary_run, candidate_run


def _row_from_metrics(
    seed: int,
    strategy: str,
    selected: bool,
    metrics_by_split: dict[str, dict[str, Any]],
    variant_info: dict[str, Any],
    primary_run: Path,
    candidate_run: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": seed,
        "strategy": strategy,
        "selected_by_val": selected,
        "variant_family": variant_info.get("variant_family"),
        "variant_id": variant_info.get("variant_id"),
        "fit_mode": variant_info.get("fit_mode"),
        "fusion_space": variant_info.get("fusion_space"),
        "gate_shape": variant_info.get("gate_shape"),
        "entropy_threshold": variant_info.get("entropy_threshold"),
        "entropy_slope": variant_info.get("entropy_slope"),
        "gate_l1": variant_info.get("gate_l1"),
        "init_bias": variant_info.get("init_bias"),
        "gate_final_loss": variant_info.get("final_loss"),
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
        row[prefix + "gate_mean"] = metrics["gate_mean"]
        row[prefix + "gate_high_rate"] = metrics["gate_high_rate"]
        row[prefix + "gate_mean_true_30"] = metrics["gate_mean_true_30"]
        row[prefix + "beneficial_high_gate_count"] = metrics["beneficial_high_gate_count"]
        row[prefix + "harmful_high_gate_count"] = metrics["harmful_high_gate_count"]
    return row


def _by_class_rows(
    seed: int,
    strategies: list[str],
    selected_strategy: str,
    metrics_by_strategy: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for strategy in strategies:
        include = (
            strategy in {"primary_only", "candidate_only", "rule:rule_entropy_adv_0p03", "oracle"}
            or strategy == selected_strategy
        )
        if not include:
            continue
        for split, metrics in metrics_by_strategy[strategy].items():
            for item in metrics["per_class"]:
                rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "selected_strategy": strategy == selected_strategy,
                        "split": split,
                        **item,
                    }
                )
    return rows


def _default_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    stem = f"a4b_5_gated_late_fusion_seed{args.seed}"
    json_path = Path(args.output_json) if args.output_json else Path("outputs") / f"{stem}.json"
    summary_path = Path(args.output_summary) if args.output_summary else Path("outputs") / f"{stem}_summary.csv"
    by_class_path = Path(args.output_by_class) if args.output_by_class else Path("outputs") / f"{stem}_by_class.csv"
    return json_path, summary_path, by_class_path


def main() -> int:
    args = parse_args()
    primary_run, candidate_run = _select_runs(args)
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
        split: _gate_features(
            split_data[split]["primary_logits"],
            split_data[split]["candidate_logits"],
            angle_values,
        )
        for split in ["train", "val", "test"]
    }

    strategies: list[str] = []
    variant_info: dict[str, dict[str, Any]] = {}
    gates_by_strategy: dict[str, dict[str, np.ndarray]] = {}
    fusion_space_by_strategy: dict[str, str] = {}

    for strategy, family in [
        ("primary_only", "baseline"),
        ("candidate_only", "baseline"),
        ("rule:rule_entropy_adv_0p03", "a4b4_rule"),
        ("oracle", "oracle"),
    ]:
        strategies.append(strategy)
        variant_info[strategy] = {"variant_family": family, "variant_id": strategy, "fusion_space": "argmax"}
        fusion_space_by_strategy[strategy] = "prob"

    thresholds = _parse_csv_floats(args.entropy_thresholds)
    slopes = _parse_csv_floats(args.entropy_slopes)
    for threshold in thresholds:
        for slope in slopes:
            tag_threshold = str(threshold).replace("-", "m").replace(".", "p")
            tag_slope = str(slope).replace(".", "p")
            strategy = f"a4b5a_entropy_soft_prob_t{tag_threshold}_k{tag_slope}"
            strategies.append(strategy)
            variant_info[strategy] = {
                "variant_family": "a4b5a_entropy_soft_gate",
                "variant_id": "A4b-5a",
                "fit_mode": "validation_grid",
                "fusion_space": "prob",
                "gate_shape": "scalar",
                "entropy_threshold": threshold,
                "entropy_slope": slope,
            }
            fusion_space_by_strategy[strategy] = "prob"
            gates_by_strategy[strategy] = {
                split: _entropy_soft_gates(
                    split_data[split]["primary_logits"],
                    split_data[split]["candidate_logits"],
                    threshold,
                    slope,
                )
                for split in ["train", "val", "test"]
            }

    fit_modes = _parse_fit_modes(args.fit_modes)
    learned_specs: list[GateSpec] = []
    for fit_mode in fit_modes:
        learned_specs.extend(
            [
                GateSpec(name=f"a4b5b_learned_scalar_prob_{fit_mode}", fusion_space="prob", gate_shape="scalar", fit_mode=fit_mode),
                GateSpec(name=f"a4b5c_learned_scalar_logit_{fit_mode}", fusion_space="logit", gate_shape="scalar", fit_mode=fit_mode),
                GateSpec(name=f"a4b5d_class_aware_prob_{fit_mode}", fusion_space="prob", gate_shape="class", fit_mode=fit_mode),
                GateSpec(
                    name=f"a4b5e_conservative_scalar_prob_{fit_mode}",
                    fusion_space="prob",
                    gate_shape="scalar",
                    fit_mode=fit_mode,
                    init_bias=float(args.conservative_init_bias),
                    gate_l1=float(args.conservative_gate_l1),
                ),
            ]
        )

    for spec in learned_specs:
        if spec.fit_mode == "train":
            gate_predictions, info = _fit_train_gate(raw_features, split_data, angle_values, spec, args)
        elif spec.fit_mode == "val-cv":
            gate_predictions, info = _fit_val_cv_gate(raw_features, split_data, angle_values, spec, args)
        else:
            raise ValueError(f"Unknown fit mode: {spec.fit_mode}")
        strategies.append(spec.name)
        gates_by_strategy[spec.name] = gate_predictions
        fusion_space_by_strategy[spec.name] = spec.fusion_space
        family = spec.name.split("_", 1)[0]
        variant_id = {
            "a4b5b": "A4b-5b",
            "a4b5c": "A4b-5c",
            "a4b5d": "A4b-5d",
            "a4b5e": "A4b-5e",
        }.get(family, family)
        variant_info[spec.name] = {
            "variant_family": spec.name.rsplit("_", 1)[0],
            "variant_id": variant_id,
            **info,
        }

    metrics_by_strategy: dict[str, dict[str, dict[str, Any]]] = {}
    for strategy in strategies:
        metrics_by_strategy[strategy] = {}
        for split in ["train", "val", "test"]:
            metrics_by_strategy[strategy][split] = _evaluate_fusion(
                split,
                strategy,
                split_data[split]["primary_logits"],
                split_data[split]["candidate_logits"],
                split_data[split]["labels"],
                angle_values,
                gates=gates_by_strategy.get(strategy, {}).get(split),
                fusion_space=fusion_space_by_strategy.get(strategy, "prob"),
            )

    selectable = [strategy for strategy in strategies if strategy.startswith("a4b5")]
    selected_strategy = max(selectable, key=lambda item: _score(metrics_by_strategy[item]["val"]))

    summary_rows = [
        _row_from_metrics(
            args.seed,
            strategy,
            strategy == selected_strategy,
            metrics_by_strategy[strategy],
            variant_info[strategy],
            primary_run,
            candidate_run,
        )
        for strategy in strategies
    ]
    by_class_rows = _by_class_rows(args.seed, strategies, selected_strategy, metrics_by_strategy)
    json_path, summary_path, by_class_path = _default_outputs(args)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "analysis": "a4b_5_gated_late_fusion",
        "seed": args.seed,
        "primary_run": str(primary_run),
        "candidate_run": str(candidate_run),
        "primary": _run_label(primary_run),
        "candidate": _run_label(candidate_run),
        "selected_strategy": selected_strategy,
        "selection_rule": "max validation accuracy, then lower validation MAE, then higher macro-F1, then lower mean gate",
        "variants": variant_info,
        "summary": summary_rows,
        "data_info": {
            "primary": primary["data_info"],
            "candidate": candidate["data_info"],
        },
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary_path, summary_rows)
    _write_csv(by_class_path, by_class_rows)

    selected = metrics_by_strategy[selected_strategy]
    print(
        "Selected A4b-5 gated fusion | "
        f"strategy={selected_strategy} "
        f"val_acc={selected['val']['accuracy']:.4f} "
        f"test_acc={selected['test']['accuracy']:.4f} "
        f"test_mae={selected['test']['mae_argmax']:.3f} "
        f"test_macro_f1={selected['test']['macro_f1']:.4f} "
        f"test_gate_mean={selected['test']['gate_mean']:.4f}"
    )
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote per-class CSV: {by_class_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
