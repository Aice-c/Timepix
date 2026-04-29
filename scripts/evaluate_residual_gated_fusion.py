#!/usr/bin/env python
"""Evaluate A4b-6 residual gated fusion over frozen logits.

This script reloads two frozen experts and tests whether the relative-ToA
candidate should act as a constrained correction to the ToT baseline:

    logits_final = logits_tot + residual_weight * (logits_candidate - logits_tot)

Validation selects the A4b-6 variant. Test is reported only after selection.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
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

from evaluate_gated_late_fusion import (  # noqa: E402
    DEFAULT_SLOPES,
    DEFAULT_THRESHOLDS,
    _entropy_soft_gates,
    _gate_features,
    _make_stratified_folds,
    _parse_csv_floats,
    _parse_fit_modes,
)
from evaluate_selector_fusion import (  # noqa: E402
    DEFAULT_EXPERIMENT_ROOT,
    _discover_candidate_run,
    _discover_tot_run,
    _errors,
    _load_json,
    _load_run_payload,
    _metadata_path,
    _metrics_from_preds,
    _rule_masks,
    _run_dir,
    _run_label,
    _standardize,
    _strategy_preds,
    _validate_pair,
    _write_csv,
)


DEFAULT_BETA_GRID = "0.05,0.10,0.20,0.30,0.50,0.75,1.00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A4b-6 residual gated fusion over frozen logits")
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
    parser.add_argument("--fit-modes", default="train,val-cv", help="Comma-separated learned fit modes")
    parser.add_argument("--cv-folds", type=int, default=5, help="Validation folds for val-cv learned residual gates")
    parser.add_argument("--gate-hidden-dim", type=int, default=0, help="Optional gate MLP hidden size; 0 means linear")
    parser.add_argument("--gate-epochs", type=int, default=500, help="Gate training epochs")
    parser.add_argument("--gate-lr", type=float, default=0.01, help="Gate learning rate")
    parser.add_argument("--gate-weight-decay", type=float, default=0.0001, help="Gate weight decay")
    parser.add_argument("--gate-seed", type=int, default=42, help="Gate initialization seed")
    parser.add_argument("--beta-grid", default=DEFAULT_BETA_GRID, help="Comma-separated beta grid for residual search")
    parser.add_argument(
        "--max-per-class-combos",
        type=int,
        default=50000,
        help="Safety cap for exhaustive per-class beta grid search",
    )
    parser.add_argument("--beta-init", type=float, default=0.10, help="Initial beta for learned residual variants")
    parser.add_argument("--conservative-init-bias", type=float, default=-2.0, help="ToT-biased gate init")
    parser.add_argument("--conservative-gate-l1", type=float, default=0.01, help="Mean-gate penalty")
    parser.add_argument("--conservative-beta-l1", type=float, default=0.001, help="Mean-beta penalty")
    parser.add_argument("--entropy-thresholds", default=DEFAULT_THRESHOLDS, help="Comma-separated entropy thresholds")
    parser.add_argument("--entropy-slopes", default=DEFAULT_SLOPES, help="Comma-separated entropy sigmoid slopes")
    parser.add_argument("--output-json", default=None, help="Output JSON path")
    parser.add_argument("--output-summary", default=None, help="Output summary CSV path")
    parser.add_argument("--output-by-class", default=None, help="Output per-class CSV path")
    return parser.parse_args()


def _tag_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _clip_probability(value: float) -> float:
    return min(max(float(value), 1e-4), 1.0 - 1e-4)


def _logit_probability(value: float) -> float:
    value = _clip_probability(value)
    return math.log(value / (1.0 - value))


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


def _broadcast_factor(factor: np.ndarray | float, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(factor, dtype=np.float32)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        if values.shape[0] == reference.shape[0]:
            values = values[:, None]
        else:
            values = values[None, :]
    return np.broadcast_to(values, reference.shape).astype(np.float32)


def _residual_logits(primary_logits: np.ndarray, candidate_logits: np.ndarray, weights: np.ndarray | float) -> np.ndarray:
    residual_weights = _broadcast_factor(weights, primary_logits)
    return primary_logits + residual_weights * (candidate_logits - primary_logits)


def _effective_factor(factor: np.ndarray, preds: np.ndarray | None = None) -> np.ndarray:
    if factor.ndim == 1 or factor.shape[1] == 1:
        return factor.reshape(-1)
    if preds is None:
        return factor.mean(axis=1)
    return factor[np.arange(factor.shape[0]), preds.astype(int)]


def _residual_stats(
    weights: np.ndarray,
    gates: np.ndarray | None,
    preds: np.ndarray,
    labels: np.ndarray,
    primary_logits: np.ndarray,
    candidate_logits: np.ndarray,
    angle_values: list[float],
) -> dict[str, Any]:
    effective_weight = _effective_factor(weights, preds)
    effective_gate = _effective_factor(gates, preds) if gates is not None else effective_weight
    primary_errors, _ = _errors(primary_logits, labels, angle_values)
    candidate_errors, _ = _errors(candidate_logits, labels, angle_values)
    high_weight = effective_weight >= 0.5
    beneficial = candidate_errors < primary_errors
    harmful = candidate_errors > primary_errors
    angles = np.asarray(angle_values, dtype=float)
    true_30 = np.isclose(angles[labels.astype(int)], 30.0)
    return {
        "gate_mean": float(effective_gate.mean()) if effective_gate.size else 0.0,
        "gate_std": float(effective_gate.std()) if effective_gate.size else 0.0,
        "gate_p90": float(np.quantile(effective_gate, 0.90)) if effective_gate.size else 0.0,
        "gate_high_rate": float((effective_gate >= 0.5).mean()) if effective_gate.size else 0.0,
        "gate_mean_true_30": float(effective_gate[true_30].mean()) if true_30.any() else 0.0,
        "residual_weight_mean": float(effective_weight.mean()) if effective_weight.size else 0.0,
        "residual_weight_std": float(effective_weight.std()) if effective_weight.size else 0.0,
        "residual_weight_p90": float(np.quantile(effective_weight, 0.90)) if effective_weight.size else 0.0,
        "residual_high_rate": float(high_weight.mean()) if high_weight.size else 0.0,
        "residual_weight_mean_true_30": float(effective_weight[true_30].mean()) if true_30.any() else 0.0,
        "beneficial_high_residual_count": int(np.logical_and(high_weight, beneficial).sum()),
        "harmful_high_residual_count": int(np.logical_and(high_weight, harmful).sum()),
        "neutral_high_residual_count": int(np.logical_and.reduce((high_weight, ~beneficial, ~harmful)).sum()),
    }


def _evaluate_residual(
    split: str,
    strategy: str,
    primary_logits: np.ndarray,
    candidate_logits: np.ndarray,
    labels: np.ndarray,
    angle_values: list[float],
    *,
    weights: np.ndarray | None = None,
    gates: np.ndarray | None = None,
) -> dict[str, Any]:
    if strategy in {"primary_only", "candidate_only", "oracle"} or strategy.startswith("rule:"):
        selector_probs = None
        if strategy.startswith("rule:"):
            rule_name = strategy.split(":", 1)[1]
            selector_probs = _rule_masks(primary_logits, candidate_logits, angle_values)[rule_name].astype(np.float32)
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
        residual_weights = select.astype(np.float32)[:, None]
        stats = _residual_stats(
            residual_weights,
            residual_weights,
            preds,
            labels,
            primary_logits,
            candidate_logits,
            angle_values,
        )
    else:
        if weights is None:
            raise ValueError("Residual strategies require weights")
        residual_weights = _broadcast_factor(weights, primary_logits)
        logits = _residual_logits(primary_logits, candidate_logits, residual_weights)
        preds = logits.argmax(axis=1)
        metrics = _metrics_from_preds(preds, labels, angle_values)
        gate_values = None if gates is None else _broadcast_factor(gates, primary_logits)
        stats = _residual_stats(
            residual_weights,
            gate_values,
            preds,
            labels,
            primary_logits,
            candidate_logits,
            angle_values,
        )
    metrics.update(
        {
            "split": split,
            "strategy": strategy,
            "n": int(labels.shape[0]),
            **stats,
            "selection_rate": stats["residual_high_rate"],
            "selected_count": int(round(stats["residual_high_rate"] * labels.shape[0])),
        }
    )
    return metrics


def _score(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("accuracy", 0.0)),
        -float(metrics.get("mae_argmax", 0.0)),
        float(metrics.get("macro_f1", 0.0)),
        -float(metrics.get("residual_weight_mean", metrics.get("gate_mean", 0.0))),
    )


class ResidualSpec:
    def __init__(
        self,
        *,
        name: str,
        beta_shape: str,
        fit_mode: str,
        init_bias: float = 0.0,
        gate_l1: float = 0.0,
        beta_l1: float = 0.0,
    ) -> None:
        self.name = name
        self.beta_shape = beta_shape
        self.fit_mode = fit_mode
        self.init_bias = float(init_bias)
        self.gate_l1 = float(gate_l1)
        self.beta_l1 = float(beta_l1)


def _torch_train_residual_gate(
    features: np.ndarray,
    primary_logits: np.ndarray,
    candidate_logits: np.ndarray,
    labels: np.ndarray,
    spec: ResidualSpec,
    args: argparse.Namespace,
    num_classes: int,
):
    import torch
    import torch.nn.functional as F
    from torch import nn

    class ResidualGateModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            output_dim = 1
            if int(args.gate_hidden_dim) > 0:
                self.gate = nn.Sequential(
                    nn.Linear(features.shape[1], int(args.gate_hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(int(args.gate_hidden_dim), output_dim),
                )
                final_layer = self.gate[-1]
            else:
                self.gate = nn.Linear(features.shape[1], output_dim)
                final_layer = self.gate
            if hasattr(final_layer, "bias") and final_layer.bias is not None:
                nn.init.constant_(final_layer.bias, spec.init_bias)
            beta_dim = num_classes if spec.beta_shape == "class" else 1
            beta_logit = _logit_probability(float(args.beta_init))
            self.beta_logit = nn.Parameter(torch.full((beta_dim,), float(beta_logit)))

        def forward(self, x):
            gate = torch.sigmoid(self.gate(x))
            beta = torch.sigmoid(self.beta_logit).view(1, -1)
            return gate, beta

    torch.manual_seed(int(args.gate_seed))
    model = ResidualGateModel()
    x = torch.from_numpy(features.astype(np.float32))
    p_logits = torch.from_numpy(primary_logits.astype(np.float32))
    c_logits = torch.from_numpy(candidate_logits.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.int64))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.gate_lr),
        weight_decay=float(args.gate_weight_decay),
    )
    final_loss = 0.0
    model.train()
    for _ in range(int(args.gate_epochs)):
        optimizer.zero_grad(set_to_none=True)
        gate, beta = model(x)
        mixed = p_logits + gate * beta * (c_logits - p_logits)
        loss = F.cross_entropy(mixed, y)
        if spec.gate_l1 > 0:
            loss = loss + float(spec.gate_l1) * gate.mean()
        if spec.beta_l1 > 0:
            loss = loss + float(spec.beta_l1) * beta.mean()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    model.eval()
    with torch.no_grad():
        _, beta = model(x[:1])
        beta_values = beta.cpu().numpy().reshape(-1).astype(float).tolist()
    info = {
        "input_dim": int(features.shape[1]),
        "beta_shape": spec.beta_shape,
        "fit_mode": spec.fit_mode,
        "init_bias": spec.init_bias,
        "gate_l1": spec.gate_l1,
        "beta_l1": spec.beta_l1,
        "hidden_dim": int(args.gate_hidden_dim),
        "epochs": int(args.gate_epochs),
        "lr": float(args.gate_lr),
        "weight_decay": float(args.gate_weight_decay),
        "beta_init": float(args.beta_init),
        "beta_values": beta_values,
        "beta_mean": float(np.mean(beta_values)) if beta_values else 0.0,
        "final_loss": final_loss,
    }
    return model, info


def _predict_residual(model, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    import torch

    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32))
        gate, beta = model(x)
        weights = gate * beta
        return (
            gate.cpu().numpy().astype(np.float32),
            weights.cpu().numpy().astype(np.float32),
            beta.cpu().numpy().reshape(-1).astype(float).tolist(),
        )


def _fit_train_residual(
    raw_features: dict[str, np.ndarray],
    split_data: dict[str, dict[str, Any]],
    angle_values: list[float],
    spec: ResidualSpec,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    train_features, [val_features, test_features], _ = _standardize(
        raw_features["train"],
        raw_features["val"],
        raw_features["test"],
    )
    model, info = _torch_train_residual_gate(
        train_features,
        split_data["train"]["primary_logits"],
        split_data["train"]["candidate_logits"],
        split_data["train"]["labels"],
        spec,
        args,
        len(angle_values),
    )
    gates: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    for split, features in [("train", train_features), ("val", val_features), ("test", test_features)]:
        gates[split], weights[split], _ = _predict_residual(model, features)
    return weights, gates, info


def _fit_val_cv_residual(
    raw_features: dict[str, np.ndarray],
    split_data: dict[str, dict[str, Any]],
    angle_values: list[float],
    spec: ResidualSpec,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    val_features = raw_features["val"]
    primary_errors, _ = _errors(split_data["val"]["primary_logits"], split_data["val"]["labels"], angle_values)
    candidate_errors, _ = _errors(split_data["val"]["candidate_logits"], split_data["val"]["labels"], angle_values)
    target = (candidate_errors < primary_errors).astype(np.float32)
    folds = _make_stratified_folds(target, int(args.cv_folds), int(args.gate_seed))
    oof_gates = np.zeros((target.shape[0], 1), dtype=np.float32)
    beta_dim = len(angle_values) if spec.beta_shape == "class" else 1
    oof_weights = np.zeros((target.shape[0], beta_dim), dtype=np.float32)
    fold_infos = []
    for fold_index, heldout_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(target.shape[0]), heldout_idx, assume_unique=True)
        fold_train, [fold_heldout], _ = _standardize(val_features[train_idx], val_features[heldout_idx])
        model, fold_info = _torch_train_residual_gate(
            fold_train,
            split_data["val"]["primary_logits"][train_idx],
            split_data["val"]["candidate_logits"][train_idx],
            split_data["val"]["labels"][train_idx],
            spec,
            args,
            len(angle_values),
        )
        gate_values, weight_values, _ = _predict_residual(model, fold_heldout)
        oof_gates[heldout_idx] = gate_values
        oof_weights[heldout_idx] = weight_values
        fold_infos.append(
            {
                "fold": fold_index,
                "train_count": int(train_idx.shape[0]),
                "heldout_count": int(heldout_idx.shape[0]),
                "heldout_positive_rate": float(target[heldout_idx].mean()) if heldout_idx.size else 0.0,
                "beta_values": fold_info.get("beta_values"),
                "final_loss": fold_info.get("final_loss"),
            }
        )

    full_val_features, [train_features, test_features], _ = _standardize(
        raw_features["val"],
        raw_features["train"],
        raw_features["test"],
    )
    model, info = _torch_train_residual_gate(
        full_val_features,
        split_data["val"]["primary_logits"],
        split_data["val"]["candidate_logits"],
        split_data["val"]["labels"],
        spec,
        args,
        len(angle_values),
    )
    train_gates, train_weights, _ = _predict_residual(model, train_features)
    test_gates, test_weights, _ = _predict_residual(model, test_features)
    info["cv_folds"] = int(len(folds))
    info["cv_fold_info"] = fold_infos
    info["validation_weights"] = "out_of_fold"
    info["final_fit"] = "full_validation"
    return (
        {"train": train_weights, "val": oof_weights, "test": test_weights},
        {"train": train_gates, "val": oof_gates, "test": test_gates},
        info,
    )


def _best_per_class_beta(
    split_data: dict[str, dict[str, Any]],
    angle_values: list[float],
    beta_values: list[float],
    max_combos: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    num_classes = len(angle_values)
    total = len(beta_values) ** num_classes
    if total > int(max_combos):
        raise ValueError(
            f"Per-class beta grid would test {total} combinations; "
            f"increase --max-per-class-combos or shrink --beta-grid."
        )
    best_beta = None
    best_metrics = None
    best_score = None
    for combo in itertools.product(beta_values, repeat=num_classes):
        weights = np.asarray(combo, dtype=np.float32)[None, :]
        metrics = _evaluate_residual(
            "val",
            "a4b6b_per_class_beta_grid",
            split_data["val"]["primary_logits"],
            split_data["val"]["candidate_logits"],
            split_data["val"]["labels"],
            angle_values,
            weights=weights,
        )
        score = _score(metrics)
        if best_score is None or score > best_score:
            best_score = score
            best_beta = weights
            best_metrics = metrics
    assert best_beta is not None and best_metrics is not None
    return best_beta.astype(np.float32), {
        "searched_combinations": int(total),
        "beta_values": best_beta.reshape(-1).astype(float).tolist(),
        "validation_score": best_score,
        "validation_metrics": {
            "accuracy": best_metrics["accuracy"],
            "mae_argmax": best_metrics["mae_argmax"],
            "macro_f1": best_metrics["macro_f1"],
            "residual_weight_mean": best_metrics["residual_weight_mean"],
        },
    }


def _row_from_metrics(
    seed: int,
    strategy: str,
    selected: bool,
    metrics_by_split: dict[str, dict[str, Any]],
    variant_info: dict[str, Any],
    primary_run: Path,
    candidate_run: Path,
) -> dict[str, Any]:
    beta_values = variant_info.get("beta_values")
    if isinstance(beta_values, list):
        beta_values_text = ";".join(f"{float(item):.6g}" for item in beta_values)
    else:
        beta_values_text = beta_values
    row: dict[str, Any] = {
        "seed": seed,
        "strategy": strategy,
        "selected_by_val": selected,
        "selector_mode": "residual" if strategy.startswith("a4b6") else variant_info.get("selector_mode"),
        "selector_fit": variant_info.get("fit_mode") or variant_info.get("selector_fit"),
        "variant_family": variant_info.get("variant_family"),
        "variant_id": variant_info.get("variant_id"),
        "fit_mode": variant_info.get("fit_mode"),
        "beta_shape": variant_info.get("beta_shape"),
        "beta_value": variant_info.get("beta_value"),
        "beta_values": beta_values_text,
        "beta_mean": variant_info.get("beta_mean"),
        "entropy_threshold": variant_info.get("entropy_threshold"),
        "entropy_slope": variant_info.get("entropy_slope"),
        "gate_l1": variant_info.get("gate_l1"),
        "beta_l1": variant_info.get("beta_l1"),
        "init_bias": variant_info.get("init_bias"),
        "beta_init": variant_info.get("beta_init"),
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
        row[prefix + "selection_rate"] = metrics["selection_rate"]
        row[prefix + "selected_count"] = metrics["selected_count"]
        row[prefix + "gate_mean"] = metrics["gate_mean"]
        row[prefix + "gate_high_rate"] = metrics["gate_high_rate"]
        row[prefix + "gate_mean_true_30"] = metrics["gate_mean_true_30"]
        row[prefix + "residual_weight_mean"] = metrics["residual_weight_mean"]
        row[prefix + "residual_high_rate"] = metrics["residual_high_rate"]
        row[prefix + "residual_weight_mean_true_30"] = metrics["residual_weight_mean_true_30"]
        row[prefix + "beneficial_high_residual_count"] = metrics["beneficial_high_residual_count"]
        row[prefix + "harmful_high_residual_count"] = metrics["harmful_high_residual_count"]
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
    stem = f"a4b_6_residual_gated_fusion_seed{args.seed}"
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
    weights_by_strategy: dict[str, dict[str, np.ndarray]] = {}
    gates_by_strategy: dict[str, dict[str, np.ndarray]] = {}

    for strategy, family in [
        ("primary_only", "baseline"),
        ("candidate_only", "baseline"),
        ("rule:rule_entropy_adv_0p03", "a4b4_rule"),
        ("oracle", "oracle"),
    ]:
        strategies.append(strategy)
        variant_info[strategy] = {"variant_family": family, "variant_id": strategy}

    beta_values = _parse_csv_floats(args.beta_grid)
    for beta in beta_values:
        strategy = f"a4b6a_scalar_beta_b{_tag_float(beta)}"
        strategies.append(strategy)
        weights_by_strategy[strategy] = {
            split: np.full((split_data[split]["labels"].shape[0], 1), float(beta), dtype=np.float32)
            for split in ["train", "val", "test"]
        }
        variant_info[strategy] = {
            "variant_family": "a4b6a_scalar_beta_residual",
            "variant_id": "A4b-6a",
            "fit_mode": "validation_grid",
            "beta_shape": "scalar",
            "beta_value": float(beta),
            "beta_mean": float(beta),
        }

    best_beta, best_beta_info = _best_per_class_beta(
        split_data,
        angle_values,
        beta_values,
        int(args.max_per_class_combos),
    )
    strategy = "a4b6b_per_class_beta_grid"
    strategies.append(strategy)
    weights_by_strategy[strategy] = {
        split: np.broadcast_to(best_beta, split_data[split]["primary_logits"].shape).astype(np.float32)
        for split in ["train", "val", "test"]
    }
    variant_info[strategy] = {
        "variant_family": "a4b6b_per_class_beta_residual",
        "variant_id": "A4b-6b",
        "fit_mode": "validation_grid",
        "beta_shape": "class",
        **best_beta_info,
        "beta_mean": float(np.mean(best_beta)),
    }

    thresholds = _parse_csv_floats(args.entropy_thresholds)
    slopes = _parse_csv_floats(args.entropy_slopes)
    for threshold in thresholds:
        for slope in slopes:
            entropy_gates = {
                split: _entropy_soft_gates(
                    split_data[split]["primary_logits"],
                    split_data[split]["candidate_logits"],
                    threshold,
                    slope,
                )
                for split in ["train", "val", "test"]
            }
            for beta in beta_values:
                strategy = (
                    f"a4b6e_entropy_residual_t{_tag_float(threshold)}_"
                    f"k{_tag_float(slope)}_b{_tag_float(beta)}"
                )
                strategies.append(strategy)
                gates_by_strategy[strategy] = entropy_gates
                weights_by_strategy[strategy] = {
                    split: entropy_gates[split] * float(beta) for split in ["train", "val", "test"]
                }
                variant_info[strategy] = {
                    "variant_family": "a4b6e_entropy_constrained_residual",
                    "variant_id": "A4b-6e",
                    "fit_mode": "validation_grid",
                    "beta_shape": "scalar",
                    "beta_value": float(beta),
                    "beta_mean": float(beta),
                    "entropy_threshold": float(threshold),
                    "entropy_slope": float(slope),
                }

    fit_modes = _parse_fit_modes(args.fit_modes)
    learned_specs: list[ResidualSpec] = []
    for fit_mode in fit_modes:
        learned_specs.extend(
            [
                ResidualSpec(name=f"a4b6c_learned_gate_scalar_beta_{fit_mode}", beta_shape="scalar", fit_mode=fit_mode),
                ResidualSpec(name=f"a4b6d_learned_gate_class_beta_{fit_mode}", beta_shape="class", fit_mode=fit_mode),
                ResidualSpec(
                    name=f"a4b6e_conservative_learned_scalar_{fit_mode}",
                    beta_shape="scalar",
                    fit_mode=fit_mode,
                    init_bias=float(args.conservative_init_bias),
                    gate_l1=float(args.conservative_gate_l1),
                    beta_l1=float(args.conservative_beta_l1),
                ),
            ]
        )

    for spec in learned_specs:
        if spec.fit_mode == "train":
            weight_predictions, gate_predictions, info = _fit_train_residual(
                raw_features,
                split_data,
                angle_values,
                spec,
                args,
            )
        elif spec.fit_mode == "val-cv":
            weight_predictions, gate_predictions, info = _fit_val_cv_residual(
                raw_features,
                split_data,
                angle_values,
                spec,
                args,
            )
        else:
            raise ValueError(f"Unknown fit mode: {spec.fit_mode}")
        strategies.append(spec.name)
        weights_by_strategy[spec.name] = weight_predictions
        gates_by_strategy[spec.name] = gate_predictions
        family = spec.name.split("_", 1)[0]
        variant_id = {
            "a4b6c": "A4b-6c",
            "a4b6d": "A4b-6d",
            "a4b6e": "A4b-6e",
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
            metrics_by_strategy[strategy][split] = _evaluate_residual(
                split,
                strategy,
                split_data[split]["primary_logits"],
                split_data[split]["candidate_logits"],
                split_data[split]["labels"],
                angle_values,
                weights=weights_by_strategy.get(strategy, {}).get(split),
                gates=gates_by_strategy.get(strategy, {}).get(split),
            )

    selectable = [strategy for strategy in strategies if strategy.startswith("a4b6")]
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
        "analysis": "a4b_6_residual_gated_fusion",
        "seed": args.seed,
        "primary_run": str(primary_run),
        "candidate_run": str(candidate_run),
        "primary": _run_label(primary_run),
        "candidate": _run_label(candidate_run),
        "selected_strategy": selected_strategy,
        "selection_rule": (
            "max validation accuracy, then lower validation MAE, then higher macro-F1, "
            "then lower residual weight"
        ),
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
        "Selected A4b-6 residual fusion | "
        f"strategy={selected_strategy} "
        f"val_acc={selected['val']['accuracy']:.4f} "
        f"test_acc={selected['test']['accuracy']:.4f} "
        f"test_mae={selected['test']['mae_argmax']:.3f} "
        f"test_macro_f1={selected['test']['macro_f1']:.4f} "
        f"test_residual_weight={selected['test']['residual_weight_mean']:.4f}"
    )
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote per-class CSV: {by_class_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
