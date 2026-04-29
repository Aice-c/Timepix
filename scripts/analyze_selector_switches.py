#!/usr/bin/env python
"""Diagnose selector switch behavior for A4b-4d.

This script reloads the same frozen ToT and candidate checkpoints used by
A4b-4, applies a fixed validation-selected rule such as
``entropy_adv_0p03``, and reports how often switches are beneficial, harmful,
neutral, or missed. It does not train a model and does not choose a new rule.
"""

from __future__ import annotations

import argparse
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
    _add_hit_mask,
    _checkpoint_path,
    _discover_candidate_run,
    _discover_tot_run,
    _entropy,
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
    _softmax,
    _toa_transform,
    _training_seed,
    _validate_pair,
    _write_csv,
)


DEFAULT_RULE = "entropy_adv_0p03"
DEFAULT_SPLITS = "val,test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A4b-4d selector switch diagnostics")
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
        "--rule",
        default=DEFAULT_RULE,
        help="Fixed A4b-4 rule to diagnose, e.g. entropy_adv_0p03 or rule_entropy_adv_0p03",
    )
    parser.add_argument(
        "--splits",
        default=DEFAULT_SPLITS,
        help="Comma-separated splits to report; usually val,test",
    )
    parser.add_argument("--output-json", default=None, help="Output JSON path")
    parser.add_argument("--output-summary", default=None, help="Output summary CSV path")
    parser.add_argument("--output-by-class", default=None, help="Output per-class switch CSV path")
    parser.add_argument("--output-samples", default=None, help="Output per-sample CSV path")
    parser.add_argument("--output-distribution", default=None, help="Output score-distribution CSV path")
    return parser.parse_args()


def _normalize_rule_name(raw: str) -> str:
    rule = raw.strip()
    if rule.startswith("rule:"):
        rule = rule.split(":", 1)[1]
    if not rule.startswith("rule_"):
        rule = f"rule_{rule}"
    return rule


def _display_rule_name(rule: str) -> str:
    return rule[5:] if rule.startswith("rule_") else rule


def _parse_splits(raw: str) -> list[str]:
    splits = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"train", "val", "test"}
    bad = sorted(set(splits) - allowed)
    if bad:
        raise ValueError(f"Unknown splits: {bad}")
    if not splits:
        raise ValueError("--splits must contain at least one split")
    return splits


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return float(numerator) / denominator


def _score_arrays(
    primary_logits: np.ndarray,
    candidate_logits: np.ndarray,
    angle_values: list[float],
) -> dict[str, np.ndarray]:
    primary_probs = _softmax(primary_logits)
    candidate_probs = _softmax(candidate_logits)
    primary_preds = primary_probs.argmax(axis=1)
    candidate_preds = candidate_probs.argmax(axis=1)
    angles = np.asarray(angle_values, dtype=float)
    max_span = float(max(angles.max() - angles.min(), 1.0))
    primary_conf = primary_probs.max(axis=1)
    candidate_conf = candidate_probs.max(axis=1)
    primary_margin = _margin(primary_probs).squeeze(1)
    candidate_margin = _margin(candidate_probs).squeeze(1)
    primary_entropy = _entropy(primary_probs).squeeze(1)
    candidate_entropy = _entropy(candidate_probs).squeeze(1)
    return {
        "primary_conf": primary_conf,
        "candidate_conf": candidate_conf,
        "confidence_adv": candidate_conf - primary_conf,
        "primary_margin": primary_margin,
        "candidate_margin": candidate_margin,
        "margin_adv": candidate_margin - primary_margin,
        "primary_entropy": primary_entropy,
        "candidate_entropy": candidate_entropy,
        "entropy_adv": primary_entropy - candidate_entropy,
        "pred_angle_diff": np.abs(angles[primary_preds] - angles[candidate_preds]) / max_span,
        "disagree": (primary_preds != candidate_preds).astype(float),
    }


def _final_preds(primary_preds: np.ndarray, candidate_preds: np.ndarray, select: np.ndarray) -> np.ndarray:
    return np.where(select, candidate_preds, primary_preds)


def _split_context(
    primary: dict[str, Any],
    candidate: dict[str, Any],
    split: str,
    rule_name: str,
) -> dict[str, Any]:
    _validate_pair(primary, candidate, split)
    angle_values = primary["angle_values"]
    primary_split = primary["splits"][split]
    candidate_split = candidate["splits"][split]
    primary_logits = primary_split["logits"]
    candidate_logits = candidate_split["logits"]
    labels = primary_split["labels"]
    rule_masks = _rule_masks(primary_logits, candidate_logits, angle_values)
    if rule_name not in rule_masks:
        choices = ", ".join(sorted(rule_masks))
        raise ValueError(f"Unknown rule {rule_name!r}. Available rules: {choices}")
    primary_errors, primary_preds = _errors(primary_logits, labels, angle_values)
    candidate_errors, candidate_preds = _errors(candidate_logits, labels, angle_values)
    select = rule_masks[rule_name].astype(bool)
    beneficial = candidate_errors < primary_errors
    harmful = candidate_errors > primary_errors
    neutral = candidate_errors == primary_errors
    final_preds = _final_preds(primary_preds, candidate_preds, select)
    oracle_preds = _final_preds(primary_preds, candidate_preds, beneficial)
    return {
        "split": split,
        "keys": primary_split["keys"],
        "labels": labels,
        "angle_values": angle_values,
        "primary_logits": primary_logits,
        "candidate_logits": candidate_logits,
        "primary_preds": primary_preds,
        "candidate_preds": candidate_preds,
        "primary_errors": primary_errors,
        "candidate_errors": candidate_errors,
        "selected": select,
        "beneficial": beneficial,
        "harmful": harmful,
        "neutral": neutral,
        "final_preds": final_preds,
        "oracle_preds": oracle_preds,
        "scores": _score_arrays(primary_logits, candidate_logits, angle_values),
    }


def _summary_row(ctx: dict[str, Any], rule_label: str, seed: int | None) -> dict[str, Any]:
    labels = ctx["labels"]
    selected = ctx["selected"]
    beneficial = ctx["beneficial"]
    harmful = ctx["harmful"]
    neutral = ctx["neutral"]
    n = int(labels.shape[0])
    selected_count = int(selected.sum())
    oracle_count = int(beneficial.sum())
    beneficial_selected = int(np.logical_and(selected, beneficial).sum())
    harmful_selected = int(np.logical_and(selected, harmful).sum())
    neutral_selected = int(np.logical_and(selected, neutral).sum())
    missed_beneficial = int(np.logical_and(~selected, beneficial).sum())

    primary_metrics = _metrics_from_preds(ctx["primary_preds"], labels, ctx["angle_values"])
    candidate_metrics = _metrics_from_preds(ctx["candidate_preds"], labels, ctx["angle_values"])
    final_metrics = _metrics_from_preds(ctx["final_preds"], labels, ctx["angle_values"])
    oracle_metrics = _metrics_from_preds(ctx["oracle_preds"], labels, ctx["angle_values"])
    return {
        "seed": seed,
        "split": ctx["split"],
        "rule": rule_label,
        "n": n,
        "selected_count": selected_count,
        "selection_rate": _safe_rate(selected_count, n),
        "oracle_count": oracle_count,
        "oracle_selection_rate": _safe_rate(oracle_count, n),
        "beneficial_selected_count": beneficial_selected,
        "harmful_selected_count": harmful_selected,
        "neutral_selected_count": neutral_selected,
        "missed_beneficial_count": missed_beneficial,
        "switch_precision_lower_error": _safe_rate(beneficial_selected, selected_count),
        "switch_recall_lower_error": _safe_rate(beneficial_selected, oracle_count),
        "harmful_switch_rate": _safe_rate(harmful_selected, selected_count),
        "neutral_switch_rate": _safe_rate(neutral_selected, selected_count),
        "primary_accuracy": primary_metrics["accuracy"],
        "candidate_accuracy": candidate_metrics["accuracy"],
        "final_accuracy": final_metrics["accuracy"],
        "oracle_accuracy": oracle_metrics["accuracy"],
        "primary_mae": primary_metrics["mae_argmax"],
        "candidate_mae": candidate_metrics["mae_argmax"],
        "final_mae": final_metrics["mae_argmax"],
        "oracle_mae": oracle_metrics["mae_argmax"],
        "primary_macro_f1": primary_metrics["macro_f1"],
        "candidate_macro_f1": candidate_metrics["macro_f1"],
        "final_macro_f1": final_metrics["macro_f1"],
        "oracle_macro_f1": oracle_metrics["macro_f1"],
    }


def _by_class_rows(ctx: dict[str, Any], rule_label: str, seed: int | None) -> list[dict[str, Any]]:
    rows = []
    labels = ctx["labels"]
    angles = np.asarray(ctx["angle_values"], dtype=float)
    selected = ctx["selected"]
    beneficial = ctx["beneficial"]
    harmful = ctx["harmful"]
    neutral = ctx["neutral"]
    for cls, angle in enumerate(angles.tolist()):
        mask = labels == cls
        n = int(mask.sum())
        selected_count = int(np.logical_and(mask, selected).sum())
        oracle_count = int(np.logical_and(mask, beneficial).sum())
        beneficial_selected = int(np.logical_and.reduce((mask, selected, beneficial)).sum())
        harmful_selected = int(np.logical_and.reduce((mask, selected, harmful)).sum())
        neutral_selected = int(np.logical_and.reduce((mask, selected, neutral)).sum())
        missed_beneficial = int(np.logical_and.reduce((mask, ~selected, beneficial)).sum())
        row = {
            "seed": seed,
            "split": ctx["split"],
            "rule": rule_label,
            "class_index": cls,
            "class_angle": float(angle),
            "n": n,
            "selected_count": selected_count,
            "selection_rate": _safe_rate(selected_count, n),
            "oracle_count": oracle_count,
            "oracle_selection_rate": _safe_rate(oracle_count, n),
            "beneficial_selected_count": beneficial_selected,
            "harmful_selected_count": harmful_selected,
            "neutral_selected_count": neutral_selected,
            "missed_beneficial_count": missed_beneficial,
            "switch_precision_lower_error": _safe_rate(beneficial_selected, selected_count),
            "switch_recall_lower_error": _safe_rate(beneficial_selected, oracle_count),
            "harmful_switch_rate": _safe_rate(harmful_selected, selected_count),
            "neutral_switch_rate": _safe_rate(neutral_selected, selected_count),
            "primary_accuracy": _safe_rate(int((ctx["primary_preds"][mask] == labels[mask]).sum()), n),
            "candidate_accuracy": _safe_rate(int((ctx["candidate_preds"][mask] == labels[mask]).sum()), n),
            "final_accuracy": _safe_rate(int((ctx["final_preds"][mask] == labels[mask]).sum()), n),
            "oracle_accuracy": _safe_rate(int((ctx["oracle_preds"][mask] == labels[mask]).sum()), n),
        }
        rows.append(row)
    return rows


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _distribution_rows(ctx: dict[str, Any], rule_label: str, seed: int | None) -> list[dict[str, Any]]:
    selected = ctx["selected"]
    beneficial = ctx["beneficial"]
    harmful = ctx["harmful"]
    neutral = ctx["neutral"]
    groups = {
        "selected_beneficial": selected & beneficial,
        "selected_harmful": selected & harmful,
        "selected_neutral": selected & neutral,
        "missed_beneficial": (~selected) & beneficial,
        "not_selected_not_oracle": (~selected) & (~beneficial),
        "all_selected": selected,
        "all_oracle_beneficial": beneficial,
    }
    score_names = [
        "entropy_adv",
        "confidence_adv",
        "margin_adv",
        "primary_entropy",
        "candidate_entropy",
        "primary_conf",
        "candidate_conf",
        "primary_margin",
        "candidate_margin",
        "pred_angle_diff",
    ]
    rows = []
    for group_name, mask in groups.items():
        for score_name in score_names:
            values = ctx["scores"][score_name][mask]
            stats = _quantiles(values)
            rows.append(
                {
                    "seed": seed,
                    "split": ctx["split"],
                    "rule": rule_label,
                    "group": group_name,
                    "score": score_name,
                    "n": int(mask.sum()),
                    **stats,
                }
            )
    return rows


def _sample_rows(ctx: dict[str, Any], rule_label: str, seed: int | None) -> list[dict[str, Any]]:
    rows = []
    angles = np.asarray(ctx["angle_values"], dtype=float)
    selected = ctx["selected"]
    beneficial = ctx["beneficial"]
    harmful = ctx["harmful"]
    neutral = ctx["neutral"]
    scores = ctx["scores"]
    for i, key in enumerate(ctx["keys"]):
        if selected[i] and beneficial[i]:
            outcome = "selected_beneficial"
        elif selected[i] and harmful[i]:
            outcome = "selected_harmful"
        elif selected[i] and neutral[i]:
            outcome = "selected_neutral"
        elif (not selected[i]) and beneficial[i]:
            outcome = "missed_beneficial"
        else:
            outcome = "not_selected_not_oracle"
        label = int(ctx["labels"][i])
        primary_pred = int(ctx["primary_preds"][i])
        candidate_pred = int(ctx["candidate_preds"][i])
        final_pred = int(ctx["final_preds"][i])
        rows.append(
            {
                "seed": seed,
                "split": ctx["split"],
                "rule": rule_label,
                "key": key,
                "label_class": label,
                "label_angle": float(angles[label]),
                "primary_pred_class": primary_pred,
                "primary_pred_angle": float(angles[primary_pred]),
                "candidate_pred_class": candidate_pred,
                "candidate_pred_angle": float(angles[candidate_pred]),
                "final_pred_class": final_pred,
                "final_pred_angle": float(angles[final_pred]),
                "primary_error": float(ctx["primary_errors"][i]),
                "candidate_error": float(ctx["candidate_errors"][i]),
                "selected": bool(selected[i]),
                "oracle_beneficial": bool(beneficial[i]),
                "harmful_if_selected": bool(harmful[i]),
                "neutral_if_selected": bool(neutral[i]),
                "switch_outcome": outcome,
                "primary_correct": bool(primary_pred == label),
                "candidate_correct": bool(candidate_pred == label),
                "final_correct": bool(final_pred == label),
                "entropy_adv": float(scores["entropy_adv"][i]),
                "confidence_adv": float(scores["confidence_adv"][i]),
                "margin_adv": float(scores["margin_adv"][i]),
                "primary_entropy": float(scores["primary_entropy"][i]),
                "candidate_entropy": float(scores["candidate_entropy"][i]),
                "primary_conf": float(scores["primary_conf"][i]),
                "candidate_conf": float(scores["candidate_conf"][i]),
                "primary_margin": float(scores["primary_margin"][i]),
                "candidate_margin": float(scores["candidate_margin"][i]),
                "pred_angle_diff": float(scores["pred_angle_diff"][i]),
                "disagree": bool(scores["disagree"][i]),
            }
        )
    return rows


def _default_outputs(args: argparse.Namespace, rule_label: str) -> tuple[Path, Path, Path, Path, Path]:
    stem = f"a4b_4d_switch_diagnostics_{rule_label}_seed{args.seed}"
    json_path = Path(args.output_json) if args.output_json else Path("outputs") / f"{stem}.json"
    summary_path = Path(args.output_summary) if args.output_summary else Path("outputs") / f"{stem}_summary.csv"
    by_class_path = Path(args.output_by_class) if args.output_by_class else Path("outputs") / f"{stem}_by_class.csv"
    sample_path = Path(args.output_samples) if args.output_samples else Path("outputs") / f"{stem}_samples.csv"
    distribution_path = (
        Path(args.output_distribution) if args.output_distribution else Path("outputs") / f"{stem}_distribution.csv"
    )
    return json_path, summary_path, by_class_path, sample_path, distribution_path


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


def main() -> int:
    args = parse_args()
    rule_name = _normalize_rule_name(args.rule)
    rule_label = _display_rule_name(rule_name)
    splits = _parse_splits(args.splits)

    primary_run, candidate_run = _select_runs(args)
    primary = _load_run_payload(primary_run, args)
    candidate = _load_run_payload(candidate_run, args)

    summary_rows: list[dict[str, Any]] = []
    by_class_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    contexts: dict[str, dict[str, Any]] = {}

    for split in splits:
        ctx = _split_context(primary, candidate, split, rule_name)
        contexts[split] = ctx
        summary_rows.append(_summary_row(ctx, rule_label, args.seed))
        by_class_rows.extend(_by_class_rows(ctx, rule_label, args.seed))
        sample_rows.extend(_sample_rows(ctx, rule_label, args.seed))
        distribution_rows.extend(_distribution_rows(ctx, rule_label, args.seed))

    json_path, summary_path, by_class_path, sample_path, distribution_path = _default_outputs(args, rule_label)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "analysis": "a4b_4d_selector_switch_diagnostics",
        "seed": args.seed,
        "splits": splits,
        "rule": rule_label,
        "rule_internal_name": rule_name,
        "rule_definition": {
            "entropy_adv_0p03": "candidate selected when predictions disagree and candidate_entropy <= primary_entropy - 0.03"
        }
        if rule_label == "entropy_adv_0p03"
        else {},
        "primary_run": str(primary_run),
        "candidate_run": str(candidate_run),
        "primary": _run_label(primary_run),
        "candidate": _run_label(candidate_run),
        "primary_metadata": {
            "modalities": _modalities(_load_json(_metadata_path(primary_run))),
            "seed": _training_seed(_load_json(_metadata_path(primary_run))),
            "checkpoint": str(_checkpoint_path(primary_run)),
        },
        "candidate_metadata": {
            "modalities": _modalities(_load_json(_metadata_path(candidate_run))),
            "seed": _training_seed(_load_json(_metadata_path(candidate_run))),
            "toa_transform": _toa_transform(_load_json(_metadata_path(candidate_run))),
            "add_hit_mask": _add_hit_mask(_load_json(_metadata_path(candidate_run))),
            "checkpoint": str(_checkpoint_path(candidate_run)),
        },
        "summary": summary_rows,
        "outputs": {
            "summary_csv": str(summary_path),
            "by_class_csv": str(by_class_path),
            "samples_csv": str(sample_path),
            "distribution_csv": str(distribution_path),
        },
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary_path, summary_rows)
    _write_csv(by_class_path, by_class_rows)
    _write_csv(sample_path, sample_rows)
    _write_csv(distribution_path, distribution_rows)

    test_row = next((row for row in summary_rows if row["split"] == "test"), summary_rows[-1])
    print(
        "A4b-4d switch diagnostics | "
        f"rule={rule_label} split={test_row['split']} "
        f"final_acc={test_row['final_accuracy']:.4f} "
        f"selection_rate={test_row['selection_rate']:.4f} "
        f"switch_precision={test_row['switch_precision_lower_error']:.4f} "
        f"switch_recall={test_row['switch_recall_lower_error']:.4f} "
        f"harmful_rate={test_row['harmful_switch_rate']:.4f}"
    )
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote per-class CSV: {by_class_path}")
    print(f"Wrote sample CSV: {sample_path}")
    print(f"Wrote distribution CSV: {distribution_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
