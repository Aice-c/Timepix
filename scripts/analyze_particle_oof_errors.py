#!/usr/bin/env python
"""Analyze full-dataset OOF particle classification errors across models."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_MODEL_ORDER = ["tot", "dual_concat", "gmu_totstrong"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, help="Experiment group under outputs/experiments")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="LABEL=GLOB",
        help="Model label and predictions glob relative to the group directory. Can be repeated.",
    )
    parser.add_argument("--out-dir", required=True, help="Output diagnostics directory")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _parse_model_specs(items: list[str]) -> dict[str, str]:
    if not items:
        return {
            "tot": "*p9a_ptype_stage1_gmm02_p_v3_oof_tot*/predictions.csv",
            "dual_concat": "*p9a_ptype_stage1_gmm02_p_v3_oof_dual_concat*/predictions.csv",
            "gmu_totstrong": "*p9a_ptype_stage1_gmm02_p_v3_oof_gmu_totstrong*/predictions.csv",
        }
    specs = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--model must be LABEL=GLOB, got: {item}")
        label, pattern = item.split("=", 1)
        label = label.strip()
        pattern = pattern.strip()
        if not label or not pattern:
            raise ValueError(f"--model must be LABEL=GLOB, got: {item}")
        specs[label] = pattern
    return specs


def _prob_confidence(row: dict[str, str]) -> float:
    probs = []
    for key, value in row.items():
        if key.startswith("prob_") and value != "":
            try:
                probs.append(float(value))
            except ValueError:
                pass
    return max(probs) if probs else 0.0


def _fold_from_metadata(run_dir: Path) -> tuple[int, list[str]]:
    metadata = _load_json(run_dir / "metadata.json")
    split_path = Path(metadata["data_info"]["split_path"])
    split_payload = _load_json(split_path)
    split_meta = split_payload.get("metadata", {})
    if "fold" in split_meta:
        fold = int(split_meta["fold"])
    else:
        stem = split_path.stem
        marker = "_fold"
        if marker not in stem:
            raise ValueError(f"Cannot infer OOF fold from split path: {split_path}")
        fold = int(stem.rsplit(marker, 1)[1])
    return fold, [str(key) for key in split_payload["test"]]


def _load_model_predictions(group_root: Path, label: str, pattern: str) -> list[dict[str, Any]]:
    paths = sorted(Path(path) for path in glob.glob(str(group_root / pattern)))
    if not paths:
        raise FileNotFoundError(f"No predictions matched for {label}: {group_root / pattern}")
    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for pred_path in paths:
        run_dir = pred_path.parent
        fold, test_keys = _fold_from_metadata(run_dir)
        pred_rows = _read_csv(pred_path)
        if len(pred_rows) != len(test_keys):
            raise ValueError(f"Row/key mismatch for {pred_path}: rows={len(pred_rows)} keys={len(test_keys)}")
        for idx, (row, sample_key) in enumerate(zip(pred_rows, test_keys)):
            if sample_key in seen_keys:
                raise ValueError(f"Duplicate OOF sample for {label}: {sample_key}")
            seen_keys.add(sample_key)
            true_class = str(row.get("true_class", sample_key.split("/", 1)[0]))
            pred_class = str(row.get("pred_class", ""))
            rows.append(
                {
                    "model": label,
                    "fold": fold,
                    "row": int(row.get("row", idx)),
                    "sample_key": sample_key,
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "correct": true_class == pred_class,
                    "confidence": _prob_confidence(row),
                    "predictions_csv": str(pred_path),
                }
            )
    return sorted(rows, key=lambda row: row["sample_key"])


def _classes(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["true_class"]) for row in rows} | {str(row["pred_class"]) for row in rows})


def _model_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in sorted({row["model"] for row in rows}):
        group = [row for row in rows if row["model"] == model]
        total = len(group)
        correct = sum(1 for row in group if row["correct"])
        class_recalls = []
        f1_values = []
        weighted_f1_parts = []
        for cls in _classes(group):
            tp = sum(1 for row in group if row["true_class"] == cls and row["pred_class"] == cls)
            fp = sum(1 for row in group if row["true_class"] != cls and row["pred_class"] == cls)
            fn = sum(1 for row in group if row["true_class"] == cls and row["pred_class"] != cls)
            support = tp + fn
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            if support:
                class_recalls.append(recall)
                f1_values.append(f1)
                weighted_f1_parts.append(f1 * support)
        out.append(
            {
                "model": model,
                "n": total,
                "accuracy": correct / total if total else 0.0,
                "balanced_accuracy": fmean(class_recalls) if class_recalls else 0.0,
                "macro_f1": fmean(f1_values) if f1_values else 0.0,
                "weighted_f1": sum(weighted_f1_parts) / total if total else 0.0,
                "mean_confidence": fmean([float(row["confidence"]) for row in group]) if group else 0.0,
            }
        )
    return out


def _per_class_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in sorted({row["model"] for row in rows}):
        group = [row for row in rows if row["model"] == model]
        for cls in _classes(group):
            tp = sum(1 for row in group if row["true_class"] == cls and row["pred_class"] == cls)
            fp = sum(1 for row in group if row["true_class"] != cls and row["pred_class"] == cls)
            fn = sum(1 for row in group if row["true_class"] == cls and row["pred_class"] != cls)
            support = tp + fn
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            out.append(
                {
                    "model": model,
                    "class": cls,
                    "support": support,
                    "correct": tp,
                    "wrong": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return out


def _wrong_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in sorted({row["model"] for row in rows}):
        group = [row for row in rows if row["model"] == model]
        for cls in _classes(group):
            subset = [row for row in group if row["true_class"] == cls]
            wrong = [row for row in subset if not row["correct"]]
            transitions = Counter(row["pred_class"] for row in wrong)
            out.append(
                {
                    "model": model,
                    "true_class": cls,
                    "support": len(subset),
                    "correct": len(subset) - len(wrong),
                    "wrong": len(wrong),
                    "error_rate": len(wrong) / len(subset) if subset else 0.0,
                    "wrong_to": ";".join(f"{pred}:{count}" for pred, count in sorted(transitions.items())),
                }
            )
    return out


def _confusion(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in sorted({row["model"] for row in rows}):
        group = [row for row in rows if row["model"] == model]
        counts = Counter((row["true_class"], row["pred_class"]) for row in group)
        for (true_class, pred_class), count in sorted(counts.items()):
            out.append({"model": model, "true_class": true_class, "pred_class": pred_class, "count": count})
    return out


def _outcome_name(correct: dict[str, bool], preds: dict[str, str]) -> str:
    tot = correct.get("tot", False)
    dual = correct.get("dual_concat", False)
    gmu = correct.get("gmu_totstrong", False)
    if tot and dual and gmu:
        return "all_correct"
    if not tot and not dual and not gmu:
        return "all_wrong_same_pred" if len(set(preds.values())) == 1 else "all_wrong_different_pred"
    if tot and not dual and not gmu:
        return "only_tot_correct"
    if dual and not tot and not gmu:
        return "only_dual_concat_correct"
    if gmu and not tot and not dual:
        return "only_gmu_totstrong_correct"
    if tot and dual and not gmu:
        return "tot_dual_concat_correct_gmu_totstrong_wrong"
    if tot and gmu and not dual:
        return "tot_gmu_totstrong_correct_dual_concat_wrong"
    if dual and gmu and not tot:
        return "dual_concat_gmu_totstrong_correct_tot_wrong"
    return "other"


def _sample_outcomes(rows: list[dict[str, Any]], model_order: list[str]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[str(row["sample_key"])][str(row["model"])] = row
    out = []
    for sample_key, model_rows in sorted(by_key.items()):
        if not all(model in model_rows for model in model_order):
            continue
        true_class = str(next(iter(model_rows.values()))["true_class"])
        correct = {model: bool(model_rows[model]["correct"]) for model in model_order}
        preds = {model: str(model_rows[model]["pred_class"]) for model in model_order}
        confidences = {model: float(model_rows[model]["confidence"]) for model in model_order}
        row: dict[str, Any] = {
            "sample_key": sample_key,
            "true_class": true_class,
            "outcome": _outcome_name(correct, preds),
        }
        for model in model_order:
            row[f"{model}_pred"] = preds[model]
            row[f"{model}_correct"] = correct[model]
            row[f"{model}_confidence"] = confidences[model]
            row[f"{model}_fold"] = model_rows[model]["fold"]
        out.append(row)
    return out


def _summarize_outcomes(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(outcomes)
    counts = Counter(row["outcome"] for row in outcomes)
    return [
        {"outcome": outcome, "n": count, "rate": count / total if total else 0.0}
        for outcome, count in sorted(counts.items())
    ]


def _outcomes_by_class(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals = Counter(row["true_class"] for row in outcomes)
    counts = Counter((row["true_class"], row["outcome"]) for row in outcomes)
    return [
        {
            "true_class": cls,
            "outcome": outcome,
            "n": count,
            "rate_within_class": count / totals[cls] if totals[cls] else 0.0,
        }
        for (cls, outcome), count in sorted(counts.items())
    ]


def _pairwise_gain_harm(outcomes: list[dict[str, Any]], pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    for base, candidate in pairs:
        gain = [row for row in outcomes if str(row[f"{base}_correct"]).lower() == "false" and str(row[f"{candidate}_correct"]).lower() == "true"]
        harm = [row for row in outcomes if str(row[f"{base}_correct"]).lower() == "true" and str(row[f"{candidate}_correct"]).lower() == "false"]
        out.append(
            {
                "base": base,
                "candidate": candidate,
                "gain": len(gain),
                "harm": len(harm),
                "net": len(gain) - len(harm),
            }
        )
    return out


def _readme(out_dir: Path, model_specs: dict[str, str], n_rows: int) -> None:
    lines = [
        "# P9a OOF Particle Error Analysis",
        "",
        "This directory contains full-dataset out-of-fold predictions and error overlap diagnostics.",
        "Each sample is evaluated only by fold models that did not train on that sample.",
        "",
        f"Total merged per-model prediction rows: `{n_rows}`.",
        "",
        "## Models",
        "",
    ]
    for label, pattern in model_specs.items():
        lines.append(f"- `{label}`: `{pattern}`")
    lines.extend(
        [
            "",
            "## Important outputs",
            "",
            "- `oof_predictions_long.csv`: all model/fold/sample predictions.",
            "- `wrong_count_by_model_class.csv`: per-model class error counts.",
            "- `three_model_sample_outcomes.csv`: sample-level correctness pattern across ToT, dual-concat, and GMU.",
            "- `all_wrong_same_pred_samples.csv`: high-priority manual-review candidates.",
            "- `cosr_confusion_samples.csv`: Co/Sr boundary-confusion candidates.",
            "- `model_specific_failure_samples.csv`: model-specific failures; diagnostic only, not deletion criteria.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    group_root = Path("outputs") / "experiments" / args.group
    if not group_root.exists():
        raise FileNotFoundError(f"Experiment group not found: {group_root}")
    model_specs = _parse_model_specs(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for label, pattern in model_specs.items():
        model_rows = _load_model_predictions(group_root, label, pattern)
        rows.extend(model_rows)

    model_order = [model for model in DEFAULT_MODEL_ORDER if model in model_specs]
    model_order.extend(model for model in model_specs if model not in model_order)
    outcomes = _sample_outcomes(rows, model_order)

    _write_csv(out_dir / "oof_predictions_long.csv", rows)
    _write_csv(out_dir / "model_metrics.csv", _model_metrics(rows))
    _write_csv(out_dir / "per_class_metrics.csv", _per_class_metrics(rows))
    _write_csv(out_dir / "wrong_count_by_model_class.csv", _wrong_counts(rows))
    _write_csv(out_dir / "confusion_by_model.csv", _confusion(rows))
    _write_csv(out_dir / "error_transitions_by_model.csv", [row for row in _confusion(rows) if row["true_class"] != row["pred_class"]])

    _write_csv(out_dir / "three_model_sample_outcomes.csv", outcomes)
    _write_csv(out_dir / "three_model_outcomes_summary.csv", _summarize_outcomes(outcomes))
    _write_csv(out_dir / "three_model_outcomes_by_class.csv", _outcomes_by_class(outcomes))
    _write_csv(
        out_dir / "pairwise_gain_harm_summary.csv",
        _pairwise_gain_harm(outcomes, [("tot", "dual_concat"), ("tot", "gmu_totstrong"), ("dual_concat", "gmu_totstrong")]),
    )

    all_wrong_same = [row for row in outcomes if row["outcome"] == "all_wrong_same_pred"]
    all_wrong_diff = [row for row in outcomes if row["outcome"] == "all_wrong_different_pred"]
    cosr_confusions = [
        row
        for row in outcomes
        if row["true_class"] in {"Co", "Sr"}
        and any(row.get(f"{model}_pred") in {"Co", "Sr"} and row.get(f"{model}_pred") != row["true_class"] for model in model_order)
    ]
    model_specific = [
        row
        for row in outcomes
        if row["outcome"]
        in {
            "only_tot_correct",
            "only_dual_concat_correct",
            "only_gmu_totstrong_correct",
            "tot_dual_concat_correct_gmu_totstrong_wrong",
            "tot_gmu_totstrong_correct_dual_concat_wrong",
            "dual_concat_gmu_totstrong_correct_tot_wrong",
        }
    ]
    _write_csv(out_dir / "all_wrong_same_pred_samples.csv", all_wrong_same)
    _write_csv(out_dir / "all_wrong_different_pred_samples.csv", all_wrong_diff)
    _write_csv(out_dir / "cosr_confusion_samples.csv", cosr_confusions)
    _write_csv(out_dir / "model_specific_failure_samples.csv", model_specific)
    _readme(out_dir, model_specs, len(rows))

    print(f"Wrote OOF analysis to {out_dir}")
    print(f"Prediction rows: {len(rows)}")
    print(f"Samples with all models: {len(outcomes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
