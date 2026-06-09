#!/usr/bin/env python
"""Evaluate a Co/Sr binary refiner against particle source model predictions.

This is a diagnostic evaluator for P6a. It reports conditional metrics on true
Co/Sr test samples using existing 4-class model predictions and binary refiner
predictions. It does not simulate false refiner triggers on non-Co/Sr samples.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


PAIR_CLASSES = ("Co", "Sr")
DEFAULT_MAIN_SPECS = [
    (
        "gmu_totstrong",
        "outputs/experiments/p2c_ptype_stage1_gmm02_p_v3_gmu_base_vs_totstrong_3seed/"
        "*gmu_aux_totstrong*/predictions.csv",
    ),
    (
        "dual_concat",
        "outputs/experiments/p2a_ptype_stage1_gmm02_p_v3_modality_lr3e6_3seed/"
        "*dual_concat*/predictions.csv",
    ),
    (
        "tot_maxpool",
        "outputs/experiments/p4a_ptype_stage1_gmm02_p_v3_tot_backbone_3seed/"
        "*model.name-resnet18_maxpool*/predictions.csv",
    ),
]


@dataclass
class RunPredictions:
    label: str
    seed: int
    path: Path
    class_names: list[str]
    split_path: Path
    rows: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate P6a Co/Sr refiner triggers")
    parser.add_argument(
        "--full-split",
        default="outputs/splits/ptype_stage1_gmm02_p_v3_ToT-ToA_seed42_0.8_0.1_0.1.json",
        help="Full 4-class split manifest used by main models",
    )
    parser.add_argument(
        "--refiner-split",
        default="outputs/splits/ptype_stage1_gmm02_p_v3_Co-Sr_ToT-ToA_seed42_0.8_0.1_0.1.json",
        help="Filtered Co/Sr split manifest used by the refiner",
    )
    parser.add_argument(
        "--refiner-glob",
        default="outputs/experiments/p6a_ptype_stage1_gmm02_p_v3_cosr_refiner_seed42/*/predictions.csv",
        help="Glob for refiner predictions.csv files",
    )
    parser.add_argument(
        "--main-run",
        action="append",
        default=[],
        metavar="LABEL=GLOB",
        help="Main-model predictions glob. Can be repeated. Defaults cover gmu_totstrong, dual_concat, tot_maxpool.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/p6a_ptype_stage1_gmm02_p_v3_cosr_refiner",
        help="Output directory for diagnostics",
    )
    parser.add_argument("--pair", nargs=2, default=list(PAIR_CLASSES), help="Pair classes, default: Co Sr")
    return parser.parse_args()


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _split_seed_from_path(path: Path) -> int:
    text = str(path)
    match = re.search(r"training\.seed-(\d+)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"seed[-_]?(\d+)", text)
    if match:
        return int(match.group(1))
    return 42


def _metrics_path(predictions_path: Path) -> Path:
    return predictions_path.with_name("metrics.json")


def _class_names_from_predictions(rows: list[dict[str, str]]) -> list[str]:
    names = [key.removeprefix("prob_") for key in rows[0] if key.startswith("prob_")]
    if not names:
        names = sorted({row["true_class"] for row in rows} | {row["pred_class"] for row in rows})
    return names


def _load_run(label: str, path: str | Path) -> RunPredictions:
    path = Path(path)
    rows = _read_csv(path)
    class_names = _class_names_from_predictions(rows)
    seed = _split_seed_from_path(path)
    split_path = Path("")
    metrics_path = _metrics_path(path)
    if metrics_path.exists():
        metrics = _read_json(metrics_path)
        seed = int(metrics.get("config", {}).get("training", {}).get("seed", seed))
        split_path = Path(metrics.get("data_info", {}).get("split_path", ""))
    return RunPredictions(label, seed, path, class_names, split_path, rows)


def _expand_specs(specs: list[str]) -> list[tuple[str, str]]:
    if specs:
        parsed: list[tuple[str, str]] = []
        for item in specs:
            if "=" not in item:
                raise ValueError(f"--main-run expects LABEL=GLOB, got {item}")
            label, pattern = item.split("=", 1)
            parsed.append((label, pattern))
        return parsed
    return list(DEFAULT_MAIN_SPECS)


def _discover_runs(specs: list[tuple[str, str]]) -> list[RunPredictions]:
    runs: list[RunPredictions] = []
    for label, pattern in specs:
        for path in sorted(glob.glob(pattern)):
            runs.append(_load_run(label, path))
    if not runs:
        raise FileNotFoundError("No main predictions found")
    return runs


def _discover_refiners(pattern: str) -> list[RunPredictions]:
    runs = [_load_run("cosr_refiner", path) for path in sorted(glob.glob(pattern))]
    if not runs:
        raise FileNotFoundError(f"No refiner predictions found for pattern: {pattern}")
    return runs


def _row_by_key(rows: list[dict[str, str]], split_keys: list[str]) -> dict[str, dict[str, str]]:
    if len(rows) != len(split_keys):
        raise ValueError(f"Prediction rows ({len(rows)}) do not match split keys ({len(split_keys)})")
    return {key: row for key, row in zip(split_keys, rows)}


def _top_classes(row: dict[str, str], class_names: Iterable[str]) -> list[tuple[str, float]]:
    pairs = []
    for name in class_names:
        value = row.get(f"prob_{name}", "")
        try:
            prob = float(value)
        except ValueError:
            prob = 0.0
        pairs.append((name, prob))
    return sorted(pairs, key=lambda item: item[1], reverse=True)


def _binary_metrics(y_true: list[str], y_pred: list[str], pair: tuple[str, str]) -> dict[str, float | int]:
    total = len(y_true)
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    out: dict[str, float | int] = {
        "n": total,
        "accuracy": correct / total if total else 0.0,
    }
    recalls: list[float] = []
    f1s: list[float] = []
    for cls in pair:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(1 for true in y_true if true == cls)
        out[f"{cls}_support"] = support
        out[f"{cls}_precision"] = precision
        out[f"{cls}_recall"] = recall
        out[f"{cls}_f1"] = f1
        recalls.append(recall)
        f1s.append(f1)
    out["balanced_accuracy"] = mean(recalls) if recalls else 0.0
    out["macro_f1"] = mean(f1s) if f1s else 0.0
    out[f"{pair[0]}_to_{pair[1]}"] = sum(
        1 for true, pred in zip(y_true, y_pred) if true == pair[0] and pred == pair[1]
    )
    out[f"{pair[1]}_to_{pair[0]}"] = sum(
        1 for true, pred in zip(y_true, y_pred) if true == pair[1] and pred == pair[0]
    )
    out["pair_to_other"] = sum(1 for true, pred in zip(y_true, y_pred) if true in pair and pred not in pair)
    return out


def _mean_std(rows: list[dict], group_keys: list[str], value_keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[item] for item in group_keys)
        grouped.setdefault(key, []).append(row)
    out: list[dict] = []
    for key, items in sorted(grouped.items()):
        row = {name: value for name, value in zip(group_keys, key)}
        row["n_runs"] = len(items)
        for value_key in value_keys:
            values = [float(item[value_key]) for item in items]
            row[f"{value_key}_mean"] = mean(values)
            row[f"{value_key}_std"] = stdev(values) if len(values) > 1 else 0.0
        out.append(row)
    return out


def _evaluate_trigger(
    main: RunPredictions,
    refiner: RunPredictions,
    full_test_keys: list[str],
    refiner_test_keys: list[str],
    pair: tuple[str, str],
) -> tuple[list[dict], list[dict]]:
    main_by_key = _row_by_key(main.rows, full_test_keys)
    refiner_by_key = _row_by_key(refiner.rows, refiner_test_keys)
    pair_keys = [key for key in refiner_test_keys if key in main_by_key]
    pair_set = set(pair)

    sample_rows: list[dict] = []
    metric_rows: list[dict] = []
    trigger_defs = {
        "base": lambda row, top: False,
        "top1_pair": lambda row, top: bool(top and top[0][0] in pair_set),
        "top2_exact_pair": lambda row, top: len(top) >= 2 and {top[0][0], top[1][0]} == pair_set,
        "always_refine_true_pair": lambda row, top: True,
    }
    predictions: dict[str, list[str]] = {name: [] for name in trigger_defs}
    y_true: list[str] = []

    for key in pair_keys:
        main_row = main_by_key[key]
        ref_row = refiner_by_key[key]
        true_cls = main_row["true_class"]
        main_pred = main_row["pred_class"]
        ref_pred = ref_row["pred_class"]
        top = _top_classes(main_row, main.class_names)
        y_true.append(true_cls)
        sample = {
            "seed": main.seed,
            "main_model": main.label,
            "sample_key": key,
            "true_class": true_cls,
            "main_pred": main_pred,
            "refiner_pred": ref_pred,
            "top1": top[0][0] if top else "",
            "top1_prob": top[0][1] if top else "",
            "top2": top[1][0] if len(top) > 1 else "",
            "top2_prob": top[1][1] if len(top) > 1 else "",
            "base_correct": int(main_pred == true_cls),
            "refiner_correct": int(ref_pred == true_cls),
        }
        for trigger_name, should_refine in trigger_defs.items():
            pred = ref_pred if trigger_name != "base" and should_refine(main_row, top) else main_pred
            predictions[trigger_name].append(pred)
            sample[f"{trigger_name}_pred"] = pred
            sample[f"{trigger_name}_correct"] = int(pred == true_cls)
        sample_rows.append(sample)

    for trigger_name, y_pred in predictions.items():
        metrics = _binary_metrics(y_true, y_pred, pair)
        row = {
            "seed": main.seed,
            "main_model": main.label,
            "refiner_model": refiner.label,
            "trigger": trigger_name,
            "main_predictions": str(main.path),
            "refiner_predictions": str(refiner.path),
        }
        row.update(metrics)
        metric_rows.append(row)
    return metric_rows, sample_rows


def main() -> int:
    args = parse_args()
    pair = (str(args.pair[0]), str(args.pair[1]))
    full_split = _read_json(args.full_split)
    refiner_split = _read_json(args.refiner_split)
    full_test_keys = list(full_split["test"])
    refiner_test_keys = list(refiner_split["test"])

    main_runs = _discover_runs(_expand_specs(args.main_run))
    refiner_runs = _discover_refiners(args.refiner_glob)
    refiner_by_seed = {run.seed: run for run in refiner_runs}

    all_metric_rows: list[dict] = []
    all_sample_rows: list[dict] = []
    for main_run in main_runs:
        refiner = refiner_by_seed.get(main_run.seed)
        if refiner is None:
            continue
        metric_rows, sample_rows = _evaluate_trigger(
            main_run,
            refiner,
            full_test_keys,
            refiner_test_keys,
            pair,
        )
        all_metric_rows.extend(metric_rows)
        all_sample_rows.extend(sample_rows)

    if not all_metric_rows:
        raise RuntimeError("No main/refiner runs share a seed")

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "p6a_cosr_refiner_trigger_metrics.csv", all_metric_rows)
    _write_csv(out_dir / "p6a_cosr_refiner_sample_outcomes.csv", all_sample_rows)
    mean_std = _mean_std(
        all_metric_rows,
        ["main_model", "trigger"],
        ["accuracy", "balanced_accuracy", "macro_f1", f"{pair[0]}_recall", f"{pair[1]}_recall"],
    )
    _write_csv(out_dir / "p6a_cosr_refiner_mean_std.csv", mean_std)

    report_lines = [
        "# P6a Co/Sr Refiner Diagnostic",
        "",
        "Scope: true Co/Sr test subset only. This pilot does not estimate false refiner triggers on Am/P samples.",
        "",
        "## Trigger Mean/Std",
        "",
        "| Main | Trigger | n | Acc | Macro-F1 | Co Recall | Sr Recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mean_std:
        report_lines.append(
            "| {main_model} | {trigger} | {n_runs} | {accuracy_mean:.4f}±{accuracy_std:.4f} | "
            "{macro_f1_mean:.4f}±{macro_f1_std:.4f} | {co:.4f}±{co_std:.4f} | "
            "{sr:.4f}±{sr_std:.4f} |".format(
                main_model=row["main_model"],
                trigger=row["trigger"],
                n_runs=row["n_runs"],
                accuracy_mean=row["accuracy_mean"],
                accuracy_std=row["accuracy_std"],
                macro_f1_mean=row["macro_f1_mean"],
                macro_f1_std=row["macro_f1_std"],
                co=row[f"{pair[0]}_recall_mean"],
                co_std=row[f"{pair[0]}_recall_std"],
                sr=row[f"{pair[1]}_recall_mean"],
                sr_std=row[f"{pair[1]}_recall_std"],
            )
        )
    (out_dir / "p6a_cosr_refiner_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote P6a diagnostics to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
