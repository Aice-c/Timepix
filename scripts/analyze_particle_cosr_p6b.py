#!/usr/bin/env python
"""P6b diagnostics for Co/Sr refinement and pair-logit thresholds.

The script consumes prediction CSVs produced by ``dump_run_predictions.py``.
It selects any Co/Sr threshold on validation only, then reports test behavior
for the selected threshold without using test for selection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


Rows = list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P6b Co/Sr post-hoc diagnostics")
    parser.add_argument("--prediction-dir", required=True, help="Directory from dump_run_predictions.py")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--main-label", default="gmu_totstrong", help="Main model label used in dumped CSV names")
    parser.add_argument("--refiner-label", default="cosr_refiner", help="Refiner label used in dumped CSV names")
    parser.add_argument("--seed", type=int, default=42, help="Seed suffix to load")
    parser.add_argument("--pair", nargs=2, default=["Co", "Sr"], help="Pair classes, default: Co Sr")
    parser.add_argument(
        "--tau-grid",
        default="-2,-1.5,-1,-0.75,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,0.75,1,1.5,2",
        help="Comma-separated logit-difference thresholds for Sr-vs-Co",
    )
    parser.add_argument(
        "--max-val-acc-drop",
        type=float,
        default=0.001,
        help="Allowed validation accuracy drop for threshold selection",
    )
    return parser.parse_args()


def _read_predictions(prediction_dir: Path, label: str, seed: int, split: str) -> Rows:
    path = prediction_dir / f"{label}_seed{seed}_{split}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction dump: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _class_names(rows: Rows) -> list[str]:
    if not rows:
        raise ValueError("Prediction CSV is empty")
    names = [col.removeprefix("prob_") for col in rows[0] if col.startswith("prob_")]
    if not names:
        raise ValueError("Prediction CSV does not contain prob_* columns")
    return names


def _int_array(rows: Rows, column: str) -> np.ndarray:
    return np.array([int(row[column]) for row in rows], dtype=int)


def _float_array(rows: Rows, column: str) -> np.ndarray:
    return np.array([float(row[column]) for row in rows], dtype=float)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], pair: tuple[str, str]) -> dict[str, Any]:
    num_classes = len(class_names)
    correct = y_true == y_pred
    acc = float(correct.mean()) if y_true.size else 0.0
    recalls: list[float] = []
    f1s: list[float] = []
    precisions: list[float] = []
    for cls in range(num_classes):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))

    pair_ids = [class_names.index(pair[0]), class_names.index(pair[1])]
    pair_mask = np.isin(y_true, pair_ids)
    if pair_mask.any():
        pair_acc = float((y_true[pair_mask] == y_pred[pair_mask]).mean())
        pair_recalls = [recalls[pair_ids[0]], recalls[pair_ids[1]]]
        pair_f1s = [f1s[pair_ids[0]], f1s[pair_ids[1]]]
    else:
        pair_acc = 0.0
        pair_recalls = [0.0, 0.0]
        pair_f1s = [0.0, 0.0]

    out: dict[str, Any] = {
        "n": int(y_true.size),
        "accuracy": acc,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "pair_accuracy": pair_acc,
        "pair_balanced_accuracy": float(np.mean(pair_recalls)),
        "pair_macro_f1": float(np.mean(pair_f1s)),
    }
    for cls_idx, name in enumerate(class_names):
        out[f"precision_{name}"] = precisions[cls_idx]
        out[f"recall_{name}"] = recalls[cls_idx]
        out[f"f1_{name}"] = f1s[cls_idx]
    return out


def _top2_pair_mask(rows: Rows, class_names: list[str], pair: tuple[str, str]) -> np.ndarray:
    probs = np.stack([_float_array(rows, f"prob_{name}") for name in class_names], axis=1)
    top2 = np.argsort(probs, axis=1)[:, -2:]
    pair_ids = {class_names.index(pair[0]), class_names.index(pair[1])}
    return np.array([set(row.tolist()) == pair_ids for row in top2], dtype=bool)


def _pair_score(rows: Rows, pair: tuple[str, str]) -> np.ndarray:
    co, sr = pair
    return _float_array(rows, f"logit_{sr}") - _float_array(rows, f"logit_{co}")


def _threshold_preds(rows: Rows, class_names: list[str], pair: tuple[str, str], tau: float) -> np.ndarray:
    pred = _int_array(rows, "pred_label").copy()
    mask = _top2_pair_mask(rows, class_names, pair)
    score = _pair_score(rows, pair)
    co_id = class_names.index(pair[0])
    sr_id = class_names.index(pair[1])
    pred[mask] = np.where(score[mask] > tau, sr_id, co_id)
    return pred


def _threshold_sweep(rows_in: Rows, class_names: list[str], pair: tuple[str, str], taus: list[float]) -> list[dict[str, Any]]:
    y_true = _int_array(rows_in, "true_label")
    base_pred = _int_array(rows_in, "pred_label")
    pair_mask = _top2_pair_mask(rows_in, class_names, pair)
    rows: list[dict[str, Any]] = []
    for tau in taus:
        pred = _threshold_preds(rows_in, class_names, pair, tau)
        changed = pred != base_pred
        row = {
            "tau": float(tau),
            "trigger_rate": float(pair_mask.mean()) if pair_mask.size else 0.0,
            "changed_count": int(changed.sum()),
            "changed_rate": float(changed.mean()) if changed.size else 0.0,
        }
        row.update(_metrics(y_true, pred, class_names, pair))
        rows.append(row)
    return rows


def _select_tau(rows: list[dict[str, Any]], base_metrics: dict[str, Any], max_acc_drop: float) -> dict[str, Any]:
    min_acc = float(base_metrics["accuracy"]) - max_acc_drop
    eligible = [row for row in rows if float(row["accuracy"]) >= min_acc]
    if not eligible:
        eligible = rows
    return max(
        eligible,
        key=lambda row: (
            float(row["macro_f1"]),
            float(row["balanced_accuracy"]),
            float(row["pair_macro_f1"]),
            float(row["accuracy"]),
            -abs(float(row["tau"])),
        ),
    )


def _overlap_rows(
    main_rows: Rows,
    refiner_rows: Rows,
    class_names: list[str],
    pair: tuple[str, str],
    split: str,
    main_label: str,
    refiner_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_ids = {class_names.index(pair[0]), class_names.index(pair[1])}
    main_pair = [row for row in main_rows if int(row["true_label"]) in pair_ids]
    ref_by_key = {row["sample_key"]: row for row in refiner_rows}
    merged = [(row, ref_by_key[row["sample_key"]]) for row in main_pair if row["sample_key"] in ref_by_key]
    if len(merged) != len(main_pair):
        raise ValueError(f"Refiner/main sample alignment mismatch for split={split}: {len(merged)} vs {len(main_pair)}")

    main_correct = np.array([int(main["pred_label"]) == int(main["true_label"]) for main, _ in merged], dtype=bool)
    ref_correct = np.array([int(ref["pred_label"]) == int(ref["true_label"]) for _, ref in merged], dtype=bool)
    both_correct = int((main_correct & ref_correct).sum())
    gain = int((~main_correct & ref_correct).sum())
    harm = int((main_correct & ~ref_correct).sum())
    both_wrong = int((~main_correct & ~ref_correct).sum())
    n = int(len(merged))
    summary = {
        "split": split,
        "main_label": main_label,
        "refiner_label": refiner_label,
        "n": n,
        "both_correct": both_correct,
        "main_wrong_refiner_correct_gain": gain,
        "main_correct_refiner_wrong_harm": harm,
        "both_wrong": both_wrong,
        "net_gain_minus_harm": gain - harm,
        "gain_rate": gain / n if n else 0.0,
        "harm_rate": harm / n if n else 0.0,
        "main_accuracy": float(main_correct.mean()) if n else 0.0,
        "refiner_accuracy": float(ref_correct.mean()) if n else 0.0,
    }

    sample_rows: list[dict[str, Any]] = []
    for main, ref in merged:
        main_ok = int(main["pred_label"]) == int(main["true_label"])
        ref_ok = int(ref["pred_label"]) == int(ref["true_label"])
        if main_ok and ref_ok:
            status = "both_correct"
        elif (not main_ok) and ref_ok:
            status = "gain"
        elif main_ok and (not ref_ok):
            status = "harm"
        else:
            status = "both_wrong"
        sample_rows.append(
            {
                "split": split,
                "status": status,
                "sample_key": main["sample_key"],
                "true_class": main["true_class"],
                "main_pred": main["pred_class"],
                "refiner_pred": ref["pred_class"],
            }
        )
    return summary, sample_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _baseline_row(rows: Rows, class_names: list[str], pair: tuple[str, str], split: str) -> dict[str, Any]:
    row = {"split": split, "strategy": "base", "tau": ""}
    row.update(_metrics(_int_array(rows, "true_label"), _int_array(rows, "pred_label"), class_names, pair))
    return row


def _report_md(
    out_dir: Path,
    selected: dict[str, Any],
    base_val: dict[str, Any],
    base_test: dict[str, Any],
    selected_test: dict[str, Any],
    overlap: list[dict[str, Any]],
) -> None:
    lines = [
        "# P6b Co/Sr Diagnostics",
        "",
        "Validation-only threshold selection was used. Test metrics are reported after applying the selected validation threshold.",
        "",
        "## Selected Threshold",
        "",
        f"- tau: `{float(selected['tau']):.6g}`",
        f"- Val macro-F1: `{float(selected['macro_f1']):.6f}`",
        f"- Val accuracy: `{float(selected['accuracy']):.6f}`",
        "",
        "## Base vs Selected Test",
        "",
        "| Strategy | Test Acc | Test Macro-F1 | Co Recall | Sr Recall | Pair Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| base | {float(base_test['accuracy']):.4f} | {float(base_test['macro_f1']):.4f} | "
            f"{float(base_test.get('recall_Co', 0.0)):.4f} | {float(base_test.get('recall_Sr', 0.0)):.4f} | "
            f"{float(base_test['pair_macro_f1']):.4f} |"
        ),
        (
            f"| threshold | {float(selected_test['accuracy']):.4f} | {float(selected_test['macro_f1']):.4f} | "
            f"{float(selected_test.get('recall_Co', 0.0)):.4f} | {float(selected_test.get('recall_Sr', 0.0)):.4f} | "
            f"{float(selected_test['pair_macro_f1']):.4f} |"
        ),
        "",
        "## Overlap Summary",
        "",
        "| Split | n | main acc | refiner acc | gain | harm | net |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overlap:
        lines.append(
            f"| {row['split']} | {row['n']} | {float(row['main_accuracy']):.4f} | "
            f"{float(row['refiner_accuracy']):.4f} | {row['main_wrong_refiner_correct_gain']} | "
            f"{row['main_correct_refiner_wrong_harm']} | {row['net_gain_minus_harm']} |"
        )
    lines.extend(
        [
            "",
            "## Validation Baseline",
            "",
            f"- Base Val Acc: `{float(base_val['accuracy']):.6f}`",
            f"- Base Val Macro-F1: `{float(base_val['macro_f1']):.6f}`",
        ]
    )
    (out_dir / "p6b_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    prediction_dir = Path(args.prediction_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair = (str(args.pair[0]), str(args.pair[1]))
    taus = [float(item.strip()) for item in args.tau_grid.split(",") if item.strip()]

    main_val = _read_predictions(prediction_dir, args.main_label, args.seed, "val")
    main_test = _read_predictions(prediction_dir, args.main_label, args.seed, "test")
    ref_val = _read_predictions(prediction_dir, args.refiner_label, args.seed, "val")
    ref_test = _read_predictions(prediction_dir, args.refiner_label, args.seed, "test")
    class_names = _class_names(main_val)
    if pair[0] not in class_names or pair[1] not in class_names:
        raise ValueError(f"Pair classes {pair} are not both present in {class_names}")

    base_val = _metrics(_int_array(main_val, "true_label"), _int_array(main_val, "pred_label"), class_names, pair)
    base_test = _metrics(_int_array(main_test, "true_label"), _int_array(main_test, "pred_label"), class_names, pair)
    val_rows = _threshold_sweep(main_val, class_names, pair, taus)
    selected = _select_tau(val_rows, base_val, float(args.max_val_acc_drop))
    selected_tau = float(selected["tau"])

    test_pred = _threshold_preds(main_test, class_names, pair, selected_tau)
    selected_test = _metrics(_int_array(main_test, "true_label"), test_pred, class_names, pair)
    selected_test_row = {"split": "test", "strategy": "validation_selected_threshold", "tau": selected_tau}
    selected_test_row.update(selected_test)
    selected_val_row = {"split": "val", "strategy": "validation_selected_threshold", **selected}

    baseline_rows = [
        _baseline_row(main_val, class_names, pair, "val"),
        _baseline_row(main_test, class_names, pair, "test"),
    ]

    overlap: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for split, main_rows, ref_rows in [("val", main_val, ref_val), ("test", main_test, ref_test)]:
        summary, samples = _overlap_rows(main_rows, ref_rows, class_names, pair, split, args.main_label, args.refiner_label)
        overlap.append(summary)
        sample_rows.extend(samples)

    changes: list[dict[str, Any]] = []
    for split, rows, pred in [
        ("val", main_val, _threshold_preds(main_val, class_names, pair, selected_tau)),
        ("test", main_test, test_pred),
    ]:
        base_pred = _int_array(rows, "pred_label")
        changed = np.flatnonzero(pred != base_pred)
        for idx in changed:
            row = rows[int(idx)]
            score = float(_pair_score([row], pair)[0])
            changes.append(
                {
                    "split": split,
                    "sample_key": row["sample_key"],
                    "true_class": row["true_class"],
                    "base_pred": row["pred_class"],
                    "threshold_pred": class_names[int(pred[idx])],
                    "score_sr_minus_co": score,
                    "tau": selected_tau,
                }
            )

    _write_csv(out_dir / "p6b_threshold_sweep_val.csv", val_rows)
    _write_csv(out_dir / "p6b_threshold_selected.csv", [selected_val_row, selected_test_row])
    _write_csv(out_dir / "p6b_base_metrics.csv", baseline_rows)
    _write_csv(out_dir / "p6b_refiner_overlap.csv", overlap)
    _write_csv(out_dir / "p6b_refiner_overlap_samples.csv", sample_rows)
    _write_csv(out_dir / "p6b_threshold_changed_samples.csv", changes)
    _report_md(out_dir, selected, base_val, base_test, selected_test, overlap)

    print(f"Selected tau={selected_tau:.6g}; wrote P6b diagnostics to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
