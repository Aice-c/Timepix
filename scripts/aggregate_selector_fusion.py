#!/usr/bin/env python
"""Aggregate A4b selector-fusion summaries across seeds.

`evaluate_selector_fusion.py` writes one summary CSV per seed. This helper keeps
the stable comparison rows (`primary_only`, `candidate_only`, `oracle`) and the
validation-selected selector row for each seed, then reports mean/std metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


DEFAULT_METRICS = [
    "val_accuracy",
    "test_accuracy",
    "val_mae_argmax",
    "test_mae_argmax",
    "val_p90_error",
    "test_p90_error",
    "val_macro_f1",
    "test_macro_f1",
    "val_selection_rate",
    "test_selection_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate A4b selector-fusion summaries")
    parser.add_argument("--inputs", nargs="+", required=True, help="Selector summary CSV files")
    parser.add_argument("--out", required=True, help="Output aggregate CSV path")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated numeric fields to aggregate",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10g}"


def _seed_key(value: str) -> tuple[int, str]:
    try:
        return int(value), ""
    except ValueError:
        return 10**9, value


def _method_name(row: dict[str, str]) -> str | None:
    strategy = str(row.get("strategy", ""))
    if strategy in {"primary_only", "candidate_only", "oracle"}:
        return strategy
    if _parse_bool(row.get("selected_by_val")):
        mode = row.get("selector_mode") or "selector"
        fit = row.get("selector_fit") or ""
        return f"validation_selected_{mode}_{fit}".rstrip("_")
    return None


def _selected_strategy(row: dict[str, str]) -> str:
    strategy = str(row.get("strategy", ""))
    threshold = row.get("threshold")
    if threshold not in {None, ""}:
        return f"{strategy}:{threshold}"
    return strategy


def main() -> int:
    args = parse_args()
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for raw_path in args.inputs:
        path = Path(raw_path)
        for row in _read_rows(path):
            method = _method_name(row)
            if method is None:
                continue
            row = dict(row)
            row["_source_file"] = str(path)
            row["_method"] = method
            row["_selected_strategy"] = _selected_strategy(row)
            grouped[method].append(row)

    out_fields = ["method", "n_runs", "seeds", "selected_strategies", "source_files"]
    for metric in metrics:
        out_fields.extend([f"{metric}_mean", f"{metric}_std"])

    out_rows: list[dict[str, str]] = []
    for method in sorted(grouped):
        rows = grouped[method]
        out_row = {
            "method": method,
            "n_runs": str(len(rows)),
            "seeds": ";".join(sorted({row.get("seed", "") for row in rows}, key=_seed_key)),
            "selected_strategies": ";".join(sorted({row.get("_selected_strategy", "") for row in rows})),
            "source_files": ";".join(sorted({row.get("_source_file", "") for row in rows})),
        }
        for metric in metrics:
            values = [_parse_float(row.get(metric)) for row in rows]
            values = [value for value in values if value is not None]
            if values:
                mean_value = fmean(values)
                std_value = stdev(values) if len(values) > 1 else 0.0
            else:
                mean_value = None
                std_value = None
            out_row[f"{metric}_mean"] = _format_number(mean_value)
            out_row[f"{metric}_std"] = _format_number(std_value)
        out_rows.append(out_row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} aggregate rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
