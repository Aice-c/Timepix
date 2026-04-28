#!/usr/bin/env python
"""Aggregate repeated-seed experiment summaries into mean/std rows."""

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
    "test_macro_f1",
    "best_epoch",
    "stopped_epoch",
    "fit_seconds",
    "test_seconds",
    "total_seconds",
]

RUN_SPECIFIC_FIELDS = {
    "experiment_name",
    "experiment_dir",
    "seed",
    "git_commit",
    "git_dirty",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate seed repeats from a summary CSV")
    parser.add_argument("--summary", required=True, help="Input CSV from scripts/summarize.py")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--group-by",
        default=None,
        help="Comma-separated grouping fields. Defaults to stable config fields in the summary.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated numeric fields to aggregate.",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _default_group_fields(header: list[str], metrics: list[str]) -> list[str]:
    metric_set = set(metrics)
    return [
        field
        for field in header
        if field not in RUN_SPECIFIC_FIELDS
        and field not in metric_set
        and not field.endswith("_seconds")
        and field not in {"best_epoch", "stopped_epoch", "early_stopped"}
    ]


def _sort_seed_values(values: set[str]) -> list[str]:
    def key(value: str):
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted((value for value in values if value != ""), key=key)


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10g}"


def main() -> int:
    args = parse_args()
    rows, header = _read_rows(Path(args.summary))
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    if args.group_by:
        group_fields = [item.strip() for item in args.group_by.split(",") if item.strip()]
    else:
        group_fields = _default_group_fields(header, metrics)

    missing = [field for field in group_fields + metrics if field not in header]
    if missing:
        raise ValueError(f"Input summary is missing columns: {missing}")

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        groups[key].append(row)

    out_fields = list(group_fields) + ["n_runs", "seeds"]
    for metric in metrics:
        out_fields.extend([f"{metric}_mean", f"{metric}_std"])

    out_rows: list[dict[str, str]] = []
    for key, group_rows in sorted(groups.items()):
        out_row = {field: value for field, value in zip(group_fields, key)}
        out_row["n_runs"] = str(len(group_rows))
        out_row["seeds"] = ";".join(_sort_seed_values({row.get("seed", "") for row in group_rows}))
        for metric in metrics:
            values = [_parse_float(row.get(metric)) for row in group_rows]
            values = [value for value in values if value is not None]
            if not values:
                mean_value = None
                std_value = None
            else:
                mean_value = fmean(values)
                std_value = stdev(values) if len(values) > 1 else 0.0
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
