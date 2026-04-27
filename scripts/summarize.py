#!/usr/bin/env python
"""Summarize Timepix experiment metadata into a CSV table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment outputs")
    parser.add_argument("--root", default="outputs/experiments", help="Experiment output root")
    parser.add_argument("--out", default="outputs/experiment_summary.csv", help="Output CSV path")
    return parser.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    rows = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        metadata = _load_json(metadata_path)
        dataset = metadata.get("dataset", {})
        metrics = metadata.get("metrics", {})
        val = metrics.get("validation", {})
        test = metrics.get("test", {})
        model = metadata.get("model", {})
        loss = metadata.get("loss", {})
        rows.append(
            {
                "experiment_name": metadata.get("experiment_name"),
                "experiment_dir": metadata.get("experiment_dir"),
                "dataset": dataset.get("name"),
                "particle": dataset.get("particle"),
                "modalities": "+".join(dataset.get("modalities", [])),
                "task": metadata.get("task"),
                "model": model.get("name"),
                "fusion_mode": model.get("fusion_mode"),
                "loss": loss.get("name"),
                "label_encoding": loss.get("label_encoding"),
                "best_epoch": metrics.get("best_epoch"),
                "val_accuracy": val.get("accuracy"),
                "test_accuracy": test.get("accuracy"),
                "val_mae_argmax": val.get("mae_argmax", val.get("mae")),
                "test_mae_argmax": test.get("mae_argmax", test.get("mae")),
                "val_p90_error": val.get("p90_error"),
                "test_p90_error": test.get("p90_error"),
                "test_macro_f1": test.get("macro_f1"),
                "params_total": metadata.get("param_count", {}).get("total"),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "experiment_name",
        "experiment_dir",
        "dataset",
        "particle",
        "modalities",
        "task",
        "model",
        "fusion_mode",
        "loss",
        "label_encoding",
        "best_epoch",
        "val_accuracy",
        "test_accuracy",
        "val_mae_argmax",
        "test_mae_argmax",
        "val_p90_error",
        "test_p90_error",
        "test_macro_f1",
        "params_total",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
