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


DEFAULT_EXPERIMENT_ROOT = Path("outputs/experiments")


SUMMARY_FIELDS = [
    "experiment_group",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment outputs")
    parser.add_argument("--root", default=None, help="Experiment output root or group directory")
    parser.add_argument("--out", default=None, help="Output CSV path")
    parser.add_argument("--group", default=None, help="Summarize outputs/experiments/<group>")
    parser.add_argument("--all", action="store_true", help="Recursively summarize all experiment groups")
    return parser.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _metadata_paths(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/metadata.json" if recursive else "*/metadata.json"
    return sorted(root.glob(pattern))


def _infer_group(metadata: dict, metadata_path: Path, root: Path, recursive: bool) -> str:
    group = metadata.get("experiment_group")
    if group:
        return str(group)
    if recursive:
        try:
            relative = metadata_path.relative_to(root)
            if len(relative.parts) >= 3:
                return relative.parts[0]
        except ValueError:
            pass
    return "default"


def _row_from_metadata(metadata: dict, metadata_path: Path, root: Path, recursive: bool) -> dict:
    dataset = metadata.get("dataset", {})
    metrics = metadata.get("metrics", {})
    val = metrics.get("validation", {})
    test = metrics.get("test", {})
    model = metadata.get("model", {})
    loss = metadata.get("loss", {})
    return {
        "experiment_group": _infer_group(metadata, metadata_path, root, recursive),
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


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, bool]:
    if args.group and args.all:
        raise ValueError("--group and --all cannot be used together")

    if args.group:
        root = DEFAULT_EXPERIMENT_ROOT / args.group
        out = Path(args.out) if args.out else Path("outputs") / f"{args.group}_summary.csv"
        return root, out, False

    if args.all:
        root = DEFAULT_EXPERIMENT_ROOT
        out = Path(args.out) if args.out else Path("outputs/experiment_summary.csv")
        return root, out, True

    if args.root:
        root = Path(args.root)
        out = Path(args.out) if args.out else Path("outputs/experiment_summary.csv")
        return root, out, False

    root = DEFAULT_EXPERIMENT_ROOT
    out = Path(args.out) if args.out else Path("outputs/experiment_summary.csv")
    return root, out, True


def main() -> int:
    args = parse_args()
    root, out, recursive = _resolve_inputs(args)
    rows = [
        _row_from_metadata(_load_json(metadata_path), metadata_path, root, recursive)
        for metadata_path in _metadata_paths(root, recursive)
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
