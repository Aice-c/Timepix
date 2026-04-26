#!/usr/bin/env python
"""Run one Timepix experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timepix.config import load_experiment_config, parse_override, set_by_dotted_key
from timepix.training.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Timepix experiment")
    parser.add_argument("--config", required=True, help="Experiment YAML file")
    parser.add_argument("--data-root", default=None, help="Override dataset.root for this machine")
    parser.add_argument("--output-root", default=None, help="Override output root")
    parser.add_argument("--name", default=None, help="Override experiment name")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set training.epochs=5",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item}")
        key, value = item.split("=", 1)
        set_by_dotted_key(cfg, key, parse_override(value))

    metadata = run_experiment(
        cfg,
        output_root=args.output_root,
        data_root_override=args.data_root,
        experiment_name=args.name,
    )
    print(f"Experiment finished: {metadata['experiment_dir']}")
    metrics = metadata["metrics"]
    print(f"Best epoch: {metrics['best_epoch']}")
    if metadata["task"] == "classification":
        print(f"Val accuracy: {metrics['validation'].get('accuracy', 0):.4f}")
        print(f"Test accuracy: {metrics['test'].get('accuracy', 0):.4f}")
        print(f"Test MAE(argmax): {metrics['test'].get('mae_argmax', 0):.3f}")
    else:
        print(f"Val MAE: {metrics['validation'].get('mae', 0):.3f}")
        print(f"Test MAE: {metrics['test'].get('mae', 0):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

