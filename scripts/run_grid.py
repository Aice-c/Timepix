#!/usr/bin/env python
"""Run a grid of Timepix experiments from one YAML file."""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timepix.config import load_experiment_config, set_by_dotted_key
from timepix.training.runner import run_experiment
from timepix.utils.paths import slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a grid of Timepix experiments")
    parser.add_argument("--config", required=True, help="Grid YAML file")
    parser.add_argument("--data-root", default=None, help="Override dataset.root for this machine")
    parser.add_argument("--output-root", default=None, help="Override output root")
    parser.add_argument("--dry-run", action="store_true", help="Print planned experiments only")
    return parser.parse_args()


def _short_value(value: Any) -> str:
    if isinstance(value, dict):
        return "_".join(f"{k}-{_short_value(v)}" for k, v in value.items())
    if isinstance(value, list):
        return "-".join(str(v) for v in value)
    return str(value)


def main() -> int:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    grid = cfg.pop("grid", None)
    if not grid:
        raise ValueError("Grid config must contain a 'grid' mapping")

    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    base_name = cfg.get("experiment_name", "grid_experiment")

    combos = list(itertools.product(*values))
    print(f"Planned experiments: {len(combos)}")
    for idx, combo in enumerate(combos, start=1):
        run_cfg = copy.deepcopy(cfg)
        parts = []
        for key, value in zip(keys, combo):
            set_by_dotted_key(run_cfg, key, value)
            parts.append(f"{key}-{_short_value(value)}")
        name = f"{base_name}_{idx:03d}_{slugify('_'.join(parts))}"
        run_cfg["experiment_name"] = name
        if args.dry_run:
            print(name)
            continue
        metadata = run_experiment(
            run_cfg,
            output_root=args.output_root,
            data_root_override=args.data_root,
            experiment_name=name,
        )
        print(f"[{idx}/{len(combos)}] finished: {metadata['experiment_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

