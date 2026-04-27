#!/usr/bin/env python
"""Run a grid of Timepix experiments from one YAML file."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timepix.config import load_experiment_config, resolve_project_path, set_by_dotted_key
from timepix.config_validation import validate_experiment_config, validate_grid_mapping
from timepix.utils.paths import slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a grid of Timepix experiments")
    parser.add_argument("--config", required=True, help="Grid YAML file")
    parser.add_argument("--data-root", default=None, help="Override dataset.root for this machine")
    parser.add_argument("--output-root", default=None, help="Override output root")
    parser.add_argument("--dry-run", action="store_true", help="Print planned experiments only")
    parser.add_argument("--skip-existing", action="store_true", help="Skip experiments whose metadata already exists")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running later grid items after a failure")
    parser.add_argument("--manifest", default=None, help="CSV path for grid run status records")
    return parser.parse_args()


def _short_value(value: Any) -> str:
    if isinstance(value, dict):
        return "_".join(f"{k}-{_short_value(v)}" for k, v in value.items())
    if isinstance(value, list):
        return "-".join(str(v) for v in value)
    return str(value)


def _default_manifest_path(config_path: str | None) -> Path:
    stem = Path(config_path or "grid").stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "grid_manifests" / f"{stem}_{timestamp}.csv"


def _write_manifest(path: Path, rows: list[dict[str, Any]], grid_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["index", "experiment_name", "status", "experiment_dir", "error", *grid_keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _experiment_group_root(cfg: dict, output_root: str | None) -> Path:
    root = output_root or cfg.get("output", {}).get("root", "outputs/experiments")
    group = slugify(str(cfg.get("experiment_group") or "default"))
    return resolve_project_path(root) / group


def _existing_experiments(group_root: Path) -> dict[str, str]:
    existing: dict[str, str] = {}
    for metadata_path in sorted(group_root.glob("*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        name = metadata.get("experiment_name")
        exp_dir = metadata.get("experiment_dir") or str(metadata_path.parent)
        if name:
            existing[str(name)] = str(exp_dir)
    return existing


def main() -> int:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    grid = validate_grid_mapping(cfg.pop("grid", None))

    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    base_name = cfg.get("experiment_name", "grid_experiment")
    group_root = _experiment_group_root(cfg, args.output_root)
    existing = _existing_experiments(group_root) if args.skip_existing else {}
    manifest_path = Path(args.manifest) if args.manifest else _default_manifest_path(cfg.get("_config_path"))
    manifest_rows: list[dict[str, Any]] = []

    combos = list(itertools.product(*values))
    print(f"Planned experiments: {len(combos)}")
    for idx, combo in enumerate(combos, start=1):
        run_cfg = copy.deepcopy(cfg)
        parts = []
        grid_values: dict[str, Any] = {}
        for key, value in zip(keys, combo):
            set_by_dotted_key(run_cfg, key, value)
            grid_values[key] = value
            parts.append(f"{key}-{_short_value(value)}")
        name = f"{base_name}_{idx:03d}_{slugify('_'.join(parts))}"
        run_cfg["experiment_name"] = name
        validate_experiment_config(run_cfg)
        if args.dry_run:
            print(name)
            continue
        row = {
            "index": idx,
            "experiment_name": name,
            "status": "planned",
            "experiment_dir": "",
            "error": "",
            **{key: _short_value(value) for key, value in grid_values.items()},
        }
        if name in existing:
            row["status"] = "skipped_existing"
            row["experiment_dir"] = existing[name]
            manifest_rows.append(row)
            _write_manifest(manifest_path, manifest_rows, keys)
            print(f"[{idx}/{len(combos)}] skipped existing: {existing[name]}")
            continue
        row["status"] = "running"
        manifest_rows.append(row)
        _write_manifest(manifest_path, manifest_rows, keys)
        from timepix.training.runner import run_experiment

        try:
            metadata = run_experiment(
                run_cfg,
                output_root=args.output_root,
                data_root_override=args.data_root,
                experiment_name=name,
            )
            row["status"] = "done"
            row["experiment_dir"] = metadata["experiment_dir"]
            print(f"[{idx}/{len(combos)}] finished: {metadata['experiment_dir']}")
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            _write_manifest(manifest_path, manifest_rows, keys)
            print(f"[{idx}/{len(combos)}] failed: {row['error']}")
            if not args.continue_on_error:
                raise
        _write_manifest(manifest_path, manifest_rows, keys)
    if not args.dry_run:
        print(f"Grid manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

