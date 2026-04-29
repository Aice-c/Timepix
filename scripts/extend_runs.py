#!/usr/bin/env python
"""Extend completed experiment runs from their last checkpoints.

This is intended for cases where a grid was run with too small an epoch budget
and the existing `last_checkpoint.pth` files should be continued to a larger
`training.epochs` value. Runs can be resumed in place or copied into a new
experiment group before resuming.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timepix.config import load_yaml, parse_override, resolve_project_path, set_by_dotted_key
from timepix.config_validation import validate_experiment_config
from timepix.training.logger import write_yaml
from timepix.utils.paths import slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend existing runs to a larger epoch budget")
    parser.add_argument("--source-group", required=True, help="Existing experiment group to extend")
    parser.add_argument("--target-epochs", type=int, required=True, help="New training.epochs value")
    parser.add_argument("--output-root", default="outputs/experiments", help="Experiment output root")
    parser.add_argument(
        "--target-group",
        default=None,
        help="Optional new group to copy runs into before resuming. Omit to resume in place.",
    )
    parser.add_argument("--data-root", default=None, help="Override dataset.root for this machine")
    parser.add_argument("--dry-run", action="store_true", help="Print planned resume actions only")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue later runs after a failure")
    parser.add_argument("--skip-completed", action="store_true", help="Skip runs already at or beyond target epochs")
    parser.add_argument(
        "--skip-early-stopped",
        action="store_true",
        help="Skip source runs whose metrics.json says early_stopped=true.",
    )
    parser.add_argument(
        "--resume-target-existing",
        action="store_true",
        help="If target run directory already exists, resume it instead of skipping.",
    )
    parser.add_argument(
        "--reset-patience",
        action="store_true",
        help="Reset checkpoint patience_counter to 0 before extending.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="CSV path for extension status records. Defaults to outputs/grid_manifests/...",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value before resuming, e.g. --set training.early_stopping_patience=5",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _atomic_torch_save(obj: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def _manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest:
        return Path(args.manifest)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source = slugify(args.source_group)
    suffix = f"to_ep{args.target_epochs}"
    return Path("outputs") / "grid_manifests" / f"extend_{source}_{suffix}_{timestamp}.csv"


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "source_run",
        "target_run",
        "status",
        "start_epoch",
        "target_epochs",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _source_runs(output_root: Path, source_group: str) -> list[Path]:
    group_root = output_root / slugify(source_group)
    if not group_root.is_dir():
        raise FileNotFoundError(f"Source group does not exist: {group_root}")
    runs = [path.parent for path in sorted(group_root.glob("*/metadata.json"))]
    if not runs:
        raise RuntimeError(f"No runs with metadata.json found in {group_root}")
    return runs


def _derive_target_name(source_run: Path, source_group: str, target_group: str, metadata: dict[str, Any]) -> str:
    source_name = str(metadata.get("experiment_name") or source_run.name)
    if source_group in source_name:
        return slugify(source_name.replace(source_group, target_group, 1))
    return slugify(f"{source_name}_{target_group}")


def _load_checkpoint_config(checkpoint: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    cfg = checkpoint.get("config")
    if isinstance(cfg, dict):
        return copy.deepcopy(cfg)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Checkpoint has no config and config.yaml is missing: {run_dir}")
    return load_yaml(config_path)


def _metrics_max_epochs(run_dir: Path) -> int | None:
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    try:
        metrics = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    value = metrics.get("max_epochs")
    return int(value) if value is not None else None


def _metrics_early_stopped(run_dir: Path) -> bool:
    path = run_dir / "metrics.json"
    if not path.exists():
        return False
    try:
        metrics = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(metrics.get("early_stopped", False))


def _prepare_resume_target(
    source_run: Path,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[Path, Path, dict[str, Any], int]:
    import torch

    source_checkpoint_path = source_run / "last_checkpoint.pth"
    if not source_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing last_checkpoint.pth: {source_run}")
    source_checkpoint = torch.load(source_checkpoint_path, map_location="cpu")
    source_epoch = int(source_checkpoint.get("epoch", 0))
    source_metadata = _load_json(source_run / "metadata.json")

    if args.target_group:
        target_group = slugify(args.target_group)
        target_name = _derive_target_name(source_run, args.source_group, target_group, source_metadata)
        target_run = output_root / target_group / target_name
        target_checkpoint_path = target_run / "last_checkpoint.pth"
        if args.dry_run:
            return target_run, target_checkpoint_path, {}, source_epoch
        if target_run.exists():
            if args.skip_completed and (_metrics_max_epochs(target_run) or 0) >= int(args.target_epochs):
                return target_run, target_checkpoint_path, {}, source_epoch
            if not args.resume_target_existing:
                raise FileExistsError(
                    f"Target run already exists: {target_run}. "
                    "Use --resume-target-existing to continue it, or choose another --target-group."
                )
        else:
            shutil.copytree(source_run, target_run)
    else:
        target_run = source_run
        target_checkpoint_path = source_checkpoint_path
        if args.dry_run:
            return target_run, target_checkpoint_path, {}, source_epoch

    checkpoint = torch.load(target_checkpoint_path, map_location="cpu")
    cfg = _load_checkpoint_config(checkpoint, target_run)
    cfg.setdefault("training", {})["epochs"] = int(args.target_epochs)
    cfg["training"]["resume_from"] = str(target_checkpoint_path)
    if args.target_group:
        cfg["experiment_group"] = slugify(args.target_group)
        cfg["experiment_name"] = target_run.name
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item}")
        key, value = item.split("=", 1)
        set_by_dotted_key(cfg, key, parse_override(value))
    cfg.setdefault("training", {})["epochs"] = int(args.target_epochs)
    cfg["training"]["resume_from"] = str(target_checkpoint_path)
    validate_experiment_config(cfg)

    checkpoint["experiment_dir"] = str(target_run.resolve())
    checkpoint["config"] = cfg
    if args.reset_patience:
        checkpoint["patience_counter"] = 0
    _atomic_torch_save(checkpoint, target_checkpoint_path)
    write_yaml(target_run / "config.yaml", cfg)
    return target_run, target_checkpoint_path, cfg, int(checkpoint.get("epoch", source_epoch))


def main() -> int:
    args = parse_args()
    from timepix.training.runner import run_experiment

    output_root = resolve_project_path(args.output_root)
    runs = _source_runs(output_root, args.source_group)
    manifest_path = _manifest_path(args)
    rows: list[dict[str, Any]] = []

    print(f"Planned run extensions: {len(runs)}")
    for index, source_run in enumerate(runs, start=1):
        row = {
            "index": index,
            "source_run": str(source_run),
            "target_run": "",
            "status": "planned",
            "start_epoch": "",
            "target_epochs": int(args.target_epochs),
            "error": "",
        }
        rows.append(row)
        _write_manifest(manifest_path, rows)
        try:
            if args.skip_early_stopped and _metrics_early_stopped(source_run):
                row["status"] = "skipped_early_stopped"
                print(f"[{index}/{len(runs)}] skipped early-stopped source: {source_run}")
                _write_manifest(manifest_path, rows)
                continue
            target_run, checkpoint_path, cfg, checkpoint_epoch = _prepare_resume_target(source_run, args, output_root)
            row["target_run"] = str(target_run)
            row["start_epoch"] = checkpoint_epoch + 1
            if checkpoint_epoch >= int(args.target_epochs):
                row["status"] = "skipped_completed"
                print(f"[{index}/{len(runs)}] skipped completed: {target_run}")
                _write_manifest(manifest_path, rows)
                continue
            if args.dry_run:
                row["status"] = "dry_run"
                print(f"[{index}/{len(runs)}] would resume {target_run} from epoch {checkpoint_epoch + 1} to {args.target_epochs}")
                _write_manifest(manifest_path, rows)
                continue
            row["status"] = "running"
            _write_manifest(manifest_path, rows)
            metadata = run_experiment(
                cfg,
                output_root=args.output_root,
                data_root_override=args.data_root,
                experiment_name=cfg.get("experiment_name"),
            )
            row["status"] = "done"
            row["target_run"] = metadata["experiment_dir"]
            print(f"[{index}/{len(runs)}] extended: {metadata['experiment_dir']}")
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            _write_manifest(manifest_path, rows)
            print(f"[{index}/{len(runs)}] failed: {row['error']}")
            if not args.continue_on_error:
                raise
        _write_manifest(manifest_path, rows)
    print(f"Extension manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
