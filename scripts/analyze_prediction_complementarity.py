#!/usr/bin/env python
"""Analyze prediction complementarity between trained Timepix runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timepix.config import resolve_project_path


DEFAULT_EXPERIMENT_ROOT = Path("outputs/experiments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether ToA or ToT+ToA predictions complement ToT errors. "
            "This uses existing predictions.csv files and does not train or load models."
        )
    )
    parser.add_argument("--root", default=str(DEFAULT_EXPERIMENT_ROOT), help="Experiment output root")
    parser.add_argument("--tot-group", default="a4_modality_comparison_seed42", help="Group containing ToT runs")
    parser.add_argument("--toa-group", default="a4_modality_comparison_seed42", help="Group containing ToA runs")
    parser.add_argument(
        "--candidate-group",
        action="append",
        default=["a4b_toa_transform_seed42"],
        help="Group containing optional ToT+ToA candidate runs; may be repeated",
    )
    parser.add_argument("--tot-run", default=None, help="Explicit ToT run directory or metadata.json")
    parser.add_argument("--toa-run", default=None, help="Explicit ToA run directory or metadata.json")
    parser.add_argument(
        "--candidate-run",
        action="append",
        default=[],
        help="Explicit candidate run directory or metadata.json; may be repeated",
    )
    parser.add_argument("--seed", type=int, default=None, help="Restrict automatic discovery to one training seed")
    parser.add_argument("--output-json", default=None, help="Output JSON path")
    parser.add_argument("--output-summary", default=None, help="Output summary CSV path")
    parser.add_argument("--output-by-class", default=None, help="Output per-class CSV path")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _run_dir(path: str | Path) -> Path:
    candidate = resolve_project_path(path)
    if candidate.name == "metadata.json":
        candidate = candidate.parent
    if not candidate.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {candidate}")
    return candidate


def _metadata_path(run_dir: Path) -> Path:
    path = run_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {run_dir}")
    return path


def _predictions_path(run_dir: Path) -> Path:
    path = run_dir / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions.csv: {run_dir}")
    return path


def _metadata(run_dir: Path) -> dict[str, Any]:
    return _load_json(_metadata_path(run_dir))


def _modalities(metadata: dict[str, Any]) -> tuple[str, ...]:
    dataset = metadata.get("dataset", {})
    data_info = metadata.get("data_info", {})
    return tuple(dataset.get("modalities") or data_info.get("modalities") or [])


def _training_seed(metadata: dict[str, Any]) -> int | None:
    seed = metadata.get("training", {}).get("seed")
    return int(seed) if seed is not None else None


def _read_predictions(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _predictions_path(run_dir).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {
            "row",
            "true_label",
            "pred_label",
            "true_angle",
            "pred_angle_argmax",
            "abs_error_argmax",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{_predictions_path(run_dir)} missing columns: {sorted(missing)}")
        for item in reader:
            rows.append(
                {
                    "row": int(item["row"]),
                    "true_label": int(item["true_label"]),
                    "pred_label": int(item["pred_label"]),
                    "true_angle": float(item["true_angle"]),
                    "pred_angle_argmax": float(item["pred_angle_argmax"]),
                    "abs_error_argmax": float(item["abs_error_argmax"]),
                }
            )
    if not rows:
        raise ValueError(f"No predictions found: {_predictions_path(run_dir)}")
    return rows


def _validate_alignment(reference: list[dict[str, Any]], other: list[dict[str, Any]], other_name: str) -> None:
    if len(reference) != len(other):
        raise ValueError(f"Prediction length mismatch for {other_name}: {len(reference)} vs {len(other)}")
    for idx, (a, b) in enumerate(zip(reference, other)):
        if a["row"] != b["row"] or a["true_label"] != b["true_label"] or a["true_angle"] != b["true_angle"]:
            raise ValueError(f"Prediction rows are not aligned for {other_name} at row index {idx}")


def _correct(row: dict[str, Any]) -> bool:
    return int(row["true_label"]) == int(row["pred_label"])


def _accuracy(rows: list[dict[str, Any]]) -> float:
    return sum(1 for row in rows if _correct(row)) / max(len(rows), 1)


def _mae(rows: list[dict[str, Any]]) -> float:
    return sum(float(row["abs_error_argmax"]) for row in rows) / max(len(rows), 1)


def _angle_key(row: dict[str, Any]) -> str:
    angle = row["true_angle"]
    return str(int(angle)) if float(angle).is_integer() else str(angle)


def _run_label(run_dir: Path, prefix: str | None = None) -> str:
    metadata = _metadata(run_dir)
    data_info = metadata.get("data_info", {})
    data_cfg = metadata.get("data", {})
    transform = data_info.get("toa_transform", data_cfg.get("toa_transform"))
    add_hit_mask = data_info.get("add_hit_mask", data_cfg.get("add_hit_mask"))
    modalities = "+".join(_modalities(metadata))
    if transform and str(transform) != "none":
        mask = "mask" if add_hit_mask else "no_mask"
        base = f"{transform}_{mask}"
    elif modalities:
        base = modalities
    else:
        base = metadata.get("experiment_name", run_dir.name)
    return f"{prefix}_{base}" if prefix else str(base)


def _discover_group_runs(root: Path, group: str, seed: int | None = None) -> list[Path]:
    group_root = resolve_project_path(root) / group
    if not group_root.is_dir():
        return []
    runs = []
    for metadata_path in sorted(group_root.glob("*/metadata.json")):
        metadata = _load_json(metadata_path)
        run_seed = _training_seed(metadata)
        if seed is not None and run_seed != seed:
            continue
        runs.append(metadata_path.parent)
    return runs


def _discover_single_modality(root: Path, group: str, modality: str, seed: int | None) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for run_dir in _discover_group_runs(root, group, seed):
        metadata = _metadata(run_dir)
        if _modalities(metadata) != (modality,):
            continue
        run_seed = _training_seed(metadata)
        if run_seed is None:
            continue
        found[run_seed] = run_dir
    return found


def _discover_candidates(root: Path, groups: list[str], explicit_runs: list[str], seed: int | None) -> dict[int, list[Path]]:
    candidates: dict[int, list[Path]] = {}
    for group in groups:
        for run_dir in _discover_group_runs(root, group, seed):
            metadata = _metadata(run_dir)
            if _modalities(metadata) != ("ToT", "ToA"):
                continue
            run_seed = _training_seed(metadata)
            if run_seed is None:
                continue
            candidates.setdefault(run_seed, []).append(run_dir)

    for item in explicit_runs:
        run_dir = _run_dir(item)
        metadata = _metadata(run_dir)
        run_seed = _training_seed(metadata)
        if seed is not None and run_seed != seed:
            continue
        candidates.setdefault(int(run_seed or 0), []).append(run_dir)

    return candidates


def _pair_summary(
    seed: int,
    base_name: str,
    base_rows: list[dict[str, Any]],
    other_name: str,
    other_rows: list[dict[str, Any]],
    comparator_type: str,
    base_run: Path,
    other_run: Path,
) -> dict[str, Any]:
    _validate_alignment(base_rows, other_rows, other_name)
    n = len(base_rows)

    base_correct = [_correct(row) for row in base_rows]
    other_correct = [_correct(row) for row in other_rows]
    base_errors = [float(row["abs_error_argmax"]) for row in base_rows]
    other_errors = [float(row["abs_error_argmax"]) for row in other_rows]
    base_wrong_indexes = [idx for idx, ok in enumerate(base_correct) if not ok]

    base_correct_other_wrong = sum(1 for a, b in zip(base_correct, other_correct) if a and not b)
    base_wrong_other_correct = sum(1 for a, b in zip(base_correct, other_correct) if not a and b)
    both_correct = sum(1 for a, b in zip(base_correct, other_correct) if a and b)
    both_wrong = sum(1 for a, b in zip(base_correct, other_correct) if not a and not b)
    other_better_on_base_wrong = sum(1 for idx in base_wrong_indexes if other_errors[idx] < base_errors[idx])
    other_equal_on_base_wrong = sum(1 for idx in base_wrong_indexes if other_errors[idx] == base_errors[idx])
    other_worse_on_base_wrong = sum(1 for idx in base_wrong_indexes if other_errors[idx] > base_errors[idx])
    oracle_correct = sum(1 for a, b in zip(base_correct, other_correct) if a or b)
    oracle_errors = [min(a, b) for a, b in zip(base_errors, other_errors)]

    base_acc = _accuracy(base_rows)
    other_acc = _accuracy(other_rows)
    base_mae = _mae(base_rows)
    other_mae = _mae(other_rows)
    oracle_acc = oracle_correct / max(n, 1)
    oracle_mae = sum(oracle_errors) / max(n, 1)

    return {
        "seed": seed,
        "comparator_type": comparator_type,
        "base": base_name,
        "other": other_name,
        "n": n,
        "base_accuracy": base_acc,
        "other_accuracy": other_acc,
        "base_mae": base_mae,
        "other_mae": other_mae,
        "both_correct": both_correct,
        "base_correct_other_wrong": base_correct_other_wrong,
        "base_wrong_other_correct": base_wrong_other_correct,
        "both_wrong": both_wrong,
        "base_wrong_count": len(base_wrong_indexes),
        "other_better_when_base_wrong": other_better_on_base_wrong,
        "other_equal_when_base_wrong": other_equal_on_base_wrong,
        "other_worse_when_base_wrong": other_worse_on_base_wrong,
        "other_better_when_base_wrong_rate": other_better_on_base_wrong / max(len(base_wrong_indexes), 1),
        "oracle_accuracy": oracle_acc,
        "oracle_accuracy_gain_vs_base": oracle_acc - base_acc,
        "oracle_mae": oracle_mae,
        "oracle_mae_gain_vs_base": base_mae - oracle_mae,
        "base_run": str(base_run),
        "other_run": str(other_run),
    }


def _pair_by_class(
    seed: int,
    base_name: str,
    base_rows: list[dict[str, Any]],
    other_name: str,
    other_rows: list[dict[str, Any]],
    comparator_type: str,
) -> list[dict[str, Any]]:
    _validate_alignment(base_rows, other_rows, other_name)
    classes = sorted({_angle_key(row) for row in base_rows}, key=lambda value: float(value))
    output = []
    for angle in classes:
        indexes = [idx for idx, row in enumerate(base_rows) if _angle_key(row) == angle]
        n = len(indexes)
        base_correct = [_correct(base_rows[idx]) for idx in indexes]
        other_correct = [_correct(other_rows[idx]) for idx in indexes]
        base_errors = [float(base_rows[idx]["abs_error_argmax"]) for idx in indexes]
        other_errors = [float(other_rows[idx]["abs_error_argmax"]) for idx in indexes]
        base_wrong_indexes = [local_idx for local_idx, ok in enumerate(base_correct) if not ok]
        oracle_correct = sum(1 for a, b in zip(base_correct, other_correct) if a or b)
        output.append(
            {
                "seed": seed,
                "comparator_type": comparator_type,
                "class_angle": angle,
                "base": base_name,
                "other": other_name,
                "n": n,
                "base_accuracy": sum(base_correct) / max(n, 1),
                "other_accuracy": sum(other_correct) / max(n, 1),
                "both_correct": sum(1 for a, b in zip(base_correct, other_correct) if a and b),
                "base_correct_other_wrong": sum(1 for a, b in zip(base_correct, other_correct) if a and not b),
                "base_wrong_other_correct": sum(1 for a, b in zip(base_correct, other_correct) if not a and b),
                "both_wrong": sum(1 for a, b in zip(base_correct, other_correct) if not a and not b),
                "other_better_when_base_wrong": sum(
                    1 for idx in base_wrong_indexes if other_errors[idx] < base_errors[idx]
                ),
                "oracle_accuracy": oracle_correct / max(n, 1),
                "oracle_accuracy_gain_vs_base": oracle_correct / max(n, 1) - sum(base_correct) / max(n, 1),
                "oracle_mae": sum(min(a, b) for a, b in zip(base_errors, other_errors)) / max(n, 1),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _default_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    stem = "a4b_prediction_complementarity"
    if args.seed is not None:
        stem = f"{stem}_seed{args.seed}"
    json_path = Path(args.output_json) if args.output_json else Path("outputs") / f"{stem}.json"
    summary_path = Path(args.output_summary) if args.output_summary else Path("outputs") / f"{stem}_summary.csv"
    by_class_path = Path(args.output_by_class) if args.output_by_class else Path("outputs") / f"{stem}_by_class.csv"
    return json_path, summary_path, by_class_path


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if bool(args.tot_run) != bool(args.toa_run):
        raise ValueError("--tot-run and --toa-run must be provided together")

    if args.tot_run and args.toa_run:
        tot_run = _run_dir(args.tot_run)
        toa_run = _run_dir(args.toa_run)
        seed = args.seed
        if seed is None:
            seed = _training_seed(_metadata(tot_run)) or 0
        pairs = {seed: {"tot": tot_run, "toa": toa_run}}
    else:
        tot_runs = _discover_single_modality(root, args.tot_group, "ToT", args.seed)
        toa_runs = _discover_single_modality(root, args.toa_group, "ToA", args.seed)
        common_seeds = sorted(set(tot_runs) & set(toa_runs))
        if not common_seeds:
            raise RuntimeError("No matching ToT/ToA single-modality runs found")
        pairs = {seed: {"tot": tot_runs[seed], "toa": toa_runs[seed]} for seed in common_seeds}

    candidate_runs = _discover_candidates(root, args.candidate_group, args.candidate_run, args.seed)
    summary_rows: list[dict[str, Any]] = []
    by_class_rows: list[dict[str, Any]] = []
    json_payload = {"analysis": "a4b_prediction_complementarity", "seeds": []}

    for seed in sorted(pairs):
        tot_run = pairs[seed]["tot"]
        toa_run = pairs[seed]["toa"]
        tot_rows = _read_predictions(tot_run)
        toa_rows = _read_predictions(toa_run)
        tot_name = "ToT"
        toa_name = "ToA"

        seed_payload: dict[str, Any] = {
            "seed": seed,
            "tot_run": str(tot_run),
            "toa_run": str(toa_run),
            "comparisons": [],
        }

        pair = _pair_summary(seed, tot_name, tot_rows, toa_name, toa_rows, "tot_vs_toa", tot_run, toa_run)
        summary_rows.append(pair)
        by_class_rows.extend(_pair_by_class(seed, tot_name, tot_rows, toa_name, toa_rows, "tot_vs_toa"))
        seed_payload["comparisons"].append(pair)

        for candidate_run in candidate_runs.get(seed, []):
            candidate_rows = _read_predictions(candidate_run)
            candidate_name = _run_label(candidate_run, prefix="candidate")
            comparison = _pair_summary(
                seed,
                tot_name,
                tot_rows,
                candidate_name,
                candidate_rows,
                "tot_vs_candidate",
                tot_run,
                candidate_run,
            )
            summary_rows.append(comparison)
            by_class_rows.extend(
                _pair_by_class(seed, tot_name, tot_rows, candidate_name, candidate_rows, "tot_vs_candidate")
            )
            seed_payload["comparisons"].append(comparison)

        json_payload["seeds"].append(seed_payload)

    json_path, summary_path, by_class_path = _default_outputs(args)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary_path, summary_rows)
    _write_csv(by_class_path, by_class_rows)

    print(f"Wrote JSON: {json_path}")
    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote per-class CSV: {by_class_path}")
    for row in summary_rows:
        print(
            "Complementarity | "
            f"seed={row['seed']} {row['base']} vs {row['other']} "
            f"base_acc={row['base_accuracy']:.4f} other_acc={row['other_accuracy']:.4f} "
            f"oracle_acc={row['oracle_accuracy']:.4f} "
            f"oracle_gain={row['oracle_accuracy_gain_vs_base']:.4f} "
            f"base_wrong_other_correct={row['base_wrong_other_correct']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
