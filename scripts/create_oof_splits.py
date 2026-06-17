#!/usr/bin/env python
"""Create stratified out-of-fold split manifests for Timepix datasets."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timepix.config import load_yaml, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True, help="Dataset YAML config path")
    parser.add_argument("--data-root", default=None, help="Override dataset.root")
    parser.add_argument("--modalities", nargs="+", default=None, help="Modalities used to collect paired samples")
    parser.add_argument("--folds", type=int, default=5, help="Number of OOF folds")
    parser.add_argument("--seed", type=int, default=42, help="Fold assignment seed")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio over the full dataset inside each fold")
    parser.add_argument("--output-dir", default="outputs/splits", help="Output directory for split manifests")
    parser.add_argument("--prefix", required=True, help="Output filename prefix")
    return parser.parse_args()


def _class_counts(keys: list[str]) -> dict[str, int]:
    counts = Counter(key.split("/", 1)[0] for key in keys)
    return dict(sorted(counts.items()))


def _normalize_key(file_name: str, modality: str) -> str:
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    normalized = stem.replace(modality, "", 1)
    return f"{normalized}{suffix}"


def _discover_label_dirs(root: Path, label_type: str, class_names: list[str] | None) -> list[str]:
    labels = sorted(path.name for path in root.iterdir() if path.is_dir())
    if class_names:
        missing = [name for name in class_names if name not in labels]
        if missing:
            raise FileNotFoundError(f"Configured class folder does not exist: {missing[0]}")
        return list(class_names)
    if label_type == "angle_folder":
        return sorted(labels, key=lambda name: float(name))
    return sorted(labels, key=str.casefold)


def _collect_keys_by_label(
    data_root: Path,
    modalities: list[str],
    *,
    label_type: str,
    class_names: list[str] | None,
) -> tuple[dict[int, list[str]], dict[int, str]]:
    label_names = _discover_label_dirs(data_root, label_type, class_names)
    keys_by_label: dict[int, list[str]] = {}
    label_map: dict[int, str] = {}
    for label_idx, label_name in enumerate(label_names):
        label_map[label_idx] = label_name
        key_sets: list[set[str]] = []
        for modality in modalities:
            modality_dir = data_root / label_name / modality
            if not modality_dir.is_dir():
                raise FileNotFoundError(f"Missing modality directory: {modality_dir}")
            keys = {
                _normalize_key(path.name, modality)
                for path in modality_dir.iterdir()
                if path.is_file()
            }
            if not keys:
                raise RuntimeError(f"No files in modality directory: {modality_dir}")
            key_sets.append(keys)
        common = sorted(set.intersection(*key_sets))
        if not common:
            raise RuntimeError(f"No paired samples for label {label_name} and modalities {modalities}")
        keys_by_label[label_idx] = [f"{label_name}/{key}" for key in common]
    return keys_by_label, label_map


def _write_manifest(path: Path, train_keys: list[str], val_keys: list[str], test_keys: list[str], metadata: dict[str, Any]) -> None:
    payload = {
        "train": train_keys,
        "val": val_keys,
        "test": test_keys,
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fold",
        "split_path",
        "train_count",
        "val_count",
        "test_count",
        "train_counts_by_class",
        "val_counts_by_class",
        "test_counts_by_class",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _round_robin_folds(keys: list[str], folds: int, rng: random.Random) -> list[list[str]]:
    shuffled = list(keys)
    rng.shuffle(shuffled)
    out = [[] for _ in range(folds)]
    for idx, key in enumerate(shuffled):
        out[idx % folds].append(key)
    return out


def main() -> int:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if not (0.0 < args.val_ratio < 0.5):
        raise ValueError("--val-ratio must be between 0 and 0.5")

    dataset_cfg_path = resolve_project_path(args.dataset_config)
    dataset_cfg = load_yaml(dataset_cfg_path)
    data_root = Path(args.data_root) if args.data_root else resolve_project_path(dataset_cfg["root"])
    modalities = args.modalities or list(dataset_cfg.get("default_modalities") or dataset_cfg.get("available_modalities") or ["ToT"])
    label_type = str(dataset_cfg.get("label_type", "angle_folder"))
    class_names = dataset_cfg.get("class_names")

    keys_by_label, label_map = _collect_keys_by_label(data_root, modalities, label_type=label_type, class_names=class_names)

    rng = random.Random(args.seed)
    folds_by_label = {
        label: _round_robin_folds(sorted(keys), args.folds, rng)
        for label, keys in sorted(keys_by_label.items())
    }

    output_dir = resolve_project_path(args.output_dir)
    summary_rows: list[dict[str, Any]] = []
    all_test_keys: list[str] = []

    for fold in range(args.folds):
        test_keys: list[str] = []
        train_pool_by_label: dict[int, list[str]] = {}
        for label, label_folds in folds_by_label.items():
            holdout = sorted(label_folds[fold])
            test_keys.extend(holdout)
            train_pool = []
            for other_fold, keys in enumerate(label_folds):
                if other_fold != fold:
                    train_pool.extend(keys)
            train_pool_by_label[label] = sorted(train_pool)

        val_keys: list[str] = []
        train_keys: list[str] = []
        fold_rng = random.Random(args.seed * 1009 + fold)
        for label, train_pool in sorted(train_pool_by_label.items()):
            pool = list(train_pool)
            fold_rng.shuffle(pool)
            full_class_count = len(keys_by_label[label])
            val_count = max(1, round(full_class_count * args.val_ratio))
            val_count = min(val_count, max(1, len(pool) - 1))
            val_keys.extend(pool[:val_count])
            train_keys.extend(pool[val_count:])

        train_keys = sorted(train_keys)
        val_keys = sorted(val_keys)
        test_keys = sorted(test_keys)
        all_test_keys.extend(test_keys)

        overlap = set(train_keys) & set(val_keys) | set(train_keys) & set(test_keys) | set(val_keys) & set(test_keys)
        if overlap:
            raise RuntimeError(f"Fold {fold} has overlapping split keys, e.g. {sorted(overlap)[0]}")

        split_path = output_dir / f"{args.prefix}_fold{fold}.json"
        metadata = {
            "type": "out_of_fold",
            "fold": fold,
            "folds": args.folds,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "dataset_config": str(dataset_cfg_path),
            "dataset_name": dataset_cfg.get("name"),
            "data_root": str(data_root),
            "modalities": modalities,
            "label_type": label_type,
            "label_map": {str(k): v for k, v in label_map.items()},
            "note": "test split is the OOF holdout for this fold",
        }
        _write_manifest(split_path, train_keys, val_keys, test_keys, metadata)
        summary_rows.append(
            {
                "fold": fold,
                "split_path": str(split_path),
                "train_count": len(train_keys),
                "val_count": len(val_keys),
                "test_count": len(test_keys),
                "train_counts_by_class": json.dumps(_class_counts(train_keys), ensure_ascii=False, sort_keys=True),
                "val_counts_by_class": json.dumps(_class_counts(val_keys), ensure_ascii=False, sort_keys=True),
                "test_counts_by_class": json.dumps(_class_counts(test_keys), ensure_ascii=False, sort_keys=True),
            }
        )

    total_keys = [key for keys in keys_by_label.values() for key in keys]
    test_counts = Counter(all_test_keys)
    missing = sorted(set(total_keys) - set(all_test_keys))
    repeated = sorted(key for key, count in test_counts.items() if count != 1)
    if missing:
        raise RuntimeError(f"{len(missing)} samples never appear in OOF test split, e.g. {missing[0]}")
    if repeated:
        raise RuntimeError(f"{len(repeated)} samples appear in OOF test split more than once, e.g. {repeated[0]}")

    summary_path = output_dir / f"{args.prefix}_summary.csv"
    _write_summary(summary_path, summary_rows)
    print(f"Wrote {args.folds} OOF split manifests to {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Total samples covered exactly once: {len(total_keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
