"""Build particle v5 by removing OOF all-model Co/Sr confusions from v3.

The removal set is defined from P9a all_wrong_same predictions:
- true Sr, all three models predicted Co
- true Co, all three models predicted Sr

The script creates a new dataset directory with the same class/modality layout
as the source dataset and writes metadata plus the removed sample manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


MODELS = ("tot", "dual_concat", "gmu_totstrong")
MODALITIES = ("ToT", "ToA")


def _read_removal_rows(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_class = row["true_class"]
            preds = [row[f"{model}_pred"] for model in MODELS]
            if true_class == "Sr" and all(pred == "Co" for pred in preds):
                rows.append(row)
            elif true_class == "Co" and all(pred == "Sr" for pred in preds):
                rows.append(row)
    return rows


def _sample_key_from_file(class_name: str, modality: str, path: Path) -> str:
    suffix = f"_{modality}.txt"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected {modality} filename: {path}")
    stem = path.name[: -len(suffix)] + "_"
    return f"{class_name}/{stem}.txt"


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build v5 particle dataset by removing P9a all-model Co/Sr OOF confusions."
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path(
            r"E:\TimepixData\particle\datasets"
            r"\particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v3\dataset"
        ),
    )
    parser.add_argument(
        "--diagnostic-csv",
        type=Path,
        default=Path(
            "outputs/diagnostics/p9a_ptype_stage1_gmm02_p_v3_oof5_seed42/"
            "all_wrong_same_pred_samples.csv"
        ),
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path(
            r"E:\TimepixData\particle\datasets"
            r"\particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v5_cosr_all3_removed"
        ),
        help="Target dataset version root. The actual data are written under target-root/dataset.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Use hardlinks to save disk space when source and target are on the same volume.",
    )
    args = parser.parse_args()

    source_dataset = args.source_dataset
    target_root = args.target_root
    target_dataset = target_root / "dataset"
    diagnostic_csv = args.diagnostic_csv

    if not source_dataset.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dataset}")
    if not diagnostic_csv.exists():
        raise FileNotFoundError(f"Diagnostic CSV not found: {diagnostic_csv}")
    if target_root.exists():
        raise FileExistsError(f"Target already exists; refusing to overwrite: {target_root}")

    removal_rows = _read_removal_rows(diagnostic_csv)
    remove_keys = {row["sample_key"] for row in removal_rows}
    remove_by_class = Counter(row["true_class"] for row in removal_rows)
    remove_by_transition = Counter(
        f'{row["true_class"]}->{row["tot_pred"]}' for row in removal_rows
    )

    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    skipped_keys_seen: set[str] = set()

    for class_dir in sorted(p for p in source_dataset.iterdir() if p.is_dir()):
        class_name = class_dir.name
        for modality in MODALITIES:
            modality_dir = class_dir / modality
            if not modality_dir.exists():
                continue
            for src_file in sorted(modality_dir.glob("*.txt")):
                sample_key = _sample_key_from_file(class_name, modality, src_file)
                counts[f"{class_name}/{modality}"]["source_files"] += 1
                if sample_key in remove_keys:
                    counts[f"{class_name}/{modality}"]["removed_files"] += 1
                    skipped_keys_seen.add(sample_key)
                    continue
                dst_file = target_dataset / class_name / modality / src_file.name
                _link_or_copy(src_file, dst_file, args.link_mode)
                counts[f"{class_name}/{modality}"]["kept_files"] += 1

    missing_removal_keys = sorted(remove_keys - skipped_keys_seen)
    if missing_removal_keys:
        preview = ", ".join(missing_removal_keys[:10])
        raise RuntimeError(
            f"{len(missing_removal_keys)} removal sample keys were not found in the source dataset: {preview}"
        )

    removed_csv = target_root / "removed_oof_all3_cosr_samples.csv"
    _write_csv(
        removed_csv,
        removal_rows,
        list(removal_rows[0].keys()) if removal_rows else ["sample_key"],
    )

    count_rows: list[dict[str, object]] = []
    for key in sorted(counts):
        class_name, modality = key.split("/")
        source_files = counts[key]["source_files"]
        kept_files = counts[key]["kept_files"]
        removed_files = counts[key]["removed_files"]
        count_rows.append(
            {
                "class": class_name,
                "modality": modality,
                "source_files": source_files,
                "kept_files": kept_files,
                "removed_files": removed_files,
            }
        )
    _write_csv(
        target_root / "dataset_counts.csv",
        count_rows,
        ["class", "modality", "source_files", "kept_files", "removed_files"],
    )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(source_dataset),
        "target_dataset": str(target_dataset),
        "diagnostic_csv": str(diagnostic_csv),
        "link_mode": args.link_mode,
        "removal_rule": (
            "Remove samples where all P9a OOF models predicted Co for true Sr "
            "or Sr for true Co."
        ),
        "models": list(MODELS),
        "removed_samples_total": len(removal_rows),
        "removed_by_class": dict(sorted(remove_by_class.items())),
        "removed_by_transition": dict(sorted(remove_by_transition.items())),
        "removed_manifest": str(removed_csv),
        "counts_csv": str(target_root / "dataset_counts.csv"),
    }
    with (target_root / "build_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
