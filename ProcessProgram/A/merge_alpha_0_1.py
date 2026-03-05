#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge Alpha class 0 and 1 datasets (ToA and ToT separately) into a single merged directory
with class-prefixed filenames to avoid collisions.

- Sources:
  - AlphaAnalysis/data/Alpha/0/ToA
  - AlphaAnalysis/data/Alpha/0/ToT
  - AlphaAnalysis/data/Alpha/1/ToA
  - AlphaAnalysis/data/Alpha/1/ToT
- Destinations (created if missing):
  - AlphaAnalysis/data/Alpha/0_1_merged/ToA
  - AlphaAnalysis/data/Alpha/0_1_merged/ToT

Usage examples:
  python data/merge_alpha_0_1.py                 # dry run summary
  python data/merge_alpha_0_1.py --run           # execute copying
  python data/merge_alpha_0_1.py --run --overwrite
  python data/merge_alpha_0_1.py --run --limit 1000  # copy at most 1000 files per run
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]  # points to AlphaAnalysis/
SRC_ROOT = ROOT / "data" / "Alpha"
DST_ROOT = SRC_ROOT / "0_1_merged"
CLASSES = (0, 1)
MODALITIES = ("ToA", "ToT")


def merge(
    src_root: Path = SRC_ROOT,
    dst_root: Path = DST_ROOT,
    classes=CLASSES,
    modalities=MODALITIES,
    run: bool = False,
    overwrite: bool = False,
    limit: int | None = None,
) -> Dict[str, Dict[str, int]]:
    """Merge files for each modality from class folders into dst_root with class prefix.

    Returns a nested stats dict per modality.
    """
    stats: Dict[str, Dict[str, int]] = {}

    for modality in modalities:
        copied = 0
        skipped = 0
        errors = 0
        dst_dir = dst_root / modality
        dst_dir.mkdir(parents=True, exist_ok=True)

        for cls in classes:
            src_dir = src_root / str(cls) / modality
            if not src_dir.exists():
                print(f"[WARN] Source directory missing: {src_dir}")
                continue

            # Iterate regular files only
            files = [p for p in src_dir.iterdir() if p.is_file()]
            files.sort()  # deterministic order

            for idx, src_file in enumerate(files, start=1):
                if limit is not None and copied >= limit:
                    break
                dst_name = f"{cls}_" + src_file.name
                dst_path = dst_dir / dst_name

                try:
                    if dst_path.exists() and not overwrite:
                        skipped += 1
                        continue
                    if run:
                        # Ensure parent exists (should already) and copy with metadata
                        shutil.copy2(src_file, dst_path)
                    copied += 1
                except Exception as e:
                    errors += 1
                    print(f"[ERROR] Failed to process {src_file} -> {dst_path}: {e}")

        stats[modality] = {"copied": copied, "skipped": skipped, "errors": errors}
        action = "DRY-RUN would copy" if not run else ("Copied" if not overwrite else "Copied/overwrote")
        print(
            f"[SUMMARY] {modality}: {action}={copied}, skipped={skipped}, errors={errors}, dst={dst_dir}"
        )

    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge Alpha class 0 and 1 datasets with prefixes")
    p.add_argument("--run", action="store_true", help="Actually perform copy. Otherwise dry-run summary only.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite destination files if they exist.")
    p.add_argument("--limit", type=int, default=None, help="Maximum number of files to copy (per run, across modalities).")
    p.add_argument("--src", type=Path, default=SRC_ROOT, help="Source root directory (default: %(default)s)")
    p.add_argument("--dst", type=Path, default=DST_ROOT, help="Destination root directory (default: %(default)s)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("[INFO] Source root:", args.src)
    print("[INFO] Destination root:", args.dst)
    print("[INFO] Mode:", "RUN" if args.run else "DRY-RUN")
    print("[INFO] Overwrite:", args.overwrite)
    if args.limit is not None:
        print("[INFO] Limit:", args.limit)

    merge(
        src_root=args.src,
        dst_root=args.dst,
        classes=CLASSES,
        modalities=MODALITIES,
        run=args.run,
        overwrite=args.overwrite,
        limit=args.limit,
    )
