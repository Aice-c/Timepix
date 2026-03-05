#!/usr/bin/env python3
"""
移动数据集脚本
Merge selected category folders from multiple dataset roots into one target, keeping ToA/ToT separated.

Assumptions (configurable by flags):
- Each source dataset root contains two modality subfolders (default: "ToA" and "ToT").
- Inside each modality folder are category folders to merge (layout = modality_first),
  OR each source contains category folders and within each category there are modality folders (layout = category_first).

Features:
- Select multiple sources and categories to merge.
- Preserve modalities (e.g., ToA/ToT) in the target.
- Handle filename conflicts: skip | overwrite | rename (default: rename by adding numeric suffix).
- Optional prefix of source dataset name to all copied filenames to avoid collisions across sources.
- Dry-run mode to preview operations.

Examples:
  # Typical case: sources/<ToA|ToT>/<class>/... -> target/<ToA|ToT>/<class>/...
  python merge_modalities_by_categories.py \
    --sources /data/ds1 /data/ds2 \
    --categories class_0 class_3 \
    --target /data/merged \
    --modalities ToA ToT \
    --on-conflict rename --prefix-source-name

  # When sources/<class>/<ToA|ToT>/... layout
  python merge_modalities_by_categories.py \
    --sources /data/cluster/Alpha_class_1 /data/cluster/Alpha_class_4 \
    --categories cluster_0 cluster_3 \
    --target /data/merged_clusters \
    --layout category_first
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge selected categories across sources while keeping modalities separated.")
    p.add_argument("--sources", nargs="+", required=True, help="One or more source dataset root directories")
    p.add_argument("--categories", nargs="+", required=True, help="Category folder names to include (exact names)")
    p.add_argument("--target", required=True, help="Target root directory to write merged data")
    p.add_argument("--modalities", nargs="+", default=["ToA", "ToT"], help="Modality folder names to process (default: ToA ToT)")
    p.add_argument(
        "--layout",
        choices=["modality_first", "category_first"],
        default="modality_first",
        help="Folder layout within sources: modality_first means sources/<modality>/<category>,\n"
             "category_first means sources/<category>/<modality>."
    )
    p.add_argument(
        "--on-conflict",
        choices=["skip", "overwrite", "rename"],
        default="rename",
        help="How to handle filename conflicts at the destination (default: rename)."
    )
    p.add_argument(
        "--prefix-source-name",
        action="store_true",
        help="Prefix copied filenames with the source dataset folder name (reduces collisions)."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without copying any files."
    )
    return p.parse_args()


def ensure_dir(path: Path, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[DRY] mkdir -p {path}")
        return
    path.mkdir(parents=True, exist_ok=True)


def iter_files_recursive(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            yield Path(dirpath) / fn


def unique_name(dst: Path) -> Path:
    """Return a non-existing path by appending _1, _2, ... before suffix."""
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def map_paths(
    source_root: Path,
    category: str,
    modalities: List[str],
    layout: str,
) -> List[Tuple[str, Path]]:
    """For a given source_root and category, return list of (modality, source_category_path)."""
    paths: List[Tuple[str, Path]] = []
    if layout == "modality_first":
        # source/<modality>/<category>
        for m in modalities:
            p = source_root / m / category
            paths.append((m, p))
    else:
        # category_first: source/<category>/<modality>
        for m in modalities:
            p = source_root / category / m
            paths.append((m, p))
    return paths


def copy_file(src: Path, dst: Path, on_conflict: str, dry_run: bool) -> Path | None:
    if dst.exists():
        if on_conflict == "skip":
            print(f"[SKIP] exists: {dst}")
            return None
        elif on_conflict == "rename":
            dst = unique_name(dst)
        elif on_conflict == "overwrite":
            pass
    if dry_run:
        print(f"[DRY] copy {src} -> {dst}")
        return dst
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst


def planned_dst_path(
    src_file: Path,
    dst_root: Path,
    modality: str,
    category: str,
    prefix: str | None,
) -> Path:
    # keep relative structure under the category root
    name = src_file.name
    if prefix:
        # Insert prefix before the filename (keep suffix)
        if name.startswith(prefix + "_"):
            out_name = name
        else:
            out_name = f"{prefix}_{name}"
    else:
        out_name = name
    # Recreate subdirectory structure under target/<modality>/<category>/...
    return dst_root / modality / category / out_name


def copy_category_from_source(
    source_root: Path,
    target_root: Path,
    category: str,
    modalities: List[str],
    layout: str,
    on_conflict: str,
    prefix_source_name: bool,
    dry_run: bool,
) -> Tuple[int, int]:
    """Copy one category across modalities from one source. Returns (files_copied, missing_paths)."""
    files = 0
    missing = 0
    src_name_prefix = source_root.name if prefix_source_name else None

    for modality, src_cat_path in map_paths(source_root, category, modalities, layout):
        if not src_cat_path.exists():
            print(f"[WARN] source missing: {src_cat_path}")
            missing += 1
            continue
        # Copy all files recursively from src_cat_path
        for f in iter_files_recursive(src_cat_path):
            if not f.is_file():
                continue
            # Preserve nested subdirectories relative to the category path
            # Compute relative path under src_cat_path
            rel = f.relative_to(src_cat_path)
            # Apply optional prefix only to the file name (last part), keep subdirs
            if src_name_prefix:
                rel_parent = rel.parent
                rel_name = rel.name
                if not rel_name.startswith(src_name_prefix + "_"):
                    rel_name = f"{src_name_prefix}_{rel_name}"
                rel = rel_parent / rel_name
            dst = target_root / modality / category / rel
            copied = copy_file(f, dst, on_conflict, dry_run)
            if copied is not None:
                files += 1

    return files, missing


def main() -> int:
    args = parse_args()
    sources = [Path(s).resolve() for s in args.sources]
    target_root = Path(args.target).resolve()
    modalities: List[str] = list(args.modalities)

    # Validate sources
    for s in sources:
        if not s.exists() or not s.is_dir():
            print(f"[ERROR] source does not exist or not a directory: {s}")
            return 2

    ensure_dir(target_root, dry_run=args.dry_run)

    total_files = 0
    total_missing = 0

    for s in sources:
        print(f"[INFO] Processing source: {s}")
        for c in args.categories:
            print(f"  [CAT] {c}")
            files, missing = copy_category_from_source(
                source_root=s,
                target_root=target_root,
                category=c,
                modalities=modalities,
                layout=args.layout,
                on_conflict=args.on_conflict,
                prefix_source_name=args.prefix_source_name,
                dry_run=args.dry_run,
            )
            print(f"    -> copied {files} files; missing paths: {missing}")
            total_files += files
            total_missing += missing

    print("\n[SUMMARY]")
    print(f"  Sources: {len(sources)} | Categories: {len(args.categories)} | Modalities: {', '.join(modalities)}")
    print(f"  Copied files: {total_files}")
    print(f"  Missing source paths encountered: {total_missing}")
    print(f"  Target: {target_root}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
