from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_INPUT_ROOT = Path(r"E:\C1Analysis\C_Processed_1")
DEFAULT_OUTPUT_ROOT = Path(r"E:\C1Analysis\C_Processed_1_ToT")


def is_angle_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        float(path.name)
    except ValueError:
        return False
    return True


def iter_angle_dirs(input_root: Path) -> list[Path]:
    angle_dirs = [p for p in input_root.iterdir() if is_angle_dir(p)]
    return sorted(angle_dirs, key=lambda p: float(p.name))


def iter_sample_files(angle_dir: Path, pattern: str) -> list[Path]:
    return sorted(p for p in angle_dir.glob(pattern) if p.is_file())


def copy_angle_files(
    angle_dir: Path,
    output_root: Path,
    *,
    pattern: str,
    overwrite: bool,
    dry_run: bool,
) -> tuple[int, int]:
    dst_dir = output_root / angle_dir.name / "ToT"
    sample_files = iter_sample_files(angle_dir, pattern)

    copied = 0
    skipped = 0
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    for src in sample_files:
        dst = dst_dir / src.name
        if dst.exists() and not overwrite:
            skipped += 1
            continue

        if not dry_run:
            shutil.copy2(src, dst)
        copied += 1

    return copied, skipped


def restructure_dataset(
    input_root: Path,
    output_root: Path,
    *,
    pattern: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    input_root = input_root.resolve()
    output_root = output_root.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")
    if input_root == output_root:
        raise ValueError(
            "Output root must be different from input root. "
            "Use a new folder such as E:\\C1Analysis\\C_Processed_1_ToT."
        )

    angle_dirs = iter_angle_dirs(input_root)
    if not angle_dirs:
        raise RuntimeError(f"No numeric angle folders found under: {input_root}")

    print(f"Input root : {input_root}")
    print(f"Output root: {output_root}")
    print(f"Pattern    : {pattern}")
    print(f"Dry run    : {dry_run}")
    print(f"Overwrite  : {overwrite}")
    print()

    total_copied = 0
    total_skipped = 0
    for angle_dir in angle_dirs:
        copied, skipped = copy_angle_files(
            angle_dir,
            output_root,
            pattern=pattern,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        total_copied += copied
        total_skipped += skipped
        action = "would copy" if dry_run else "copied"
        print(
            f"angle={angle_dir.name:>5} | {action}: {copied:6d} | "
            f"skipped existing: {skipped:6d}"
        )

    print()
    print(f"Angles processed : {len(angle_dirs)}")
    print(f"Files copied     : {total_copied}")
    print(f"Files skipped    : {total_skipped}")
    if dry_run:
        print("Dry run only. No files were written.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restructure C/proton ToT samples from <angle>/*.txt to "
            "<angle>/ToT/*.txt under a new output root."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Source dataset root. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output dataset root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="File glob pattern to copy from each angle folder. Default: *.txt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist in output ToT folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    restructure_dataset(
        args.input_root,
        args.output_root,
        pattern=args.pattern,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
