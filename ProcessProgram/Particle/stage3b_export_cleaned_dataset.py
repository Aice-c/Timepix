#!/usr/bin/env python3
"""Export source-label cleaned particle ToT/ToA dataset from Stage-3a audit."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_STAGE1_ROOT = Path(r"E:\TimepixData\particle\stage1_single_particle_candidates_100x100")
DEFAULT_STAGE3A_AUDIT = Path(
    r"E:\TimepixData\particle\stage3a_source_cleaning_audit_v2\source_cleaning_audit.csv"
)
DEFAULT_OUTPUT_ROOT = Path(r"E:\TimepixData\particle\particle_source_label_cleaned_tot_toa_v1")


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _copy_pair(stage1_root: Path, output_root: Path, row: pd.Series) -> tuple[str, str]:
    src_tot = stage1_root / "dataset" / str(row["tot_path"])
    src_toa = stage1_root / "dataset" / str(row["toa_path"])
    if not src_tot.is_file():
        raise FileNotFoundError(f"Missing ToT source file: {src_tot}")
    if not src_toa.is_file():
        raise FileNotFoundError(f"Missing ToA source file: {src_toa}")

    dst_tot_rel = Path(str(row["tot_path"]))
    dst_toa_rel = Path(str(row["toa_path"]))
    dst_tot = output_root / "dataset" / dst_tot_rel
    dst_toa = output_root / "dataset" / dst_toa_rel
    dst_tot.parent.mkdir(parents=True, exist_ok=True)
    dst_toa.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_tot, dst_tot)
    shutil.copy2(src_toa, dst_toa)
    return dst_tot_rel.as_posix(), dst_toa_rel.as_posix()


def export_cleaned_dataset(stage1_root: Path, audit_path: Path, output_root: Path) -> dict[str, object]:
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing Stage-3a audit CSV: {audit_path}")
    audit = pd.read_csv(audit_path)
    required = ["sample_key", "particle", "tot_path", "toa_path", "recommended_keep", "reject_reasons"]
    missing = [column for column in required if column not in audit.columns]
    if missing:
        raise ValueError(f"Missing required audit columns: {missing}")
    if "review_flags" not in audit.columns:
        audit["review_flags"] = "none"

    output_root.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    keep_mask = _as_bool_series(audit["recommended_keep"])
    kept = audit[keep_mask].copy()
    rejected = audit[~keep_mask].copy()

    copied_tot: list[str] = []
    copied_toa: list[str] = []
    for _, row in kept.iterrows():
        tot_rel, toa_rel = _copy_pair(stage1_root, output_root, row)
        copied_tot.append(tot_rel)
        copied_toa.append(toa_rel)
    kept["cleaned_tot_path"] = copied_tot
    kept["cleaned_toa_path"] = copied_toa

    kept.to_csv(manifests_dir / "cleaned_manifest.csv", index=False)
    rejected.to_csv(manifests_dir / "cleaned_rejected_manifest.csv", index=False)

    per_particle = []
    for particle, sub in audit.groupby("particle", sort=True):
        sub_keep = kept[kept["particle"] == particle]
        sub_reject = rejected[rejected["particle"] == particle]
        per_particle.append(
            {
                "particle": particle,
                "total": int(len(sub)),
                "exported": int(len(sub_keep)),
                "rejected": int(len(sub_reject)),
                "reject_rate": float(len(sub_reject) / len(sub)) if len(sub) else 0.0,
            }
        )
    per_particle_df = pd.DataFrame(per_particle)
    per_particle_df.to_csv(manifests_dir / "cleaned_counts_by_particle.csv", index=False)

    review_rows = []
    for particle, sub in kept.groupby("particle", sort=True):
        counts: dict[str, int] = {}
        for flag_text in sub["review_flags"].fillna("none").astype(str):
            if flag_text == "none":
                continue
            for flag in flag_text.split(";"):
                counts[flag] = counts.get(flag, 0) + 1
        for flag, count in sorted(counts.items()):
            review_rows.append({"particle": particle, "review_flag": flag, "kept_count": int(count)})
    pd.DataFrame(review_rows).to_csv(manifests_dir / "cleaned_review_flags.csv", index=False)

    summary = {
        "stage": "stage3b_source_label_cleaned_dataset_v1",
        "stage1_root": str(stage1_root),
        "audit_path": str(audit_path),
        "output_root": str(output_root),
        "label_policy": "Radiation source labels are preserved: Am, Co60, Sr.",
        "cleaning_policy": "Export rows with recommended_keep == true from Stage-3a v2; keep review_flags in manifest.",
        "total_count": int(len(audit)),
        "exported_count": int(len(kept)),
        "rejected_count": int(len(rejected)),
        "reject_rate": float(len(rejected) / len(audit)) if len(audit) else 0.0,
        "per_particle": per_particle,
        "outputs": {
            "dataset/": "Cleaned source-label ToT/ToA files in training-layout folders.",
            "manifests/cleaned_manifest.csv": "Exported candidates with original audit fields and cleaned paths.",
            "manifests/cleaned_rejected_manifest.csv": "Rejected candidates retained for audit.",
            "manifests/cleaned_counts_by_particle.csv": "Per-source exported/rejected counts.",
            "manifests/cleaned_review_flags.csv": "Review flags among exported candidates.",
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_STAGE3A_AUDIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = export_cleaned_dataset(args.stage1_root, args.audit_path, args.output_root)
    print(f"Stage-3b cleaned dataset written to: {args.output_root}")
    print(f"Exported candidates: {summary['exported_count']}")
    print(f"Rejected candidates retained in manifest: {summary['rejected_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
