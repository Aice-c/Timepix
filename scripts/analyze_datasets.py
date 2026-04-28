#!/usr/bin/env python
"""Generate thesis-oriented dataset analysis tables and figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timepix.analysis.features import extract_feature_table, feature_summary_by_angle
from timepix.analysis.io import (
    class_counts,
    load_split_counts,
    make_output_layout,
    paired_modality_report,
    scan_datasets,
    write_manifest,
)
from timepix.analysis.plotting import (
    plot_alpha_pair_grid,
    plot_box_by_angle,
    plot_class_counts,
    plot_feature_scatter,
    plot_feature_violin,
    plot_preprocessing_pipeline,
    plot_representative_grid,
)
from timepix.analysis.reports import dataset_report
from timepix.analysis.representative import deterministic_sample, select_representatives
from timepix.analysis.tables import write_markdown, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Timepix datasets for thesis figures/tables")
    parser.add_argument("--data-root", default="Data", help="Root containing dataset folders")
    parser.add_argument("--output-root", default="outputs/data_analysis", help="Output root")
    parser.add_argument("--datasets", nargs="+", default=["Alpha_100", "Proton_C"], help="Datasets to analyze")
    parser.add_argument("--sample-cap-plot", type=int, default=5000, help="Per-plot deterministic sample cap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-dir", default="outputs/splits", help="Directory containing split manifests")
    return parser.parse_args()


def _filter_angles(frame: pd.DataFrame, angles: list[float]) -> pd.DataFrame:
    if frame.empty:
        return frame
    wanted = {float(a) for a in angles}
    if "angle_value" in frame.columns:
        values = pd.to_numeric(frame["angle_value"], errors="coerce")
    else:
        values = pd.to_numeric(frame["angle"], errors="coerce")
    return frame[values.isin(wanted)].copy()


def _write_counts(layout, counts: pd.DataFrame, name: str, figures: list[Path], tables: list[Path], title: str) -> None:
    csv_path, md_path = write_table(counts, layout.tables / name)
    tables.extend([csv_path, md_path])
    png, pdf = plot_class_counts(counts, layout.figures / name, title)
    figures.extend([png, pdf])


def _write_feature_summary(layout, features: pd.DataFrame, name: str, tables: list[Path]) -> pd.DataFrame:
    summary = feature_summary_by_angle(features)
    csv_path, md_path = write_table(summary, layout.tables / name)
    tables.extend([csv_path, md_path])
    return summary


def build_dataset_summary(index_df: pd.DataFrame, datasets: list[str], split_dir: str | Path) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        subset = index_df[index_df["dataset"] == dataset]
        ok = subset[subset["status"] == "ok"]
        split_counts = load_split_counts(split_dir, dataset)
        pairing = paired_modality_report(index_df, dataset) if "ToA" in set(subset.get("modality", [])) else {}
        shape_summary = (
            ok.assign(shape=ok["shape_rows"].astype(str) + "x" + ok["shape_cols"].astype(str))["shape"]
            .value_counts()
            .to_dict()
            if not ok.empty
            else {}
        )
        rows.append(
            {
                "dataset": dataset,
                "total_files": int(len(subset)),
                "ok_files": int(len(ok)),
                "read_errors": int((subset["status"] != "ok").sum()) if not subset.empty else 0,
                "angles": ",".join(str(a) for a in sorted(ok["angle"].unique(), key=lambda x: float(x))) if not ok.empty else "",
                "modalities": ",".join(sorted(ok["modality"].unique())) if not ok.empty else "",
                "shapes": str(shape_summary),
                "paired_tot_toa": pairing.get("paired_count", ""),
                "tot_toa_fully_paired": pairing.get("is_fully_paired", ""),
                "split_files": int(split_counts["split_file"].nunique()) if not split_counts.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    layout = make_output_layout(args.output_root)
    write_manifest(
        layout,
        {
            "script": "scripts/analyze_datasets.py",
            "data_root": args.data_root,
            "datasets": args.datasets,
            "sample_cap_plot": args.sample_cap_plot,
            "seed": args.seed,
        },
    )

    figures: list[Path] = []
    tables: list[Path] = []

    index_df = scan_datasets(args.data_root, args.datasets)
    csv_path, md_path = write_table(index_df, layout.tables / "dataset_index", markdown_rows=100)
    tables.extend([csv_path, md_path])

    dataset_summary = build_dataset_summary(index_df, args.datasets, args.split_dir)
    csv_path, md_path = write_table(dataset_summary, layout.tables / "dataset_summary")
    tables.extend([csv_path, md_path])
    split_counts = pd.concat([load_split_counts(args.split_dir, dataset) for dataset in args.datasets], ignore_index=True)
    csv_path, md_path = write_table(split_counts, layout.tables / "split_counts")
    tables.extend([csv_path, md_path])

    png, pdf = plot_preprocessing_pipeline(layout.figures / "preprocessing_pipeline")
    figures.extend([png, pdf])

    all_features = extract_feature_table(index_df)
    all_features.to_csv(layout.cache / "all_event_features.csv", index=False, encoding="utf-8-sig")

    alpha_counts = class_counts(index_df, "Alpha_100")
    _write_counts(layout, alpha_counts, "class_counts_alpha100", figures, tables, "Alpha_100 class counts")

    proton_counts_all = class_counts(index_df, "Proton_C", ["ToT"])
    _write_counts(layout, proton_counts_all, "class_counts_proton_c_all", figures, tables, "Proton_C all angle counts")
    _write_counts(
        layout,
        _filter_angles(proton_counts_all, [10, 20, 30, 45, 50, 60, 70]),
        "class_counts_proton_c_10_70",
        figures,
        tables,
        "Proton_C 10-70 deg counts",
    )
    _write_counts(
        layout,
        _filter_angles(proton_counts_all, [80, 82, 84, 86, 88, 90]),
        "class_counts_proton_c_80_90",
        figures,
        tables,
        "Proton_C near-vertical counts",
    )

    alpha_tot = all_features[(all_features["dataset"] == "Alpha_100") & (all_features["modality"] == "ToT")]
    alpha_toa = all_features[(all_features["dataset"] == "Alpha_100") & (all_features["modality"] == "ToA")]
    proton_tot = all_features[(all_features["dataset"] == "Proton_C") & (all_features["modality"] == "ToT")]
    proton_10_70 = _filter_angles(proton_tot, [10, 20, 30, 45, 50, 60, 70])
    proton_80_90 = _filter_angles(proton_tot, [80, 82, 84, 86, 88, 90])

    alpha_tot.to_csv(layout.cache / "alpha100_tot_features.csv", index=False, encoding="utf-8-sig")
    alpha_toa.to_csv(layout.cache / "alpha100_toa_features.csv", index=False, encoding="utf-8-sig")
    proton_tot.to_csv(layout.cache / "proton_c_tot_features.csv", index=False, encoding="utf-8-sig")

    _write_feature_summary(layout, alpha_tot, "feature_summary_by_angle_alpha100_tot", tables)
    _write_feature_summary(layout, alpha_toa, "feature_summary_by_angle_alpha100_toa", tables)
    _write_feature_summary(layout, proton_10_70, "feature_summary_by_angle_proton_c_10_70", tables)
    _write_feature_summary(layout, proton_80_90, "feature_summary_by_angle_proton_c_80_90", tables)

    alpha_tot_rep = select_representatives(alpha_tot)
    alpha_toa_rep = select_representatives(alpha_toa)
    proton_10_70_rep = select_representatives(proton_10_70)
    proton_80_90_rep = select_representatives(proton_80_90)

    for reps, name, title in [
        (alpha_tot_rep, "representative_alpha100_tot", "Alpha_100 representative ToT samples"),
        (alpha_toa_rep, "representative_alpha100_toa", "Alpha_100 representative ToA samples"),
        (proton_10_70_rep, "representative_proton_c_10_70_tot", "Proton_C 10-70 representative ToT samples"),
        (proton_80_90_rep, "representative_proton_c_80_90_tot", "Proton_C near-vertical representative ToT samples"),
    ]:
        png, pdf = plot_representative_grid(reps, layout.figures / name, title)
        figures.extend([png, pdf])

    for name in ["representative_alpha100_tot_toa_pairs", "alpha100_tot_toa_pair_grid"]:
        png, pdf = plot_alpha_pair_grid(alpha_tot_rep, alpha_toa, layout.figures / name)
        figures.extend([png, pdf])

    for data, name, title in [
        (deterministic_sample(alpha_tot, args.sample_cap_plot, args.seed), "feature_violin_alpha100_tot", "Alpha_100 ToT feature distributions"),
        (deterministic_sample(proton_10_70, args.sample_cap_plot, args.seed), "feature_violin_proton_c_10_70", "Proton_C 10-70 feature distributions"),
        (deterministic_sample(proton_80_90, args.sample_cap_plot, args.seed), "feature_violin_proton_c_80_90", "Proton_C near-vertical feature distributions"),
    ]:
        png, pdf = plot_feature_violin(data, layout.figures / name, title)
        figures.extend([png, pdf])

    for data, name, title in [
        (deterministic_sample(alpha_tot, args.sample_cap_plot, args.seed), "feature_scatter_alpha100_tot", "Alpha_100 ToT active_count vs active_sum"),
        (deterministic_sample(proton_10_70, args.sample_cap_plot, args.seed), "feature_scatter_proton_c_10_70", "Proton_C 10-70 active_count vs active_sum"),
        (deterministic_sample(proton_80_90, args.sample_cap_plot, args.seed), "feature_scatter_proton_c_80_90", "Proton_C near-vertical active_count vs active_sum"),
    ]:
        png, pdf = plot_feature_scatter(data, layout.figures / name, title)
        figures.extend([png, pdf])

    if not alpha_toa.empty:
        for feature, name, title in [
            ("toa_span", "alpha100_toa_span_by_angle", "Alpha_100 ToA span by angle"),
            ("toa_tot_corr", "alpha100_tot_toa_corr_by_angle", "Alpha_100 ToT/ToA correlation by angle"),
        ]:
            png, pdf = plot_box_by_angle(alpha_toa, feature, layout.figures / name, title)
            figures.extend([png, pdf])

    report = dataset_report(layout.root, dataset_summary, figures, tables)
    write_markdown(layout.root / "dataset_analysis_report.md", report)
    print(f"Wrote dataset analysis to {layout.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
