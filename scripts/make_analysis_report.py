#!/usr/bin/env python
"""Combine dataset and resolution-limit analysis reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timepix.analysis.reports import combined_report
from timepix.analysis.tables import write_markdown
from timepix.analysis.workbook import write_analysis_workbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined Timepix analysis report")
    parser.add_argument("--data-analysis-root", default="outputs/data_analysis")
    parser.add_argument("--resolution-root", default="outputs/resolution_limit")
    parser.add_argument("--out", default="outputs/analysis_report.md")
    parser.add_argument("--tables-out", default="outputs/analysis_tables/timepix_analysis_tables.xlsx")
    parser.add_argument(
        "--include-raw-features",
        action="store_true",
        help="Also include large per-event feature CSVs in the combined workbook.",
    )
    return parser.parse_args()


def should_include_table(csv_path: Path, *, include_raw_features: bool) -> bool:
    if include_raw_features:
        return True
    raw_stems = {
        "proton_c_near_vertical_features",
    }
    if csv_path.stem in raw_stems:
        return False
    if csv_path.stem.startswith("04_event_features_"):
        return False
    return True


def main() -> int:
    args = parse_args()
    text = combined_report(args.data_analysis_root, args.resolution_root)
    write_markdown(args.out, text)
    tables = []
    for root_name, root in [("数据集", Path(args.data_analysis_root)), ("近垂直", Path(args.resolution_root))]:
        for csv_path in sorted(root.glob("*.csv")):
            if not should_include_table(csv_path, include_raw_features=args.include_raw_features):
                continue
            try:
                import pandas as pd

                df = pd.read_csv(csv_path)
            except Exception:
                continue
            stem = f"{root_name}_{csv_path.stem}"
            tables.append((stem, f"{root_name}分析表：{csv_path.stem}", f"来源文件：{csv_path}", df))
    if tables:
        workbook_path = write_analysis_workbook(tables, args.tables_out, title="Timepix 论文数据分析统计表总表")
        print(f"Wrote combined table workbook: {workbook_path}")
    print(f"Wrote combined analysis report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
