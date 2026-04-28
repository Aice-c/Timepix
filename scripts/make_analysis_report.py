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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined Timepix analysis report")
    parser.add_argument("--data-analysis-root", default="outputs/data_analysis")
    parser.add_argument("--resolution-root", default="outputs/resolution_limit")
    parser.add_argument("--out", default="outputs/analysis_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = combined_report(args.data_analysis_root, args.resolution_root)
    write_markdown(args.out, text)
    print(f"Wrote combined analysis report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

