"""Markdown report helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .tables import dataframe_to_markdown


def rel(path: str | Path, root: str | Path) -> str:
    path = Path(path)
    root = Path(root)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def image_link(path: str | Path, root: str | Path) -> str:
    return f"![{Path(path).stem}]({rel(path, root)})"


def dataset_report(
    output_root: str | Path,
    dataset_summary: pd.DataFrame,
    figures: list[Path],
    tables: list[Path],
) -> str:
    output_root = Path(output_root)
    lines = [
        "# Dataset Analysis Report",
        "",
        "This report summarizes dataset integrity, class counts, event-level features, and representative samples.",
        "",
        "## Dataset Summary",
        "",
        dataframe_to_markdown(dataset_summary),
        "",
        "## Key Figures",
        "",
    ]
    for path in figures:
        if path.suffix.lower() == ".png":
            lines.extend([f"### {path.stem}", "", image_link(path, output_root), ""])
    lines.extend(["## Tables", ""])
    for path in tables:
        if path.suffix.lower() == ".csv":
            lines.append(f"- `{rel(path, output_root)}`")
    lines.extend(
        [
            "",
            "## Thesis Notes",
            "",
            "- The raw Timepix3 frame is treated as a 256 x 256 detector response over one time window.",
            "- Connected activated-pixel components are extracted as candidate particle events.",
            "- ToT event statistics such as active pixel count and active ToT sum are used for data cleaning.",
            "- Alpha and C/proton are analyzed as separate datasets and should not be mixed unless a dedicated cross-particle experiment is designed.",
        ]
    )
    return "\n".join(lines) + "\n"


def resolution_report(output_root: str | Path, summary_tables: dict[str, pd.DataFrame], figures: list[Path], tables: list[Path]) -> str:
    output_root = Path(output_root)
    cautious = (
        "在当前探测器设置、事件提取方法、ToT 单模态矩阵表示和已测试模型/特征族条件下，"
        "C/质子近垂直角度 80–90°、2° 间隔的数据没有表现出足够的可分性，难以支持可靠监督分类。"
    )
    lines = [
        "# Resolution-Limit Analysis Report",
        "",
        "## Cautious Thesis Statement",
        "",
        cautious,
        "",
        "This statement should not be rewritten as an absolute claim that deep learning can never distinguish near-vertical angles.",
        "",
    ]
    for name, table in summary_tables.items():
        lines.extend([f"## {name}", "", dataframe_to_markdown(table, max_rows=30), ""])
    lines.extend(["## Key Figures", ""])
    for path in figures:
        if path.suffix.lower() == ".png":
            lines.extend([f"### {path.stem}", "", image_link(path, output_root), ""])
    lines.extend(["## Tables", ""])
    for path in tables:
        if path.suffix.lower() == ".csv":
            lines.append(f"- `{rel(path, output_root)}`")
    return "\n".join(lines) + "\n"


def combined_report(data_root: str | Path, resolution_root: str | Path) -> str:
    data_root = Path(data_root)
    resolution_root = Path(resolution_root)
    lines = [
        "# Timepix Analysis Report",
        "",
        "## Dataset Analysis",
        "",
        f"- Dataset report: `{data_root / 'dataset_analysis_report.md'}`",
        "",
        "## Near-Vertical Resolution-Limit Analysis",
        "",
        f"- Resolution-limit report: `{resolution_root / 'resolution_limit_report.md'}`",
        "",
    ]
    for report in [data_root / "dataset_analysis_report.md", resolution_root / "resolution_limit_report.md"]:
        if report.exists():
            lines.extend(["---", "", report.read_text(encoding="utf-8")])
    return "\n".join(lines)

