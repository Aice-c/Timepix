"""CSV and Markdown table writers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _format_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("\n", " ")


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy() if max_rows else df.copy()
    headers = [str(col) for col in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[col]) for col in view.columns) + " |")
    if max_rows and len(df) > max_rows:
        lines.append("")
        lines.append(f"_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, path_without_suffix: str | Path, *, markdown_rows: int | None = None) -> tuple[Path, Path]:
    path = Path(path_without_suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path.with_suffix(".csv")
    md_path = path.with_suffix(".md")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(dataframe_to_markdown(df, markdown_rows), encoding="utf-8")
    return csv_path, md_path


def write_markdown(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path

