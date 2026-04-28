"""Dataset scanning and output-directory helpers for analysis scripts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DATASET_MODALITIES = {
    "Alpha_100": ["ToT", "ToA"],
    "Alpha_50": ["ToT", "ToA"],
    "Proton_C": ["ToT"],
}
INDEX_COLUMNS = [
    "dataset",
    "dataset_root",
    "angle",
    "angle_value",
    "modality",
    "sample_key",
    "file_name",
    "path",
    "relative_path",
    "status",
    "shape_rows",
    "shape_cols",
    "error",
]


@dataclass(frozen=True)
class OutputLayout:
    root: Path
    tables: Path
    figures: Path
    cache: Path


def make_output_layout(root: str | Path) -> OutputLayout:
    root = Path(root)
    layout = OutputLayout(root=root, tables=root / "tables", figures=root / "figures", cache=root / "cache")
    for path in (layout.root, layout.tables, layout.figures, layout.cache):
        path.mkdir(parents=True, exist_ok=True)
    return layout


def write_manifest(layout: OutputLayout, payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    (layout.root / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def numeric_angle_dirs(dataset_root: Path) -> list[Path]:
    angle_dirs = []
    if not dataset_root.exists():
        return []
    for child in dataset_root.iterdir():
        if child.is_dir():
            try:
                float(child.name)
            except ValueError:
                continue
            angle_dirs.append(child)
    return sorted(angle_dirs, key=lambda p: float(p.name))


def normalize_sample_key(file_name: str, modality: str) -> str:
    stem, suffix = os.path.splitext(file_name)
    return f"{stem.replace(modality, '', 1)}{suffix}"


def _iter_modality_files(angle_dir: Path, modality: str) -> list[Path]:
    modality_dir = angle_dir / modality
    if modality_dir.is_dir():
        return sorted(p for p in modality_dir.glob("*.txt") if p.is_file())
    return sorted(p for p in angle_dir.glob(f"*{modality}*.txt") if p.is_file())


def infer_modalities(dataset: str, dataset_root: Path) -> list[str]:
    defaults = DEFAULT_DATASET_MODALITIES.get(dataset)
    if defaults:
        return list(defaults)
    found = set()
    for angle_dir in numeric_angle_dirs(dataset_root):
        for child in angle_dir.iterdir():
            if child.is_dir() and child.name in {"ToT", "ToA"}:
                found.add(child.name)
        for path in angle_dir.glob("*.txt"):
            for modality in ("ToT", "ToA"):
                if modality in path.name:
                    found.add(modality)
    return sorted(found) or ["ToT"]


def matrix_shape(path: Path) -> tuple[str, int | None, int | None, str]:
    try:
        array = np.loadtxt(path, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - report bad data rows without stopping scan
        return "read_error", None, None, str(exc)
    if array.ndim != 2:
        return "bad_shape", None, None, f"expected 2D matrix, got ndim={array.ndim}"
    return "ok", int(array.shape[0]), int(array.shape[1]), ""


def scan_dataset(data_root: str | Path, dataset: str) -> pd.DataFrame:
    base = Path(data_root)
    dataset_root = base / dataset
    modalities = infer_modalities(dataset, dataset_root)
    rows = []
    for angle_dir in numeric_angle_dirs(dataset_root):
        angle = angle_dir.name
        for modality in modalities:
            for path in _iter_modality_files(angle_dir, modality):
                status, rows_count, cols_count, error = matrix_shape(path)
                rel_path = path.relative_to(base) if path.is_relative_to(base) else path
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_root": str(dataset_root),
                        "angle": angle,
                        "angle_value": float(angle),
                        "modality": modality,
                        "sample_key": f"{angle}/{normalize_sample_key(path.name, modality)}",
                        "file_name": path.name,
                        "path": str(path),
                        "relative_path": str(rel_path),
                        "status": status,
                        "shape_rows": rows_count,
                        "shape_cols": cols_count,
                        "error": error,
                    }
                )
    return pd.DataFrame(rows, columns=INDEX_COLUMNS)


def scan_datasets(data_root: str | Path, datasets: Iterable[str]) -> pd.DataFrame:
    frames = [scan_dataset(data_root, dataset) for dataset in datasets]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def paired_modality_report(index_df: pd.DataFrame, dataset: str, modality_a: str = "ToT", modality_b: str = "ToA") -> dict:
    subset = index_df[(index_df["dataset"] == dataset) & (index_df["status"] == "ok")]
    a_keys = set(subset.loc[subset["modality"] == modality_a, "sample_key"])
    b_keys = set(subset.loc[subset["modality"] == modality_b, "sample_key"])
    return {
        "dataset": dataset,
        "modality_a": modality_a,
        "modality_b": modality_b,
        "paired_count": len(a_keys & b_keys),
        "only_a": len(a_keys - b_keys),
        "only_b": len(b_keys - a_keys),
        "is_fully_paired": len(a_keys - b_keys) == 0 and len(b_keys - a_keys) == 0 and bool(a_keys),
    }


def class_counts(index_df: pd.DataFrame, dataset: str, modalities: list[str] | None = None) -> pd.DataFrame:
    subset = index_df[(index_df["dataset"] == dataset) & (index_df["status"] == "ok")]
    if modalities is not None:
        subset = subset[subset["modality"].isin(modalities)]
    if subset.empty:
        return pd.DataFrame(columns=["dataset", "angle", "modality", "count"])
    out = subset.groupby(["dataset", "angle", "angle_value", "modality"], as_index=False).size()
    out = out.rename(columns={"size": "count"})
    return out.sort_values(["angle_value", "modality"]).drop(columns=["angle_value"])


def load_split_counts(split_dir: str | Path, dataset: str) -> pd.DataFrame:
    split_dir = Path(split_dir)
    rows = []
    if not split_dir.exists():
        return pd.DataFrame(columns=["dataset", "split_file", "split", "count"])
    for path in sorted(split_dir.glob(f"{dataset}*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for split in ("train", "val", "test"):
            rows.append({"dataset": dataset, "split_file": str(path), "split": split, "count": len(payload.get(split, []))})
    return pd.DataFrame(rows)


def read_matrix(path: str | Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32)
