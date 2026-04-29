"""Event-level feature extraction for Timepix matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_matrix
from .progress import iter_progress


EPS = 1e-12
BASE_FEATURES = [
    "active_count",
    "active_ratio",
    "active_sum",
    "active_mean",
    "active_std",
    "active_max",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_fill_ratio",
    "bbox_aspect_ratio",
    "pca_major_length",
    "pca_minor_length",
    "pca_axis_ratio",
    "eccentricity",
    "centroid_row",
    "centroid_col",
    "weighted_centroid_row",
    "weighted_centroid_col",
    "rms_radius",
    "central_energy_fraction_r3",
    "edge_energy_fraction",
    "energy_entropy",
]
TOA_FEATURES = [
    "toa_min_nonzero",
    "toa_max_nonzero",
    "toa_span",
    "toa_mean_nonzero",
    "toa_std_nonzero",
    "toa_tot_corr",
    "toa_gradient_along_pca",
]
FEATURE_TABLE_COLUMNS = ["dataset", "angle", "angle_value", "modality", "sample_key", "path", *BASE_FEATURES, *TOA_FEATURES]


def _empty_base(shape: tuple[int, int]) -> dict[str, float]:
    rows, cols = shape
    return {
        "active_count": 0,
        "active_ratio": 0.0,
        "active_sum": 0.0,
        "active_mean": 0.0,
        "active_std": 0.0,
        "active_max": 0.0,
        "bbox_width": 0.0,
        "bbox_height": 0.0,
        "bbox_area": 0.0,
        "bbox_fill_ratio": 0.0,
        "bbox_aspect_ratio": 0.0,
        "pca_major_length": 0.0,
        "pca_minor_length": 0.0,
        "pca_axis_ratio": 0.0,
        "eccentricity": 0.0,
        "centroid_row": (rows - 1) / 2.0,
        "centroid_col": (cols - 1) / 2.0,
        "weighted_centroid_row": (rows - 1) / 2.0,
        "weighted_centroid_col": (cols - 1) / 2.0,
        "rms_radius": 0.0,
        "central_energy_fraction_r3": 0.0,
        "edge_energy_fraction": 0.0,
        "energy_entropy": 0.0,
    }


def _pca_stats(coords: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
    if coords.shape[0] < 3:
        return 0.0, 0.0, 0.0, 0.0, np.asarray([1.0, 0.0])
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    order = np.argsort(eigvals)
    minor = float(np.sqrt(eigvals[order[0]]))
    major = float(np.sqrt(eigvals[order[-1]]))
    ratio = major / max(minor, EPS)
    eccentricity = float(np.sqrt(max(0.0, 1.0 - eigvals[order[0]] / max(eigvals[order[-1]], EPS))))
    axis = eigvecs[:, order[-1]]
    return major, minor, ratio, eccentricity, axis


def extract_base_features(array: np.ndarray) -> dict[str, float]:
    x = np.asarray(array, dtype=np.float64)
    shape = x.shape
    mask = x > 0.0
    active_count = int(mask.sum())
    if active_count == 0:
        return _empty_base(shape)

    values = x[mask]
    rows, cols = np.nonzero(mask)
    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())
    bbox_height = row_max - row_min + 1
    bbox_width = col_max - col_min + 1
    bbox_area = bbox_width * bbox_height
    total = float(values.sum())
    centroid_row = float(rows.mean())
    centroid_col = float(cols.mean())
    weighted_row = float(np.sum(rows * values) / max(total, EPS))
    weighted_col = float(np.sum(cols * values) / max(total, EPS))
    coords = np.column_stack([rows.astype(float), cols.astype(float)])
    pca_major, pca_minor, pca_ratio, eccentricity, _ = _pca_stats(coords)
    distances_sq = (rows - weighted_row) ** 2 + (cols - weighted_col) ** 2
    rms_radius = float(np.sqrt(np.sum(values * distances_sq) / max(total, EPS)))
    distances = np.sqrt(distances_sq)
    central_energy = float(values[distances <= 3.0].sum())
    edge_mask = (rows == 0) | (cols == 0) | (rows == shape[0] - 1) | (cols == shape[1] - 1)
    edge_energy = float(values[edge_mask].sum())
    probs = values / max(total, EPS)
    probs = probs[probs > 0.0]
    entropy = float(-np.sum(probs * np.log(probs + EPS)))

    return {
        "active_count": active_count,
        "active_ratio": float(active_count / max(x.size, 1)),
        "active_sum": total,
        "active_mean": float(values.mean()),
        "active_std": float(values.std(ddof=0)),
        "active_max": float(values.max()),
        "bbox_width": float(bbox_width),
        "bbox_height": float(bbox_height),
        "bbox_area": float(bbox_area),
        "bbox_fill_ratio": float(active_count / max(bbox_area, 1)),
        "bbox_aspect_ratio": float(max(bbox_width, bbox_height) / max(min(bbox_width, bbox_height), 1)),
        "pca_major_length": pca_major,
        "pca_minor_length": pca_minor,
        "pca_axis_ratio": pca_ratio,
        "eccentricity": eccentricity,
        "centroid_row": centroid_row,
        "centroid_col": centroid_col,
        "weighted_centroid_row": weighted_row,
        "weighted_centroid_col": weighted_col,
        "rms_radius": rms_radius,
        "central_energy_fraction_r3": central_energy / max(total, EPS),
        "edge_energy_fraction": edge_energy / max(total, EPS),
        "energy_entropy": entropy,
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    if float(np.std(a)) <= EPS or float(np.std(b)) <= EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _gradient_along_pca(array: np.ndarray) -> float:
    x = np.asarray(array, dtype=np.float64)
    mask = x > 0.0
    if int(mask.sum()) < 3:
        return 0.0
    values = x[mask]
    rows, cols = np.nonzero(mask)
    coords = np.column_stack([rows.astype(float), cols.astype(float)])
    _, _, _, _, axis = _pca_stats(coords)
    centered = coords - coords.mean(axis=0, keepdims=True)
    projected = centered @ axis
    denom = float(np.sum((projected - projected.mean()) ** 2))
    if denom <= EPS:
        return 0.0
    return float(np.sum((projected - projected.mean()) * (values - values.mean())) / denom)


def extract_toa_features(toa_array: np.ndarray, tot_array: np.ndarray | None = None) -> dict[str, float]:
    toa = np.asarray(toa_array, dtype=np.float64)
    mask = toa > 0.0
    if int(mask.sum()) == 0:
        base = {name: 0.0 for name in TOA_FEATURES}
        return base
    values = toa[mask]
    result = {
        "toa_min_nonzero": float(values.min()),
        "toa_max_nonzero": float(values.max()),
        "toa_span": float(values.max() - values.min()),
        "toa_mean_nonzero": float(values.mean()),
        "toa_std_nonzero": float(values.std(ddof=0)),
        "toa_tot_corr": 0.0,
        "toa_gradient_along_pca": _gradient_along_pca(toa),
    }
    if tot_array is not None and np.asarray(tot_array).shape == toa.shape:
        tot = np.asarray(tot_array, dtype=np.float64)
        joint = (toa > 0.0) & (tot > 0.0)
        result["toa_tot_corr"] = _corr(toa[joint], tot[joint])
    return result


def paired_tot_lookup(index_df: pd.DataFrame) -> dict[tuple[str, str], Path]:
    subset = index_df[(index_df["modality"] == "ToT") & (index_df["status"] == "ok")]
    return {(str(row.dataset), str(row.sample_key)): Path(row.path) for row in subset.itertuples(index=False)}


def extract_feature_table(index_df: pd.DataFrame, dataset: str | None = None, modality: str | None = None) -> pd.DataFrame:
    subset = index_df[index_df["status"] == "ok"].copy()
    if dataset is not None:
        subset = subset[subset["dataset"] == dataset]
    if modality is not None:
        subset = subset[subset["modality"] == modality]
    tot_lookup = paired_tot_lookup(index_df)
    rows = []
    items = list(subset.itertuples(index=False))
    desc_parts = ["Features"]
    if dataset:
        desc_parts.append(str(dataset))
    if modality:
        desc_parts.append(str(modality))
    for item in iter_progress(items, total=len(items), desc=" ".join(desc_parts), unit="sample"):
        path = Path(item.path)
        try:
            array = read_matrix(path)
            features = extract_base_features(array)
            if item.modality == "ToA":
                tot_path = tot_lookup.get((str(item.dataset), str(item.sample_key)))
                tot_array = read_matrix(tot_path) if tot_path and tot_path.exists() else None
                features.update(extract_toa_features(array, tot_array))
            else:
                features.update({name: np.nan for name in TOA_FEATURES})
            rows.append(
                {
                    "dataset": item.dataset,
                    "angle": item.angle,
                    "angle_value": item.angle_value,
                    "modality": item.modality,
                    "sample_key": item.sample_key,
                    "path": item.path,
                    **features,
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep processing other samples
            rows.append(
                {
                    "dataset": item.dataset,
                    "angle": item.angle,
                    "angle_value": item.angle_value,
                    "modality": item.modality,
                    "sample_key": item.sample_key,
                    "path": item.path,
                    "feature_error": str(exc),
                }
            )
    return pd.DataFrame(rows, columns=FEATURE_TABLE_COLUMNS if not rows else None)


def feature_summary_by_angle(features: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    if feature_names is None:
        feature_names = [name for name in BASE_FEATURES + TOA_FEATURES if name in features.columns]
    rows = []
    for (dataset, modality, angle, angle_value), group in features.groupby(["dataset", "modality", "angle", "angle_value"]):
        row = {"dataset": dataset, "modality": modality, "angle": angle, "angle_value": angle_value, "count": len(group)}
        for name in feature_names:
            values = pd.to_numeric(group[name], errors="coerce")
            row[f"{name}_mean"] = values.mean()
            row[f"{name}_std"] = values.std(ddof=0)
            row[f"{name}_median"] = values.median()
            row[f"{name}_q25"] = values.quantile(0.25)
            row[f"{name}_q75"] = values.quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "modality", "angle_value"])
