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
    "active_fraction",
    "active_sum",
    "active_mean",
    "active_std",
    "active_min",
    "active_max",
    "active_median",
    "active_q05",
    "active_q25",
    "active_q75",
    "active_q95",
    "tot_density_active",
    "tot_density_bbox",
    "bbox_x_min",
    "bbox_x_max",
    "bbox_y_min",
    "bbox_y_max",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_diagonal",
    "bbox_fill_ratio",
    "aspect_ratio",
    "centroid_x",
    "centroid_y",
    "weighted_centroid_x",
    "weighted_centroid_y",
    "centroid_offset_from_center",
    "touches_crop_boundary",
    "pca_major_axis",
    "pca_minor_axis",
    "pca_axis_ratio",
    "pca_eccentricity",
    "pca_orientation",
    "weighted_pca_major_axis",
    "weighted_pca_minor_axis",
    "weighted_pca_axis_ratio",
    "weighted_pca_eccentricity",
    "radius_mean",
    "radius_std",
    "radius_max",
    "weighted_radius_mean",
    "weighted_radius_std",
    "central_energy_ratio_r1",
    "central_energy_ratio_r2",
    "central_energy_ratio_r3",
    "radial_bin_0_sum",
    "radial_bin_1_sum",
    "radial_bin_2_sum",
    "radial_bin_3_sum",
    "hu_moment_1",
    "hu_moment_2",
    "hu_moment_3",
    "hu_moment_4",
    "hu_moment_5",
    "hu_moment_6",
    "hu_moment_7",
    "horizontal_profile_std",
    "vertical_profile_std",
    "max_row_sum",
    "max_col_sum",
    "row_entropy",
    "col_entropy",
    "spatial_entropy",
    "gradient_mean",
    "gradient_std",
    "gradient_max",
    # Backward-compatible aliases used by older helper plots.
    "active_ratio",
    "active_sum_alias",
    "active_mean_alias",
    "active_std_alias",
    "bbox_aspect_ratio",
    "pca_major_length",
    "pca_minor_length",
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
    "toa_active_count",
    "toa_valid_fraction",
    "toa_min",
    "toa_max",
    "toa_span",
    "toa_mean",
    "toa_std",
    "toa_median",
    "toa_q25",
    "toa_q75",
    "toa_relative_span",
    "toa_major_axis_corr",
    "toa_major_axis_corr_abs",
    "toa_weighted_major_axis_corr",
]

QUALITY_FEATURES = [
    "input_shape_h",
    "input_shape_w",
    "matrix_dtype",
    "matrix_min",
    "matrix_max",
    "num_nan_values",
    "num_inf_values",
    "num_negative_values",
    "is_all_zero",
]

FEATURE_TABLE_COLUMNS = [
    "dataset",
    "sample_id",
    "angle",
    "angle_value",
    "modality",
    "split",
    "source_path",
    *QUALITY_FEATURES,
    *BASE_FEATURES,
    *TOA_FEATURES,
    "feature_error",
]


def _safe_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    total = float(np.nansum(values))
    if total <= EPS:
        return 0.0
    probs = values / total
    probs = probs[np.isfinite(probs) & (probs > 0.0)]
    return float(-np.sum(probs * np.log(probs + EPS)))


def _quality_features(array: np.ndarray) -> dict[str, float | str | bool]:
    x = np.asarray(array)
    finite = np.isfinite(x)
    finite_values = x[finite]
    return {
        "input_shape_h": int(x.shape[0]) if x.ndim >= 1 else 0,
        "input_shape_w": int(x.shape[1]) if x.ndim >= 2 else 0,
        "matrix_dtype": str(x.dtype),
        "matrix_min": float(np.min(finite_values)) if finite_values.size else np.nan,
        "matrix_max": float(np.max(finite_values)) if finite_values.size else np.nan,
        "num_nan_values": int(np.isnan(x).sum()),
        "num_inf_values": int(np.isinf(x).sum()),
        "num_negative_values": int((finite_values < 0).sum()) if finite_values.size else 0,
        "is_all_zero": bool(finite_values.size == 0 or np.all(finite_values == 0)),
    }


def _pca_stats(coords_xy: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float, float, float, float, np.ndarray]:
    if coords_xy.shape[0] < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0, np.asarray([1.0, 0.0])
    if weights is None:
        center = coords_xy.mean(axis=0, keepdims=True)
        centered = coords_xy - center
        cov = np.cov(centered.T)
    else:
        w = np.asarray(weights, dtype=float)
        w = np.clip(w, 0.0, None)
        total = float(w.sum())
        if total <= EPS:
            return _pca_stats(coords_xy, None)
        center = (coords_xy * w[:, None]).sum(axis=0, keepdims=True) / total
        centered = coords_xy - center
        cov = (centered * w[:, None]).T @ centered / max(total, EPS)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    order = np.argsort(eigvals)
    minor = float(np.sqrt(eigvals[order[0]]))
    major = float(np.sqrt(eigvals[order[-1]]))
    ratio = major / max(minor, EPS)
    eccentricity = float(np.sqrt(max(0.0, 1.0 - eigvals[order[0]] / max(eigvals[order[-1]], EPS))))
    axis = eigvecs[:, order[-1]]
    orientation = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return major, minor, ratio, eccentricity, orientation, axis


def _profile_features(x: np.ndarray) -> dict[str, float]:
    row_sum = np.nansum(np.clip(x, 0, None), axis=1)
    col_sum = np.nansum(np.clip(x, 0, None), axis=0)
    return {
        "horizontal_profile_std": float(np.std(col_sum, ddof=0)),
        "vertical_profile_std": float(np.std(row_sum, ddof=0)),
        "max_row_sum": float(np.max(row_sum)) if row_sum.size else 0.0,
        "max_col_sum": float(np.max(col_sum)) if col_sum.size else 0.0,
        "row_entropy": _safe_entropy(row_sum),
        "col_entropy": _safe_entropy(col_sum),
    }


def _hu_moments(x: np.ndarray) -> dict[str, float]:
    values = np.asarray(x, dtype=float)
    values = np.where(np.isfinite(values), np.clip(values, 0.0, None), 0.0)
    total = float(values.sum())
    if total <= EPS:
        return {f"hu_moment_{idx}": 0.0 for idx in range(1, 8)}
    yy, xx = np.indices(values.shape)
    x_bar = float((xx * values).sum() / total)
    y_bar = float((yy * values).sum() / total)

    def mu(p: int, q: int) -> float:
        return float((((xx - x_bar) ** p) * ((yy - y_bar) ** q) * values).sum())

    def eta(p: int, q: int) -> float:
        return mu(p, q) / (total ** (1.0 + (p + q) / 2.0) + EPS)

    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n12, n21, n03 = eta(3, 0), eta(1, 2), eta(2, 1), eta(0, 3)
    hu = [
        n20 + n02,
        (n20 - n02) ** 2 + 4 * n11**2,
        (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2,
        (n30 + n12) ** 2 + (n21 + n03) ** 2,
        (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
        + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
        (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03),
        (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
        - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
    ]
    return {f"hu_moment_{idx}": float(value) for idx, value in enumerate(hu, start=1)}


def _empty_base(shape: tuple[int, int]) -> dict[str, float]:
    h, w = shape
    center_x = (w - 1) / 2.0
    center_y = (h - 1) / 2.0
    base = {name: 0.0 for name in BASE_FEATURES}
    base.update(
        {
            "centroid_x": center_x,
            "centroid_y": center_y,
            "weighted_centroid_x": center_x,
            "weighted_centroid_y": center_y,
            "centroid_col": center_x,
            "centroid_row": center_y,
            "weighted_centroid_col": center_x,
            "weighted_centroid_row": center_y,
        }
    )
    return base


def extract_base_features(array: np.ndarray) -> dict[str, float]:
    x = np.asarray(array, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    shape = x.shape
    if x.ndim != 2:
        return _empty_base((0, 0))
    mask = x > 0.0
    active_count = int(mask.sum())
    if active_count == 0:
        features = _empty_base(shape)
        features.update(_profile_features(x))
        features.update(_hu_moments(x))
        return features

    values = x[mask]
    yy, xx = np.nonzero(mask)
    y_min, y_max = int(yy.min()), int(yy.max())
    x_min, x_max = int(xx.min()), int(xx.max())
    bbox_height = y_max - y_min + 1
    bbox_width = x_max - x_min + 1
    bbox_area = bbox_width * bbox_height
    total = float(values.sum())
    coords_xy = np.column_stack([xx.astype(float), yy.astype(float)])
    center_x = (shape[1] - 1) / 2.0
    center_y = (shape[0] - 1) / 2.0
    centroid_x = float(xx.mean())
    centroid_y = float(yy.mean())
    weighted_x = float(np.sum(xx * values) / max(total, EPS))
    weighted_y = float(np.sum(yy * values) / max(total, EPS))
    pca_major, pca_minor, pca_ratio, pca_ecc, pca_orientation, _ = _pca_stats(coords_xy)
    wpca_major, wpca_minor, wpca_ratio, wpca_ecc, _, _ = _pca_stats(coords_xy, values)

    dx = xx - centroid_x
    dy = yy - centroid_y
    radius = np.sqrt(dx**2 + dy**2)
    wdx = xx - weighted_x
    wdy = yy - weighted_y
    wradius = np.sqrt(wdx**2 + wdy**2)
    radial_edges = np.quantile(radius, [0.25, 0.5, 0.75]) if radius.size else [0, 0, 0]
    radial_bins = [
        values[radius <= radial_edges[0]].sum(),
        values[(radius > radial_edges[0]) & (radius <= radial_edges[1])].sum(),
        values[(radius > radial_edges[1]) & (radius <= radial_edges[2])].sum(),
        values[radius > radial_edges[2]].sum(),
    ]
    gradient_y, gradient_x = np.gradient(x)
    gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
    edge_mask = (yy == 0) | (xx == 0) | (yy == shape[0] - 1) | (xx == shape[1] - 1)
    features = {
        "active_count": active_count,
        "active_fraction": float(active_count / max(x.size, 1)),
        "active_sum": total,
        "active_mean": float(values.mean()),
        "active_std": float(values.std(ddof=0)),
        "active_min": float(values.min()),
        "active_max": float(values.max()),
        "active_median": float(np.median(values)),
        "active_q05": float(np.quantile(values, 0.05)),
        "active_q25": float(np.quantile(values, 0.25)),
        "active_q75": float(np.quantile(values, 0.75)),
        "active_q95": float(np.quantile(values, 0.95)),
        "tot_density_active": float(total / max(active_count, 1)),
        "tot_density_bbox": float(total / max(bbox_area, 1)),
        "bbox_x_min": float(x_min),
        "bbox_x_max": float(x_max),
        "bbox_y_min": float(y_min),
        "bbox_y_max": float(y_max),
        "bbox_width": float(bbox_width),
        "bbox_height": float(bbox_height),
        "bbox_area": float(bbox_area),
        "bbox_diagonal": float(np.sqrt(bbox_width**2 + bbox_height**2)),
        "bbox_fill_ratio": float(active_count / max(bbox_area, 1)),
        "aspect_ratio": float(max(bbox_width, bbox_height) / max(min(bbox_width, bbox_height), 1)),
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "weighted_centroid_x": weighted_x,
        "weighted_centroid_y": weighted_y,
        "centroid_offset_from_center": float(np.sqrt((weighted_x - center_x) ** 2 + (weighted_y - center_y) ** 2)),
        "touches_crop_boundary": float(edge_mask.any()),
        "pca_major_axis": pca_major,
        "pca_minor_axis": pca_minor,
        "pca_axis_ratio": pca_ratio,
        "pca_eccentricity": pca_ecc,
        "pca_orientation": pca_orientation,
        "weighted_pca_major_axis": wpca_major,
        "weighted_pca_minor_axis": wpca_minor,
        "weighted_pca_axis_ratio": wpca_ratio,
        "weighted_pca_eccentricity": wpca_ecc,
        "radius_mean": float(radius.mean()),
        "radius_std": float(radius.std(ddof=0)),
        "radius_max": float(radius.max()),
        "weighted_radius_mean": float(np.sum(values * wradius) / max(total, EPS)),
        "weighted_radius_std": float(np.sqrt(np.sum(values * (wradius - np.sum(values * wradius) / max(total, EPS)) ** 2) / max(total, EPS))),
        "central_energy_ratio_r1": float(values[wradius <= 1.0].sum() / max(total, EPS)),
        "central_energy_ratio_r2": float(values[wradius <= 2.0].sum() / max(total, EPS)),
        "central_energy_ratio_r3": float(values[wradius <= 3.0].sum() / max(total, EPS)),
        "radial_bin_0_sum": float(radial_bins[0]),
        "radial_bin_1_sum": float(radial_bins[1]),
        "radial_bin_2_sum": float(radial_bins[2]),
        "radial_bin_3_sum": float(radial_bins[3]),
        "gradient_mean": float(gradient_mag[mask].mean()),
        "gradient_std": float(gradient_mag[mask].std(ddof=0)),
        "gradient_max": float(gradient_mag[mask].max()),
        "edge_energy_fraction": float(values[edge_mask].sum() / max(total, EPS)),
        "spatial_entropy": _safe_entropy(values),
        "energy_entropy": _safe_entropy(values),
    }
    features.update(_profile_features(x))
    features.update(_hu_moments(x))
    features.update(
        {
            "active_ratio": features["active_fraction"],
            "active_sum_alias": features["active_sum"],
            "active_mean_alias": features["active_mean"],
            "active_std_alias": features["active_std"],
            "bbox_aspect_ratio": features["aspect_ratio"],
            "pca_major_length": features["pca_major_axis"],
            "pca_minor_length": features["pca_minor_axis"],
            "eccentricity": features["pca_eccentricity"],
            "centroid_row": features["centroid_y"],
            "centroid_col": features["centroid_x"],
            "weighted_centroid_row": features["weighted_centroid_y"],
            "weighted_centroid_col": features["weighted_centroid_x"],
            "rms_radius": features["weighted_radius_mean"],
            "central_energy_fraction_r3": features["central_energy_ratio_r3"],
        }
    )
    return features


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a = a[keep]
    b = b[keep]
    if a.size < 2 or b.size < 2:
        return 0.0
    if float(np.std(a)) <= EPS or float(np.std(b)) <= EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _major_axis_corr(array: np.ndarray, *, weighted: bool = False) -> float:
    x = np.asarray(array, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    mask = x > 0.0
    if int(mask.sum()) < 3:
        return 0.0
    values = x[mask]
    yy, xx = np.nonzero(mask)
    coords_xy = np.column_stack([xx.astype(float), yy.astype(float)])
    _, _, _, _, _, axis = _pca_stats(coords_xy, values if weighted else None)
    projected = (coords_xy - coords_xy.mean(axis=0, keepdims=True)) @ axis
    return _corr(projected, values)


def extract_toa_features(toa_array: np.ndarray) -> dict[str, float]:
    toa = np.asarray(toa_array, dtype=np.float64)
    toa = np.where(np.isfinite(toa), toa, 0.0)
    mask = toa > 0.0
    result = {name: np.nan for name in TOA_FEATURES}
    result["toa_active_count"] = int(mask.sum())
    result["toa_valid_fraction"] = float(mask.sum() / max(toa.size, 1))
    if int(mask.sum()) == 0:
        return result
    values = toa[mask]
    span = float(values.max() - values.min())
    corr = _major_axis_corr(toa, weighted=False)
    weighted_corr = _major_axis_corr(toa, weighted=True)
    result.update(
        {
            "toa_min": float(values.min()),
            "toa_max": float(values.max()),
            "toa_span": span,
            "toa_mean": float(values.mean()),
            "toa_std": float(values.std(ddof=0)),
            "toa_median": float(np.median(values)),
            "toa_q25": float(np.quantile(values, 0.25)),
            "toa_q75": float(np.quantile(values, 0.75)),
            "toa_relative_span": float(span / max(abs(float(values.mean())), EPS)),
            "toa_major_axis_corr": corr,
            "toa_major_axis_corr_abs": abs(corr),
            "toa_weighted_major_axis_corr": weighted_corr,
        }
    )
    return result


def paired_tot_lookup(index_df: pd.DataFrame) -> dict[tuple[str, str], Path]:
    subset = index_df[(index_df["modality"] == "ToT") & (index_df["status"].isin(["ok", "unknown"]))]
    return {(str(row.dataset), str(row.sample_key)): Path(row.path) for row in subset.itertuples(index=False)}


def extract_feature_table(
    index_df: pd.DataFrame,
    dataset: str | None = None,
    modality: str | None = None,
    *,
    split_lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    subset = index_df[index_df["status"].isin(["ok", "unknown"])].copy()
    if dataset is not None:
        subset = subset[subset["dataset"] == dataset]
    if modality is not None:
        subset = subset[subset["modality"] == modality]
    rows = []
    items = list(subset.itertuples(index=False))
    desc_parts = ["Features"]
    if dataset:
        desc_parts.append(str(dataset))
    if modality:
        desc_parts.append(str(modality))
    for item in iter_progress(items, total=len(items), desc=" ".join(desc_parts), unit="sample"):
        path = Path(item.path)
        sample_id = str(item.sample_key)
        try:
            array = read_matrix(path)
            features = _quality_features(array)
            features.update(extract_base_features(array))
            if item.modality == "ToA":
                features.update(extract_toa_features(array))
            else:
                features.update({name: np.nan for name in TOA_FEATURES})
            rows.append(
                {
                    "dataset": item.dataset,
                    "sample_id": sample_id,
                    "angle": item.angle,
                    "angle_value": item.angle_value,
                    "modality": item.modality,
                    "split": split_lookup.get(sample_id, "unknown") if split_lookup else "unknown",
                    "source_path": item.path,
                    **features,
                    "feature_error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep processing other samples
            rows.append(
                {
                    "dataset": item.dataset,
                    "sample_id": sample_id,
                    "angle": item.angle,
                    "angle_value": item.angle_value,
                    "modality": item.modality,
                    "split": split_lookup.get(sample_id, "unknown") if split_lookup else "unknown",
                    "source_path": item.path,
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
