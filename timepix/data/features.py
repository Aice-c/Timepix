"""Handcrafted feature extraction for training experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import torch

from .io import load_matrix
from .normalization import center_crop_array


EPS = 1e-6
FeatureFn = Callable[[Mapping[str, np.ndarray]], float]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    required_modalities: tuple[str, ...]
    fn: FeatureFn


def _as_float_array(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def _hit_mask(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if "ToT" in arrays:
        return _as_float_array(arrays["ToT"]) > 0.0
    if "ToA" in arrays:
        return _as_float_array(arrays["ToA"]) > 0.0
    raise KeyError("Handcrafted geometry features require ToT or ToA")


def _tot(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if "ToT" not in arrays:
        raise KeyError("This handcrafted feature requires ToT")
    return _as_float_array(arrays["ToT"])


def _toa(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if "ToA" not in arrays:
        raise KeyError("This handcrafted feature requires ToA")
    return _as_float_array(arrays["ToA"])


def _coords_from_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.stack([cols.astype(np.float64), rows.astype(np.float64)], axis=1)


def _bbox(mask: np.ndarray) -> tuple[float, float, float]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return 0.0, 0.0, 0.0
    width = float(cols.max() - cols.min() + 1)
    height = float(rows.max() - rows.min() + 1)
    area = width * height
    return width, height, area


def _pca_axes(mask: np.ndarray) -> tuple[float, float, np.ndarray | None, np.ndarray | None]:
    coords = _coords_from_mask(mask)
    if coords.shape[0] < 2:
        return 0.0, 0.0, None, None
    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False, bias=True)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    projected = centered @ eigvecs
    major = float(projected[:, 0].max() - projected[:, 0].min())
    minor = float(projected[:, 1].max() - projected[:, 1].min())
    return major, minor, eigvecs[:, 0], coords.mean(axis=0)


def _active_pixel_count(arrays: Mapping[str, np.ndarray]) -> float:
    return float(np.count_nonzero(_hit_mask(arrays)))


def _bbox_long(arrays: Mapping[str, np.ndarray]) -> float:
    width, height, _ = _bbox(_hit_mask(arrays))
    return float(max(width, height))


def _bbox_short(arrays: Mapping[str, np.ndarray]) -> float:
    width, height, _ = _bbox(_hit_mask(arrays))
    return float(min(width, height))


def _bbox_fill_ratio(arrays: Mapping[str, np.ndarray]) -> float:
    mask = _hit_mask(arrays)
    _, _, area = _bbox(mask)
    if area <= 0.0:
        return 0.0
    return float(np.count_nonzero(mask) / (area + EPS))


def _pca_major_axis(arrays: Mapping[str, np.ndarray]) -> float:
    major, _, _, _ = _pca_axes(_hit_mask(arrays))
    return major


def _pca_minor_axis(arrays: Mapping[str, np.ndarray]) -> float:
    _, minor, _, _ = _pca_axes(_hit_mask(arrays))
    return minor


def _total_tot(arrays: Mapping[str, np.ndarray]) -> float:
    return float(np.sum(_tot(arrays)))


def _tot_density(arrays: Mapping[str, np.ndarray]) -> float:
    mask = _hit_mask(arrays)
    _, _, area = _bbox(mask)
    if area <= 0.0:
        return 0.0
    return float(np.sum(_tot(arrays)) / (area + EPS))


def _toa_values(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    values = _toa(arrays)
    return values[values > 0.0].astype(np.float64, copy=False)


def _toa_span(arrays: Mapping[str, np.ndarray]) -> float:
    values = _toa_values(arrays)
    if values.size <= 1:
        return 0.0
    return float(values.max() - values.min())


def _toa_p90_minus_p10(arrays: Mapping[str, np.ndarray]) -> float:
    values = _toa_values(arrays)
    if values.size <= 1:
        return 0.0
    return float(np.percentile(values, 90) - np.percentile(values, 10))


def _axis_time_values(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    tot = _tot(arrays)
    toa = _toa(arrays)
    if tot.shape != toa.shape:
        raise ValueError(f"ToT and ToA shapes must match for axis interaction features, got {tot.shape} and {toa.shape}")

    hit_mask = tot > 0.0
    _, _, axis, center = _pca_axes(hit_mask)
    if axis is None or center is None:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    valid = hit_mask & (toa > 0.0)
    coords = _coords_from_mask(valid)
    if coords.shape[0] < 2:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    projection = (coords - center) @ axis
    times = toa[valid].astype(np.float64, copy=False)
    return projection, times


def _toa_major_axis_slope_abs(arrays: Mapping[str, np.ndarray]) -> float:
    projection, times = _axis_time_values(arrays)
    if projection.size < 2:
        return 0.0
    x = projection - projection.mean()
    y = times - times.mean()
    denom = float(np.sum(x * x))
    if denom <= EPS:
        return 0.0
    return float(abs(np.sum(x * y) / (denom + EPS)))


def _toa_major_axis_corr_abs(arrays: Mapping[str, np.ndarray]) -> float:
    projection, times = _axis_time_values(arrays)
    if projection.size < 2:
        return 0.0
    x = projection - projection.mean()
    y = times - times.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= EPS:
        return 0.0
    return float(abs(np.sum(x * y) / (denom + EPS)))


def _legacy_total_energy_for(modality: str) -> FeatureSpec:
    def _fn(arrays: Mapping[str, np.ndarray]) -> float:
        if modality not in arrays:
            raise KeyError(f"Legacy total_energy feature requires {modality}")
        return float(np.sum(_as_float_array(arrays[modality])))

    return FeatureSpec(name=f"{modality}.total_energy", required_modalities=(modality,), fn=_fn)


FEATURE_REGISTRY: dict[str, FeatureSpec] = {
    "active_pixel_count": FeatureSpec("active_pixel_count", ("ToT",), _active_pixel_count),
    "bbox_long": FeatureSpec("bbox_long", ("ToT",), _bbox_long),
    "bbox_short": FeatureSpec("bbox_short", ("ToT",), _bbox_short),
    "bbox_fill_ratio": FeatureSpec("bbox_fill_ratio", ("ToT",), _bbox_fill_ratio),
    "pca_major_axis": FeatureSpec("pca_major_axis", ("ToT",), _pca_major_axis),
    "pca_minor_axis": FeatureSpec("pca_minor_axis", ("ToT",), _pca_minor_axis),
    "total_ToT": FeatureSpec("total_ToT", ("ToT",), _total_tot),
    "ToT_density": FeatureSpec("ToT_density", ("ToT",), _tot_density),
    "ToA_span": FeatureSpec("ToA_span", ("ToA",), _toa_span),
    "ToA_p90_minus_p10": FeatureSpec("ToA_p90_minus_p10", ("ToA",), _toa_p90_minus_p10),
    "ToA_major_axis_slope_abs": FeatureSpec("ToA_major_axis_slope_abs", ("ToT", "ToA"), _toa_major_axis_slope_abs),
    "ToA_major_axis_corr_abs": FeatureSpec("ToA_major_axis_corr_abs", ("ToT", "ToA"), _toa_major_axis_corr_abs),
    # Backward-compatible unscoped feature name. Prefer total_ToT in new configs.
    "total_energy": FeatureSpec("total_energy", ("ToT",), _total_tot),
}

TOT_ONLY_A5_FEATURES = [
    "active_pixel_count",
    "bbox_long",
    "bbox_short",
    "bbox_fill_ratio",
    "pca_major_axis",
    "pca_minor_axis",
    "total_ToT",
    "ToT_density",
]

ALPHA_A5_FEATURES = [
    *TOT_ONLY_A5_FEATURES,
    "ToA_span",
    "ToA_p90_minus_p10",
    "ToA_major_axis_slope_abs",
    "ToA_major_axis_corr_abs",
]


def _normalize_feature_names(names) -> list[str]:
    if names is None:
        return []
    if isinstance(names, Mapping):
        return [str(name) for name, use in names.items() if use]
    return [str(name) for name in names]


def parse_feature_config(config: Mapping | None, modalities: list[str]) -> list[FeatureSpec]:
    if not config or not config.get("enabled", False):
        return []

    raw_features = config.get("features", {})
    enabled: list[FeatureSpec] = []

    if isinstance(raw_features, Mapping):
        # Backward-compatible modality-scoped form:
        # features:
        #   ToT: [total_energy]
        modality_keys = set(raw_features).intersection({"ToT", "ToA"})
        if modality_keys:
            for modality in raw_features:
                names = _normalize_feature_names(raw_features.get(modality, []))
                for name in names:
                    if name == "total_energy":
                        enabled.append(_legacy_total_energy_for(str(modality)))
                    elif name in FEATURE_REGISTRY:
                        spec = FEATURE_REGISTRY[name]
                        enabled.append(spec)
                    else:
                        raise ValueError(f"Unknown handcrafted feature: {name}")
            return enabled

        # Named groups are flattened in declaration order:
        # features:
        #   selected_geometry: [...]
        for names in raw_features.values():
            for name in _normalize_feature_names(names):
                if name not in FEATURE_REGISTRY:
                    raise ValueError(f"Unknown handcrafted feature: {name}")
                enabled.append(FEATURE_REGISTRY[name])
        return enabled

    for name in _normalize_feature_names(raw_features):
        if name not in FEATURE_REGISTRY:
            raise ValueError(f"Unknown handcrafted feature: {name}")
        enabled.append(FEATURE_REGISTRY[name])
    return enabled


@dataclass
class HandcraftedFeatureExtractor:
    enabled_features: list[FeatureSpec]

    @property
    def dim(self) -> int:
        return len(self.enabled_features)

    @property
    def feature_names(self) -> list[str]:
        return [feature.name for feature in self.enabled_features]

    @property
    def required_modalities(self) -> list[str]:
        seen: list[str] = []
        for feature in self.enabled_features:
            for modality in feature.required_modalities:
                if modality not in seen:
                    seen.append(modality)
        return seen

    def extract(self, arrays: Mapping[str, np.ndarray]) -> torch.Tensor | None:
        if not self.enabled_features:
            return None
        values = [feature.fn(arrays) for feature in self.enabled_features]
        return torch.tensor(values, dtype=torch.float32)


@dataclass
class HandcraftedFeatureScaler:
    means: torch.Tensor
    stds: torch.Tensor
    feature_names: list[str]

    def apply(self, features: torch.Tensor) -> torch.Tensor:
        stds = torch.clamp(self.stds.float(), min=1e-6)
        if features.dim() == 1:
            return (features - self.means.float()) / stds
        return (features - self.means.float().unsqueeze(0)) / stds.unsqueeze(0)


def load_feature_arrays(record, modalities: list[str], data_dtype: str = "float32", crop_size: int = 0) -> dict[str, np.ndarray]:
    return {
        modality: center_crop_array(load_matrix(record.modalities[modality], data_dtype), crop_size)
        for modality in modalities
    }


def compute_feature_scaler(
    records,
    extractor: HandcraftedFeatureExtractor,
    data_dtype: str = "float32",
    crop_size: int = 0,
) -> HandcraftedFeatureScaler | None:
    if extractor.dim == 0:
        return None

    values = []
    needed_modalities = extractor.required_modalities
    for record in records:
        arrays = load_feature_arrays(record, needed_modalities, data_dtype=data_dtype, crop_size=crop_size)
        vec = extractor.extract(arrays)
        if vec is not None:
            values.append(vec.numpy())

    if not values:
        means = torch.zeros(extractor.dim, dtype=torch.float32)
        stds = torch.ones(extractor.dim, dtype=torch.float32)
    else:
        stacked = np.stack(values, axis=0).astype(np.float64)
        means = torch.tensor(stacked.mean(axis=0), dtype=torch.float32)
        stds = torch.tensor(stacked.std(axis=0), dtype=torch.float32)

    return HandcraftedFeatureScaler(means, stds, extractor.feature_names)
