"""Data transforms shared by datasets and normalization."""

from __future__ import annotations

import numpy as np


SUPPORTED_TOA_TRANSFORMS = {
    "none",
    "raw_log1p",
    "relative_minmax",
    "relative_centered",
    "relative_rank",
}


def normalize_toa_transform(name: str | None) -> str:
    transform = str(name or "none").strip().lower()
    if transform not in SUPPORTED_TOA_TRANSFORMS:
        supported = ", ".join(sorted(SUPPORTED_TOA_TRANSFORMS))
        raise ValueError(f"data.toa_transform must be one of: {supported}")
    return transform


def apply_toa_transform(array: np.ndarray, transform: str | None, eps: float = 1e-6) -> np.ndarray:
    """Apply a sample-wise ToA transform while keeping background pixels at zero."""

    transform = normalize_toa_transform(transform)
    x = np.asarray(array, dtype=np.float32)
    if transform == "none":
        return x
    if transform == "raw_log1p":
        return np.log1p(np.maximum(x, 0.0)).astype(np.float32, copy=False)

    mask = x > 0.0
    out = np.zeros_like(x, dtype=np.float32)
    values = x[mask].astype(np.float64, copy=False)
    if values.size <= 1:
        return out

    if transform == "relative_minmax":
        low = float(values.min())
        high = float(values.max())
        denom = high - low
        if denom <= eps:
            return out
        out[mask] = ((values - low) / (denom + eps)).astype(np.float32, copy=False)
        return out

    if transform == "relative_centered":
        mean = float(values.mean())
        std = float(values.std())
        if std <= eps:
            return out
        out[mask] = ((values - mean) / (std + eps)).astype(np.float32, copy=False)
        return out

    if transform == "relative_rank":
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.shape[0], dtype=np.float64)
        ranks[order] = np.arange(values.shape[0], dtype=np.float64) / max(values.shape[0] - 1, 1)
        out[mask] = ranks.astype(np.float32, copy=False)
        return out

    raise AssertionError(f"Unhandled ToA transform: {transform}")


def apply_modality_transform(modality: str, array: np.ndarray, toa_transform: str | None) -> np.ndarray:
    if modality == "ToA":
        return apply_toa_transform(array, toa_transform)
    return np.asarray(array)


def make_hit_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    mask = None
    for array in arrays.values():
        current = np.asarray(array) > 0.0
        mask = current if mask is None else np.logical_or(mask, current)
    if mask is None:
        raise ValueError("add_hit_mask requires at least one image modality")
    return mask.astype(np.float32, copy=False)
