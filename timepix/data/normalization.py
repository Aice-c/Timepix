"""Per-modality normalization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from .io import load_matrix


@dataclass
class ModalityStats:
    mean: float
    std: float
    min: float
    max: float
    log1p: bool
    ignore_zero: bool


class Normalizer:
    def __init__(self, stats: dict[str, ModalityStats], eps: float = 1e-6) -> None:
        self.stats = stats
        self.eps = eps

    def apply(self, tensor: torch.Tensor, modality: str) -> torch.Tensor:
        stats = self.stats.get(modality)
        if stats is None:
            return tensor
        x = tensor
        if stats.log1p:
            x = torch.log1p(torch.clamp(x, min=0.0))
        return (x - stats.mean) / max(stats.std, self.eps)


def center_crop_array(array: np.ndarray, crop_size: int) -> np.ndarray:
    if crop_size <= 0:
        return array
    height, width = array.shape
    if crop_size > min(height, width):
        raise ValueError(f"crop_size={crop_size} exceeds matrix shape {array.shape}")
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return array[top : top + crop_size, left : left + crop_size]


def compute_normalizer(
    records,
    modalities: list[str],
    normalization_config: Mapping[str, Mapping],
    crop_size: int,
    data_dtype: str = "float32",
) -> Normalizer | None:
    sums: dict[str, float] = {}
    sumsqs: dict[str, float] = {}
    counts: dict[str, int] = {}
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}

    enabled = [
        modality
        for modality in modalities
        if normalization_config.get(modality, {}).get("enabled", False)
    ]
    if not enabled:
        return None

    for modality in enabled:
        sums[modality] = 0.0
        sumsqs[modality] = 0.0
        counts[modality] = 0
        mins[modality] = math.inf
        maxs[modality] = -math.inf

    for record in records:
        for modality in enabled:
            cfg = normalization_config.get(modality, {})
            array = load_matrix(record.modalities[modality], data_dtype).astype(np.float64, copy=False)
            array = center_crop_array(array, crop_size)
            if cfg.get("log1p", False):
                array = np.log1p(np.maximum(array, 0.0))
            if cfg.get("ignore_zero", False):
                data = array[array != 0.0]
                if data.size == 0:
                    continue
            else:
                data = array.ravel()
            sums[modality] += float(np.sum(data))
            sumsqs[modality] += float(np.sum(data * data))
            counts[modality] += int(data.size)
            mins[modality] = min(mins[modality], float(np.min(data)))
            maxs[modality] = max(maxs[modality], float(np.max(data)))

    stats: dict[str, ModalityStats] = {}
    for modality in enabled:
        cfg = normalization_config.get(modality, {})
        n = counts[modality]
        if n == 0:
            stats[modality] = ModalityStats(0.0, 1.0, 0.0, 1.0, bool(cfg.get("log1p", False)), bool(cfg.get("ignore_zero", False)))
            continue
        mean = sums[modality] / n
        var = max(sumsqs[modality] / n - mean * mean, 0.0)
        stats[modality] = ModalityStats(
            mean=mean,
            std=max(math.sqrt(var), 1e-6),
            min=mins[modality],
            max=maxs[modality],
            log1p=bool(cfg.get("log1p", False)),
            ignore_zero=bool(cfg.get("ignore_zero", False)),
        )
    return Normalizer(stats)
