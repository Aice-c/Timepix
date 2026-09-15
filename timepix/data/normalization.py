"""Per-modality normalization."""

from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from .io import load_matrix
from .transforms import apply_modality_transform, normalize_toa_transform


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


def load_frozen_normalizer(metadata_path, modalities, normalization_config, split_path, dataset_cfg, data_cfg):
    """Reuse audited training statistics; a provenance mismatch never triggers a refit."""
    metadata_path, split_path = Path(metadata_path), Path(split_path)
    raw = metadata_path.read_bytes()
    metadata = json.loads(raw)
    info = metadata['data_info']
    with split_path.open('rb') as stream:
        split_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    if split_hash != metadata.get('split_manifest_hash'):
        raise ValueError('Frozen normalizer split hash mismatch')
    if info['class_names'] != dataset_cfg.get('class_names') or info['modalities'] != modalities:
        raise ValueError('Frozen normalizer classes/modalities mismatch')
    if metadata['dataset']['name'] != dataset_cfg['name']:
        raise ValueError('Frozen normalizer dataset identity mismatch')
    for name, default in [('crop_size', 0), ('dtype', 'float32'), ('input_representation', 'signal'),
                          ('toa_transform', 'none'), ('add_hit_mask', False)]:
        if metadata.get('data', {}).get(name, default) != data_cfg.get(name, default):
            raise ValueError(f'Frozen normalizer preprocessing mismatch: {name}')
    stats = {}
    for modality in modalities:
        config = normalization_config.get(modality, {})
        if not config.get('enabled', False):
            raise ValueError('Frozen normalizer requires enabled original modalities')
        values = ModalityStats(**info['normalizer_stats'][modality])
        if any(getattr(values, key) != bool(config.get(key, False)) for key in ('log1p', 'ignore_zero')):
            raise ValueError('Frozen normalizer transform/statistic mask mismatch')
        if not math.isfinite(values.mean) or not math.isfinite(values.std) or values.std <= 0:
            raise ValueError('Invalid frozen normalizer mean/std')
        stats[modality] = values
    return Normalizer(stats), dict(metadata_path=str(metadata_path),
        metadata_sha256=hashlib.sha256(raw).hexdigest(), split_sha256=split_hash, refitted=False)


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
    toa_transform: str | None = None,
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
    toa_transform = normalize_toa_transform(toa_transform)

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
            array = apply_modality_transform(modality, array, toa_transform).astype(np.float64, copy=False)
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
