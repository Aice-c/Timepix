"""Handcrafted feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from .io import load_matrix


FeatureList = list[tuple[str, str]]


def _total_energy(array: np.ndarray) -> float:
    return float(np.sum(array))


FEATURE_REGISTRY = {
    "total_energy": _total_energy,
}


def parse_feature_config(config: Mapping | None, modalities: list[str]) -> FeatureList:
    if not config or not config.get("enabled", False):
        return []

    raw_features = config.get("features", {})
    enabled: FeatureList = []
    for modality in modalities:
        names = raw_features.get(modality, [])
        if isinstance(names, Mapping):
            names = [name for name, use in names.items() if use]
        for name in names:
            if name not in FEATURE_REGISTRY:
                raise ValueError(f"Unknown handcrafted feature: {name}")
            enabled.append((modality, name))
    return enabled


@dataclass
class HandcraftedFeatureExtractor:
    enabled_features: FeatureList

    @property
    def dim(self) -> int:
        return len(self.enabled_features)

    def extract(self, arrays: Mapping[str, np.ndarray]) -> torch.Tensor | None:
        if not self.enabled_features:
            return None
        values = []
        for modality, name in self.enabled_features:
            if modality not in arrays:
                raise KeyError(f"Missing modality for handcrafted feature: {modality}")
            values.append(FEATURE_REGISTRY[name](arrays[modality]))
        return torch.tensor(values, dtype=torch.float32)


@dataclass
class HandcraftedFeatureScaler:
    means: torch.Tensor
    stds: torch.Tensor
    feature_names: FeatureList

    def apply(self, features: torch.Tensor) -> torch.Tensor:
        stds = torch.clamp(self.stds.float(), min=1e-6)
        if features.dim() == 1:
            return (features - self.means.float()) / stds
        return (features - self.means.float().unsqueeze(0)) / stds.unsqueeze(0)


def compute_feature_scaler(records, extractor: HandcraftedFeatureExtractor, data_dtype: str = "float32") -> HandcraftedFeatureScaler | None:
    if extractor.dim == 0:
        return None

    values = []
    needed_modalities = sorted({m for m, _ in extractor.enabled_features})
    for record in records:
        arrays = {
            modality: load_matrix(record.modalities[modality], data_dtype)
            for modality in needed_modalities
        }
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

    return HandcraftedFeatureScaler(means, stds, list(extractor.enabled_features))
