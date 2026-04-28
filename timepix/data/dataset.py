"""Dataset implementation for Timepix text matrices."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import HandcraftedFeatureExtractor, HandcraftedFeatureScaler
from .io import load_matrix
from .normalization import Normalizer, center_crop_array
from .transforms import apply_modality_transform, make_hit_mask, normalize_toa_transform


@dataclass(frozen=True)
class SampleRecord:
    label: int
    angle: str
    key: str
    modalities: dict[str, Path]


def _numeric_angle(name: str) -> float:
    try:
        return float(name)
    except ValueError as exc:
        raise ValueError(f"Angle folder name must be numeric: {name}") from exc


def _list_files(directory: Path) -> list[str]:
    return sorted(name for name in os.listdir(directory) if (directory / name).is_file())


def _normalize_key(file_name: str, modality: str) -> str:
    stem, suffix = os.path.splitext(file_name)
    normalized = stem.replace(modality, "", 1)
    return f"{normalized}{suffix}"


def collect_samples(data_root: str | Path, modalities: list[str]):
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    label_dirs: list[tuple[str, Path]] = []
    for child in root.iterdir():
        if child.is_dir():
            _numeric_angle(child.name)
            label_dirs.append((child.name, child))
    label_dirs.sort(key=lambda item: _numeric_angle(item[0]))

    if not label_dirs:
        raise RuntimeError(f"No numeric angle folders found in {root}")

    label_map: dict[int, str] = {}
    records: list[SampleRecord] = []
    for label, (angle_name, angle_dir) in enumerate(label_dirs):
        label_map[label] = angle_name
        modality_maps: dict[str, dict[str, Path]] = {}
        key_sets = []
        for modality in modalities:
            modality_dir = angle_dir / modality
            if not modality_dir.is_dir():
                raise FileNotFoundError(f"Missing modality directory: {modality_dir}")
            files = _list_files(modality_dir)
            if not files:
                raise RuntimeError(f"No files in modality directory: {modality_dir}")
            file_map = {
                _normalize_key(file_name, modality): modality_dir / file_name
                for file_name in files
            }
            modality_maps[modality] = file_map
            key_sets.append(set(file_map.keys()))

        common_keys = set.intersection(*key_sets) if key_sets else set()
        if not common_keys:
            raise RuntimeError(f"No paired samples for angle {angle_name} and modalities {modalities}")

        for sample_key in sorted(common_keys):
            records.append(
                SampleRecord(
                    label=label,
                    angle=angle_name,
                    key=f"{angle_name}/{sample_key}",
                    modalities={m: modality_maps[m][sample_key] for m in modalities},
                )
            )

    return records, label_map


class RotationAugmentor:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def rotations(self, training: bool) -> list[int]:
        if self.enabled and training:
            return [0, 1, 2, 3]
        return [0]

    def apply(self, tensor: torch.Tensor, rotation: int) -> torch.Tensor:
        if not self.enabled or rotation == 0:
            return tensor
        return torch.rot90(tensor, k=rotation, dims=[1, 2])


class TimepixDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        label_map: Mapping[int, str],
        modalities: list[str],
        training: bool,
        crop_size: int = 0,
        rotation_enabled: bool = False,
        normalizer: Normalizer | None = None,
        feature_extractor: HandcraftedFeatureExtractor | None = None,
        feature_scaler: HandcraftedFeatureScaler | None = None,
        task: str = "classification",
        max_angle: float = 90.0,
        data_dtype: str = "float32",
        toa_transform: str | None = None,
        add_hit_mask: bool = False,
    ) -> None:
        self.records = records
        self.label_map = dict(label_map)
        self.modalities = list(modalities)
        self.training = training
        self.crop_size = int(crop_size or 0)
        self.augmentor = RotationAugmentor(rotation_enabled)
        self.normalizer = normalizer
        self.feature_extractor = feature_extractor
        self.feature_scaler = feature_scaler
        self.task = task
        self.max_angle = max_angle
        self.data_dtype = data_dtype
        self.toa_transform = normalize_toa_transform(toa_transform)
        self.add_hit_mask = bool(add_hit_mask)
        self._expanded: list[tuple[SampleRecord, int]] = []
        for record in self.records:
            for rotation in self.augmentor.rotations(training):
                self._expanded.append((record, rotation))

    def __len__(self) -> int:
        return len(self._expanded)

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    def __getitem__(self, index: int):
        record, rotation = self._expanded[index]
        arrays = {
            modality: load_matrix(record.modalities[modality], self.data_dtype)
            for modality in self.modalities
        }
        cropped_arrays = {
            modality: center_crop_array(arrays[modality], self.crop_size)
            for modality in self.modalities
        }

        channels = []
        for modality in self.modalities:
            array = apply_modality_transform(modality, cropped_arrays[modality], self.toa_transform)
            tensor = torch.as_tensor(array, dtype=torch.float32).unsqueeze(0)
            tensor = self.augmentor.apply(tensor, rotation)
            if self.normalizer is not None:
                tensor = self.normalizer.apply(tensor, modality)
            channels.append(tensor)
        if self.add_hit_mask:
            hit_mask = make_hit_mask(cropped_arrays)
            tensor = torch.as_tensor(hit_mask, dtype=torch.float32).unsqueeze(0)
            tensor = self.augmentor.apply(tensor, rotation)
            channels.append(tensor)
        image = torch.cat(channels, dim=0)

        handcrafted = None
        if self.feature_extractor is not None and self.feature_extractor.dim > 0:
            handcrafted = self.feature_extractor.extract(arrays)
            if handcrafted is not None and self.feature_scaler is not None:
                handcrafted = self.feature_scaler.apply(handcrafted)

        if self.task == "regression":
            angle = float(self.label_map[record.label])
            label = torch.tensor(angle / self.max_angle, dtype=torch.float32)
        else:
            label = torch.tensor(record.label, dtype=torch.long)

        if handcrafted is not None:
            return image, label, handcrafted
        return image, label

    def _center_crop(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.crop_size <= 0:
            return tensor
        _, height, width = tensor.shape
        if self.crop_size > min(height, width):
            raise ValueError(f"crop_size={self.crop_size} exceeds tensor shape {(height, width)}")
        top = (height - self.crop_size) // 2
        left = (width - self.crop_size) // 2
        return tensor[:, top : top + self.crop_size, left : left + self.crop_size]
