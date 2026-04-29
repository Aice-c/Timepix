"""Build datasets and dataloaders from experiment config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from timepix.config import PROJECT_ROOT, resolve_project_path

from .dataset import TimepixDataset, collect_samples
from .features import HandcraftedFeatureExtractor, compute_feature_scaler, parse_feature_config
from .normalization import compute_normalizer
from .splits import load_split_manifest, save_split_manifest, stratified_split
from .transforms import normalize_toa_transform


def _validate_modalities(dataset_cfg: dict[str, Any], modalities: list[str]) -> None:
    available = dataset_cfg.get("available_modalities", [])
    missing = [m for m in modalities if m not in available]
    if missing:
        name = dataset_cfg.get("name", "dataset")
        raise ValueError(f"Dataset {name} supports modalities {available}, got {modalities}")


def _unique_modalities(*groups: list[str]) -> list[str]:
    seen: list[str] = []
    for group in groups:
        for modality in group:
            if modality not in seen:
                seen.append(modality)
    return seen


def _default_split_path(cfg: dict[str, Any], modalities: list[str]) -> Path:
    dataset = cfg["dataset"]
    split = cfg.get("split", {})
    training_seed = cfg.get("training", {}).get("seed", 42)
    split_seed = split.get("seed", training_seed)
    ratios = f"{split.get('train', 0.8)}_{split.get('val', 0.1)}_{split.get('test', 0.1)}"
    mod = "-".join(modalities)
    name = f"{dataset.get('name', 'dataset')}_{mod}_seed{split_seed}_{ratios}.json"
    return PROJECT_ROOT / "outputs" / "splits" / name


def _label_counts(records, label_map: dict[int, str]) -> dict[str, int]:
    counts = {str(label_map[label]): 0 for label in sorted(label_map)}
    for record in records:
        counts[str(label_map[record.label])] += 1
    return counts


def build_dataloaders(cfg: dict[str, Any], data_root_override: str | None = None, *, eval_mode: bool = False):
    dataset_cfg = cfg["dataset"]
    modalities = list(dataset_cfg.get("modalities") or dataset_cfg.get("default_modalities") or ["ToT"])
    _validate_modalities(dataset_cfg, modalities)

    handcrafted_cfg = cfg.get("handcrafted_features", {})
    configured_feature_modalities = list(handcrafted_cfg.get("source_modalities") or [])
    _validate_modalities(dataset_cfg, configured_feature_modalities)

    root = data_root_override or dataset_cfg.get("root")
    if root is None:
        raise ValueError("dataset.root is required")
    data_root = resolve_project_path(root)

    feature_names = parse_feature_config(handcrafted_cfg, modalities)
    feature_extractor = HandcraftedFeatureExtractor(feature_names)
    feature_source_modalities = _unique_modalities(configured_feature_modalities, feature_extractor.required_modalities)
    _validate_modalities(dataset_cfg, feature_source_modalities)
    collect_modalities = _unique_modalities(modalities, feature_source_modalities)

    records, label_map = collect_samples(data_root, collect_modalities)

    split_cfg = cfg.get("split", {})
    training_seed = int(cfg.get("training", {}).get("seed", 42))
    split_seed = int(split_cfg.get("seed", training_seed))
    reuse_split = bool(split_cfg.get("reuse_split", True))
    split_path = split_cfg.get("path")
    split_path = resolve_project_path(split_path) if split_path else _default_split_path(cfg, modalities)

    if reuse_split and split_path.exists():
        train_idx, val_idx, test_idx = load_split_manifest(split_path, records)
    else:
        train_idx, val_idx, test_idx = stratified_split(
            records,
            float(split_cfg.get("train", 0.8)),
            float(split_cfg.get("val", 0.1)),
            float(split_cfg.get("test", 0.1)),
            split_seed,
        )
        if reuse_split:
            save_split_manifest(split_path, records, train_idx, val_idx, test_idx)

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    data_cfg = cfg.get("data", {})
    crop_size = int(data_cfg.get("crop_size", 0) or 0)
    data_dtype = data_cfg.get("dtype", "float32")
    toa_transform = normalize_toa_transform(data_cfg.get("toa_transform", "none"))
    add_hit_mask = bool(data_cfg.get("add_hit_mask", False))
    normalizer = compute_normalizer(
        train_records,
        modalities,
        cfg.get("normalization", {}),
        crop_size=crop_size,
        data_dtype=data_dtype,
        toa_transform=toa_transform,
    )

    feature_scaler = None
    if feature_extractor.dim > 0 and handcrafted_cfg.get("standardize", True):
        feature_scaler = compute_feature_scaler(train_records, feature_extractor, data_dtype=data_dtype, crop_size=crop_size)

    task = cfg.get("task", {}).get("type", "classification")
    max_angle = float(cfg.get("task", {}).get("max_angle", 90.0))
    rotation_enabled = bool(cfg.get("augmentation", {}).get("rotation_90", False))

    train_dataset = TimepixDataset(
        train_records,
        label_map,
        modalities,
        training=not eval_mode,
        crop_size=crop_size,
        rotation_enabled=rotation_enabled and not eval_mode,
        normalizer=normalizer,
        feature_extractor=feature_extractor,
        feature_scaler=feature_scaler,
        task=task,
        max_angle=max_angle,
        data_dtype=data_dtype,
        toa_transform=toa_transform,
        add_hit_mask=add_hit_mask,
    )
    val_dataset = TimepixDataset(
        val_records,
        label_map,
        modalities,
        training=False,
        crop_size=crop_size,
        rotation_enabled=False,
        normalizer=normalizer,
        feature_extractor=feature_extractor,
        feature_scaler=feature_scaler,
        task=task,
        max_angle=max_angle,
        data_dtype=data_dtype,
        toa_transform=toa_transform,
        add_hit_mask=add_hit_mask,
    )
    test_dataset = TimepixDataset(
        test_records,
        label_map,
        modalities,
        training=False,
        crop_size=crop_size,
        rotation_enabled=False,
        normalizer=normalizer,
        feature_extractor=feature_extractor,
        feature_scaler=feature_scaler,
        task=task,
        max_angle=max_angle,
        data_dtype=data_dtype,
        toa_transform=toa_transform,
        add_hit_mask=add_hit_mask,
    )

    training_cfg = cfg.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 64))
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", True))

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=not eval_mode,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    }

    info = {
        "data_root": str(data_root),
        "modalities": modalities,
        "collect_modalities": collect_modalities,
        "feature_source_modalities": feature_source_modalities,
        "label_map": label_map,
        "num_classes": len(label_map),
        "training_seed": training_seed,
        "split_seed": split_seed,
        "split_path": str(split_path),
        "split_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "split_label_counts": {
            "train": _label_counts(train_records, label_map),
            "val": _label_counts(val_records, label_map),
            "test": _label_counts(test_records, label_map),
        },
        "handcrafted_dim": feature_extractor.dim,
        "handcrafted_features": feature_extractor.feature_names,
        "data_dtype": data_dtype,
        "toa_transform": toa_transform,
        "add_hit_mask": add_hit_mask,
        "input_channels": len(modalities) + int(add_hit_mask),
        "eval_mode": bool(eval_mode),
    }
    return loaders, info
