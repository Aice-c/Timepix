"""Lightweight validation for experiment dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TOP_LEVEL_KEYS = {
    "experiment_name",
    "experiment_group",
    "dataset",
    "task",
    "model",
    "loss",
    "training",
    "split",
    "data",
    "normalization",
    "augmentation",
    "handcrafted_features",
    "output",
    "grid",
    "_config_path",
    "_config_dir",
}

SECTION_KEYS = {
    "dataset": {
        "name",
        "particle",
        "root",
        "available_modalities",
        "default_modalities",
        "label_type",
        "sample_shape",
        "modalities",
        "config_path",
    },
    "task": {"type", "primary_metric", "max_angle"},
    "model": {
        "name",
        "pretrained",
        "kernel_size",
        "stride",
        "padding",
        "conv1_kernel_size",
        "conv1_stride",
        "conv1_padding",
        "feature_dim",
        "hidden_dim",
        "dropout",
        "fusion_mode",
    },
    "loss": {"name", "label_encoding", "emd_p", "gaussian_sigma"},
    "training": {
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "eta_min",
        "early_stopping_patience",
        "seed",
        "num_workers",
        "pin_memory",
        "dropout",
        "progress_bar",
        "save_last_checkpoint",
        "resume_from",
    },
    "split": {"train", "val", "test", "reuse_split", "path"},
    "data": {"crop_size", "dtype"},
    "augmentation": {"rotation_90"},
    "handcrafted_features": {"enabled", "standardize", "features"},
    "output": {"root"},
}

SUPPORTED_MODELS = {
    "resnet18",
    "resnet18_no_maxpool",
    "resnet18_maxpool",
    "resnet18_with_maxpool",
    "resnet18_original",
    "shallow_resnet",
    "shallow_cnn",
}
SUPPORTED_TASKS = {"classification", "regression"}
SUPPORTED_CLASSIFICATION_LOSSES = {"cross_entropy", "emd"}
SUPPORTED_REGRESSION_LOSSES = {"mse", "smooth_l1"}
SUPPORTED_LABEL_ENCODINGS = {"onehot", "gaussian"}
SUPPORTED_SCHEDULERS = {"none", "cosine"}
SUPPORTED_FUSION_MODES = {"none", "concat", "gated"}
SUPPORTED_DATA_DTYPES = {"float16", "float32", "float64"}
NORMALIZATION_KEYS = {"enabled", "log1p", "ignore_zero"}


def _require_mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    errors.append(f"{path} must be a mapping")
    return None


def _check_unknown_keys(mapping: Mapping[str, Any], allowed: set[str], path: str, errors: list[str]) -> None:
    for key in mapping:
        if key not in allowed:
            errors.append(f"Unknown config key: {path}.{key}")


def _check_positive_int(value: Any, path: str, errors: list[str], *, allow_zero: bool = False) -> None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        errors.append(f"{path} must be an integer")
        return
    if allow_zero:
        if number < 0:
            errors.append(f"{path} must be >= 0")
    elif number <= 0:
        errors.append(f"{path} must be > 0")


def _check_float(value: Any, path: str, errors: list[str], *, min_value: float | None = None) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{path} must be numeric")
        return
    if min_value is not None and number < min_value:
        errors.append(f"{path} must be >= {min_value}")


def validate_experiment_config(cfg: Mapping[str, Any]) -> None:
    """Raise ValueError if a resolved experiment config is inconsistent."""
    errors: list[str] = []
    _check_unknown_keys(cfg, TOP_LEVEL_KEYS, "config", errors)

    for section, allowed in SECTION_KEYS.items():
        if section in cfg:
            section_cfg = _require_mapping(cfg[section], section, errors)
            if section_cfg is not None:
                _check_unknown_keys(section_cfg, allowed, section, errors)

    dataset = _require_mapping(cfg.get("dataset", {}), "dataset", errors) or {}
    modalities = list(dataset.get("modalities") or dataset.get("default_modalities") or [])
    available = list(dataset.get("available_modalities") or [])
    if not modalities:
        errors.append("dataset.modalities or dataset.default_modalities is required")
    missing = [modality for modality in modalities if modality not in available]
    if missing:
        errors.append(f"dataset.modalities contains unsupported modalities: {missing}; available={available}")

    task_cfg = _require_mapping(cfg.get("task", {}), "task", errors) or {}
    task = task_cfg.get("type", "classification")
    if task not in SUPPORTED_TASKS:
        errors.append(f"task.type must be one of {sorted(SUPPORTED_TASKS)}, got {task!r}")

    model_cfg = _require_mapping(cfg.get("model", {}), "model", errors) or {}
    model_name = model_cfg.get("name", "resnet18")
    if model_name not in SUPPORTED_MODELS:
        errors.append(f"model.name must be one of {sorted(SUPPORTED_MODELS)}, got {model_name!r}")
    if "fusion_mode" in model_cfg and model_cfg["fusion_mode"] not in SUPPORTED_FUSION_MODES:
        errors.append(f"model.fusion_mode must be one of {sorted(SUPPORTED_FUSION_MODES)}")
    for key in ("conv1_kernel_size", "kernel_size", "conv1_stride", "stride", "feature_dim", "hidden_dim"):
        if key in model_cfg:
            _check_positive_int(model_cfg[key], f"model.{key}", errors)
    for key in ("conv1_padding", "padding"):
        if key in model_cfg:
            _check_positive_int(model_cfg[key], f"model.{key}", errors, allow_zero=True)
    if "dropout" in model_cfg:
        _check_float(model_cfg["dropout"], "model.dropout", errors, min_value=0.0)

    loss_cfg = _require_mapping(cfg.get("loss", {}), "loss", errors) or {}
    loss_name = loss_cfg.get("name", "cross_entropy")
    if task == "classification" and loss_name not in SUPPORTED_CLASSIFICATION_LOSSES:
        errors.append(f"loss.name for classification must be one of {sorted(SUPPORTED_CLASSIFICATION_LOSSES)}")
    if task == "regression" and loss_name not in SUPPORTED_REGRESSION_LOSSES:
        errors.append(f"loss.name for regression must be one of {sorted(SUPPORTED_REGRESSION_LOSSES)}")
    if loss_cfg.get("label_encoding", "onehot") not in SUPPORTED_LABEL_ENCODINGS:
        errors.append(f"loss.label_encoding must be one of {sorted(SUPPORTED_LABEL_ENCODINGS)}")

    training_cfg = _require_mapping(cfg.get("training", {}), "training", errors) or {}
    for key in ("epochs", "batch_size"):
        if key in training_cfg:
            _check_positive_int(training_cfg[key], f"training.{key}", errors)
    for key in ("num_workers", "early_stopping_patience"):
        if key in training_cfg:
            _check_positive_int(training_cfg[key], f"training.{key}", errors, allow_zero=True)
    for key in ("learning_rate", "weight_decay", "eta_min", "dropout"):
        if key in training_cfg:
            _check_float(training_cfg[key], f"training.{key}", errors, min_value=0.0)
    if training_cfg.get("scheduler", "none") not in SUPPORTED_SCHEDULERS:
        errors.append(f"training.scheduler must be one of {sorted(SUPPORTED_SCHEDULERS)}")

    split_cfg = _require_mapping(cfg.get("split", {}), "split", errors) or {}
    if {"train", "val", "test"} <= set(split_cfg):
        try:
            ratios = [float(split_cfg[name]) for name in ("train", "val", "test")]
        except (TypeError, ValueError):
            errors.append("split.train/val/test must be numeric")
        else:
            if abs(sum(ratios) - 1.0) > 1e-6:
                errors.append(f"split.train/val/test must sum to 1.0, got {sum(ratios)}")

    data_cfg = _require_mapping(cfg.get("data", {}), "data", errors) or {}
    if data_cfg.get("dtype", "float32") not in SUPPORTED_DATA_DTYPES:
        errors.append(f"data.dtype must be one of {sorted(SUPPORTED_DATA_DTYPES)}")

    normalization_cfg = cfg.get("normalization", {})
    if normalization_cfg:
        norm_map = _require_mapping(normalization_cfg, "normalization", errors) or {}
        for modality, modality_cfg in norm_map.items():
            section_cfg = _require_mapping(modality_cfg, f"normalization.{modality}", errors)
            if section_cfg is not None:
                _check_unknown_keys(section_cfg, NORMALIZATION_KEYS, f"normalization.{modality}", errors)

    if errors:
        formatted = "\n  - ".join(errors)
        raise ValueError(f"Invalid experiment config:\n  - {formatted}")


def validate_grid_mapping(grid: Any) -> Mapping[str, Any]:
    if not isinstance(grid, Mapping):
        raise ValueError("grid must be a mapping")
    if not grid:
        raise ValueError("grid must contain at least one key")
    for key, values in grid.items():
        if not isinstance(key, str) or not key:
            raise ValueError("grid keys must be non-empty strings")
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid.{key} must be a non-empty list")
    return grid
