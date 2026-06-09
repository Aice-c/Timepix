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
        "class_names",
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
        "image_size",
        "patch_size",
        "dropout",
        "fusion_mode",
        "aux_loss",
        "gate",
        "film",
        "expert_gate",
    },
    "loss": {
        "name",
        "label_encoding",
        "class_weight",
        "class_weights",
        "emd_p",
        "emd_weight",
        "emd_angle_weighted",
        "expected_mae_weight",
        "gaussian_sigma",
        "normalize_by_angle_range",
        "pair_aux",
        "pair_weight",
        "pair_classes",
    },
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
        "mixed_precision",
        "mixed_precision_dtype",
    },
    "split": {"train", "val", "test", "reuse_split", "path", "seed"},
    "data": {"crop_size", "dtype", "toa_transform", "add_hit_mask"},
    "augmentation": {"rotation_90"},
    "handcrafted_features": {"enabled", "standardize", "features", "source_modalities"},
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
    "densenet121",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_tiny",
    "handcrafted_mlp",
    "dual_stream_concat_aux",
    "dual_stream_gmu_aux",
    "toa_conditioned_film",
    "warm_started_expert_gate",
}
SUPPORTED_TASKS = {"classification", "regression"}
SUPPORTED_CLASSIFICATION_LOSSES = {"cross_entropy", "emd", "ce_expected_mae", "ce_emd", "ce_pair_aux"}
SUPPORTED_REGRESSION_LOSSES = {"mse", "smooth_l1"}
SUPPORTED_LABEL_ENCODINGS = {"onehot", "gaussian"}
SUPPORTED_SCHEDULERS = {"none", "cosine"}
SUPPORTED_FUSION_MODES = {"none", "concat", "gated"}
SUPPORTED_DATA_DTYPES = {"float16", "float32", "float64"}
SUPPORTED_TOA_TRANSFORMS = {"none", "raw_log1p", "relative_minmax", "relative_centered", "relative_rank"}
SUPPORTED_MIXED_PRECISION_DTYPES = {"float16", "fp16", "bfloat16", "bf16"}
SUPPORTED_LABEL_TYPES = {"angle_folder", "categorical_folder"}
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


def _check_bool(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, bool):
        errors.append(f"{path} must be true or false")


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
    label_type = str(dataset.get("label_type", "angle_folder")).strip().lower()
    if label_type not in SUPPORTED_LABEL_TYPES:
        errors.append(f"dataset.label_type must be one of {sorted(SUPPORTED_LABEL_TYPES)}, got {label_type!r}")
    class_names = dataset.get("class_names")
    if class_names is not None and (
        not isinstance(class_names, list) or not all(isinstance(item, str) and item for item in class_names)
    ):
        errors.append("dataset.class_names must be a list of non-empty strings when provided")

    modalities = list(dataset.get("modalities") or dataset.get("default_modalities") or [])
    available = list(dataset.get("available_modalities") or [])
    if not modalities:
        errors.append("dataset.modalities or dataset.default_modalities is required")
    missing = [modality for modality in modalities if modality not in available]
    if missing:
        errors.append(f"dataset.modalities contains unsupported modalities: {missing}; available={available}")

    handcrafted_cfg = _require_mapping(cfg.get("handcrafted_features", {}), "handcrafted_features", errors) or {}
    feature_source_modalities = list(handcrafted_cfg.get("source_modalities") or [])
    missing_feature_sources = [modality for modality in feature_source_modalities if modality not in available]
    if missing_feature_sources:
        errors.append(
            "handcrafted_features.source_modalities contains unsupported modalities: "
            f"{missing_feature_sources}; available={available}"
        )

    task_cfg = _require_mapping(cfg.get("task", {}), "task", errors) or {}
    task = task_cfg.get("type", "classification")
    if task not in SUPPORTED_TASKS:
        errors.append(f"task.type must be one of {sorted(SUPPORTED_TASKS)}, got {task!r}")
    if task == "regression" and label_type != "angle_folder":
        errors.append("task.type='regression' requires dataset.label_type='angle_folder'")

    model_cfg = _require_mapping(cfg.get("model", {}), "model", errors) or {}
    model_name = model_cfg.get("name", "resnet18")
    if model_name not in SUPPORTED_MODELS:
        errors.append(f"model.name must be one of {sorted(SUPPORTED_MODELS)}, got {model_name!r}")
    if "pretrained" in model_cfg:
        _check_bool(model_cfg["pretrained"], "model.pretrained", errors)
    if "fusion_mode" in model_cfg and model_cfg["fusion_mode"] not in SUPPORTED_FUSION_MODES:
        errors.append(f"model.fusion_mode must be one of {sorted(SUPPORTED_FUSION_MODES)}")
    for key in (
        "conv1_kernel_size",
        "kernel_size",
        "conv1_stride",
        "stride",
        "feature_dim",
        "hidden_dim",
        "image_size",
        "patch_size",
    ):
        if key in model_cfg:
            _check_positive_int(model_cfg[key], f"model.{key}", errors)
    for key in ("conv1_padding", "padding"):
        if key in model_cfg:
            _check_positive_int(model_cfg[key], f"model.{key}", errors, allow_zero=True)
    if "dropout" in model_cfg:
        _check_float(model_cfg["dropout"], "model.dropout", errors, min_value=0.0)
    if model_cfg.get("name") == "vit_tiny" and "image_size" in model_cfg and "patch_size" in model_cfg:
        try:
            if int(model_cfg["image_size"]) % int(model_cfg["patch_size"]) != 0:
                errors.append("model.image_size must be divisible by model.patch_size for vit_tiny")
        except (TypeError, ValueError):
            pass
    if model_cfg.get("name") == "vit_tiny" and model_cfg.get("pretrained", False):
        errors.append("model.pretrained must be false for vit_tiny")

    loss_cfg = _require_mapping(cfg.get("loss", {}), "loss", errors) or {}
    loss_name = loss_cfg.get("name", "cross_entropy")
    if task == "classification" and loss_name not in SUPPORTED_CLASSIFICATION_LOSSES:
        errors.append(f"loss.name for classification must be one of {sorted(SUPPORTED_CLASSIFICATION_LOSSES)}")
    if task == "regression" and loss_name not in SUPPORTED_REGRESSION_LOSSES:
        errors.append(f"loss.name for regression must be one of {sorted(SUPPORTED_REGRESSION_LOSSES)}")
    if loss_cfg.get("label_encoding", "onehot") not in SUPPORTED_LABEL_ENCODINGS:
        errors.append(f"loss.label_encoding must be one of {sorted(SUPPORTED_LABEL_ENCODINGS)}")
    if label_type == "categorical_folder":
        label_encoding = loss_cfg.get("label_encoding", "onehot")
        if loss_name in {"emd", "ce_expected_mae", "ce_emd"} or label_encoding == "gaussian":
            errors.append(
                "categorical_folder only supports unordered classification losses such as "
                "cross_entropy with onehot labels"
            )
    for key in ("gaussian_sigma", "emd_weight", "expected_mae_weight"):
        if key in loss_cfg:
            _check_float(loss_cfg[key], f"loss.{key}", errors, min_value=0.0)
    if "emd_p" in loss_cfg:
        _check_positive_int(loss_cfg["emd_p"], "loss.emd_p", errors)
    for key in ("emd_angle_weighted", "normalize_by_angle_range"):
        if key in loss_cfg:
            _check_bool(loss_cfg[key], f"loss.{key}", errors)
    for key in ("class_weight", "class_weights"):
        if key in loss_cfg:
            value = loss_cfg[key]
            if isinstance(value, str):
                if value not in {"balanced", "none"}:
                    errors.append(f"loss.{key} must be 'balanced', 'none', or a list of non-negative numbers")
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    _check_float(item, f"loss.{key}[{idx}]", errors, min_value=0.0)
            elif value is not None and value is not False:
                errors.append(f"loss.{key} must be 'balanced', 'none', or a list of non-negative numbers")
    if loss_name == "ce_pair_aux":
        pair_cfg = _require_mapping(loss_cfg.get("pair_aux", {}), "loss.pair_aux", errors) or {}
        pair_classes = pair_cfg.get("classes", loss_cfg.get("pair_classes"))
        if not isinstance(pair_classes, list) or len(pair_classes) != 2 or not all(
            isinstance(item, str) and item for item in pair_classes
        ):
            errors.append("loss.pair_aux.classes must be a list of exactly two non-empty class names")
        pair_weight = pair_cfg.get("weight", loss_cfg.get("pair_weight", 0.1))
        _check_float(pair_weight, "loss.pair_aux.weight", errors, min_value=0.0)
        pair_class_weight = pair_cfg.get("class_weight", pair_cfg.get("class_weights"))
        if isinstance(pair_class_weight, str):
            if pair_class_weight not in {"balanced", "none"}:
                errors.append(
                    "loss.pair_aux.class_weight must be 'balanced', 'none', or a list of two non-negative numbers"
                )
        elif isinstance(pair_class_weight, list):
            if len(pair_class_weight) != 2:
                errors.append("loss.pair_aux.class_weight list must contain two weights")
            for idx, item in enumerate(pair_class_weight):
                _check_float(item, f"loss.pair_aux.class_weight[{idx}]", errors, min_value=0.0)
        elif pair_class_weight is not None and pair_class_weight is not False:
            errors.append("loss.pair_aux.class_weight must be 'balanced', 'none', or a list of two non-negative numbers")

    training_cfg = _require_mapping(cfg.get("training", {}), "training", errors) or {}
    for key in ("epochs", "batch_size"):
        if key in training_cfg:
            _check_positive_int(training_cfg[key], f"training.{key}", errors)
    for key in ("num_workers", "early_stopping_patience", "seed"):
        if key in training_cfg:
            _check_positive_int(training_cfg[key], f"training.{key}", errors, allow_zero=True)
    for key in ("learning_rate", "weight_decay", "eta_min", "dropout"):
        if key in training_cfg:
            _check_float(training_cfg[key], f"training.{key}", errors, min_value=0.0)
    for key in ("pin_memory", "progress_bar", "save_last_checkpoint", "mixed_precision"):
        if key in training_cfg:
            _check_bool(training_cfg[key], f"training.{key}", errors)
    if training_cfg.get("scheduler", "none") not in SUPPORTED_SCHEDULERS:
        errors.append(f"training.scheduler must be one of {sorted(SUPPORTED_SCHEDULERS)}")
    if str(training_cfg.get("mixed_precision_dtype", "float16")).lower() not in SUPPORTED_MIXED_PRECISION_DTYPES:
        errors.append(f"training.mixed_precision_dtype must be one of {sorted(SUPPORTED_MIXED_PRECISION_DTYPES)}")

    split_cfg = _require_mapping(cfg.get("split", {}), "split", errors) or {}
    if "seed" in split_cfg:
        _check_positive_int(split_cfg["seed"], "split.seed", errors, allow_zero=True)
    if "reuse_split" in split_cfg:
        _check_bool(split_cfg["reuse_split"], "split.reuse_split", errors)
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
    if str(data_cfg.get("toa_transform", "none")).lower() not in SUPPORTED_TOA_TRANSFORMS:
        errors.append(f"data.toa_transform must be one of {sorted(SUPPORTED_TOA_TRANSFORMS)}")
    if "add_hit_mask" in data_cfg:
        _check_bool(data_cfg["add_hit_mask"], "data.add_hit_mask", errors)

    normalization_cfg = cfg.get("normalization", {})
    if normalization_cfg:
        norm_map = _require_mapping(normalization_cfg, "normalization", errors) or {}
        for modality, modality_cfg in norm_map.items():
            section_cfg = _require_mapping(modality_cfg, f"normalization.{modality}", errors)
            if section_cfg is not None:
                _check_unknown_keys(section_cfg, NORMALIZATION_KEYS, f"normalization.{modality}", errors)
        toa_norm_cfg = norm_map.get("ToA", {})
        if (
            isinstance(toa_norm_cfg, Mapping)
            and str(data_cfg.get("toa_transform", "none")).lower() != "none"
            and bool(toa_norm_cfg.get("log1p", False))
        ):
            errors.append("normalization.ToA.log1p must be false when data.toa_transform is not 'none'")

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
