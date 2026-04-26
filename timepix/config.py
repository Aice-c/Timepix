"""Configuration loading helpers for Timepix experiments."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def _expand_env(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default} placeholders."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        return match.group(0)

    return _ENV_PATTERN.sub(replace, value)


def expand_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env(value)
    if isinstance(value, list):
        return [expand_placeholders(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_placeholders(v) for k, v in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return expand_placeholders(data)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_project_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(_expand_env(str(path)))
    if path.is_absolute():
        return path
    base = Path(base_dir) if base_dir is not None else PROJECT_ROOT
    return (base / path).resolve()


def get_by_dotted_key(data: Mapping[str, Any], key: str) -> Any:
    cur: Any = data
    for part in key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(key)
        cur = cur[part]
    return cur


def set_by_dotted_key(data: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = data
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment YAML, resolving optional base and dataset config files."""
    path = resolve_project_path(path)
    raw = load_yaml(path)

    if "base" in raw:
        base_path = resolve_project_path(raw["base"], path.parent)
        base_cfg = load_experiment_config(base_path)
        raw = {k: v for k, v in raw.items() if k != "base"}
        cfg = deep_merge(base_cfg, raw)
    else:
        cfg = raw

    dataset = cfg.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("'dataset' must be a mapping")

    dataset_config = dataset.get("config")
    if dataset_config:
        ds_path = resolve_project_path(dataset_config, path.parent)
        ds_cfg = load_yaml(ds_path)
        dataset_override = {k: v for k, v in dataset.items() if k != "config"}
        merged_dataset = deep_merge(ds_cfg, dataset_override)
        merged_dataset["config_path"] = str(ds_path)
        cfg["dataset"] = merged_dataset

    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    return expand_placeholders(cfg)


def parse_override(value: str) -> Any:
    """Parse a command-line override value with YAML semantics."""
    return yaml.safe_load(value)

