"""Path helpers for experiment outputs."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "experiment"


def make_experiment_dir(output_root: str | Path, experiment_name: str) -> Path:
    root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{timestamp}_{slugify(experiment_name)}"
    path = root / name
    path.mkdir(parents=True, exist_ok=False)
    return path

