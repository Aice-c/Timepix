"""Matrix loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


SUPPORTED_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}


def resolve_dtype(dtype: str | None):
    name = dtype or "float32"
    if name not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported data dtype: {name}. Supported: {sorted(SUPPORTED_DTYPES)}")
    return SUPPORTED_DTYPES[name]


def load_matrix(path: str | Path, dtype: str | None = "float32") -> np.ndarray:
    """Load a text matrix directly into the requested floating dtype."""
    return np.loadtxt(path, dtype=resolve_dtype(dtype))

