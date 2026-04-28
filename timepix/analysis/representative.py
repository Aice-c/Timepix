"""Automatic representative-sample selection."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_REPRESENTATIVE_FEATURES = [
    "active_count",
    "active_sum",
    "bbox_width",
    "bbox_height",
    "bbox_aspect_ratio",
    "energy_entropy",
    "rms_radius",
]


def deterministic_sample(frame: pd.DataFrame, cap: int, seed: int, stratify: str | None = "angle") -> pd.DataFrame:
    if cap <= 0 or len(frame) <= cap:
        return frame.copy()
    rng = np.random.default_rng(seed)
    if stratify is None or stratify not in frame.columns:
        idx = rng.choice(frame.index.to_numpy(), size=cap, replace=False)
        return frame.loc[idx].copy()
    pieces = []
    groups = list(frame.groupby(stratify))
    base = max(1, cap // max(len(groups), 1))
    for _, group in groups:
        n = min(len(group), base)
        idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
        pieces.append(frame.loc[idx])
    sampled = pd.concat(pieces, ignore_index=False) if pieces else frame.head(0)
    if len(sampled) < cap:
        remaining = frame.drop(index=sampled.index, errors="ignore")
        if not remaining.empty:
            n = min(cap - len(sampled), len(remaining))
            idx = rng.choice(remaining.index.to_numpy(), size=n, replace=False)
            sampled = pd.concat([sampled, remaining.loc[idx]], ignore_index=False)
    return sampled.copy()


def select_representatives(
    features: pd.DataFrame,
    *,
    per_group: int = 1,
    group_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Select samples nearest to each group's robust median feature vector."""

    if features.empty:
        return features.copy()
    group_cols = group_cols or ["dataset", "modality", "angle"]
    feature_cols = feature_cols or [col for col in DEFAULT_REPRESENTATIVE_FEATURES if col in features.columns]
    rows = []
    for _, group in features.groupby(group_cols):
        matrix = group[feature_cols].apply(pd.to_numeric, errors="coerce")
        valid = matrix.replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            rows.append(group.head(per_group))
            continue
        median = valid.median(axis=0)
        scale = valid.quantile(0.75, axis=0) - valid.quantile(0.25, axis=0)
        scale = scale.replace(0.0, 1.0).fillna(1.0)
        dist = (((valid - median) / scale) ** 2).sum(axis=1)
        chosen_idx = dist.sort_values().head(per_group).index
        rows.append(group.loc[chosen_idx].assign(representative_distance=dist.loc[chosen_idx].to_numpy()))
    return pd.concat(rows, ignore_index=True) if rows else features.head(0).copy()

