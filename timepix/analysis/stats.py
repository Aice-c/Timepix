"""Statistical distance and effect-size helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def numeric_values(frame: pd.DataFrame, feature: str) -> np.ndarray:
    values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Return Cliff's delta using sorted ranks instead of an O(n*m) loop."""

    a = np.asarray(a, dtype=float)
    b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        return 0.0
    greater = np.searchsorted(b, a, side="left").sum()
    less_or_equal = np.searchsorted(b, a, side="right")
    less = (b.size - less_or_equal).sum()
    return float((greater - less) / (a.size * b.size))


def iqr_overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a25, a75 = np.percentile(a, [25, 75])
    b25, b75 = np.percentile(b, [25, 75])
    intersection = max(0.0, min(a75, b75) - max(a25, b25))
    union = max(a75, b75) - min(a25, b25)
    return float(intersection / max(union, 1e-12))


def ks_interpretation(value: float) -> str:
    if value < 0.05:
        return "very_small_distribution_difference"
    if value < 0.10:
        return "small_distribution_difference"
    if value < 0.20:
        return "moderate_distribution_difference"
    return "large_distribution_difference"


def cliffs_interpretation(value: float) -> str:
    abs_value = abs(value)
    if abs_value < 0.147:
        return "negligible"
    if abs_value < 0.33:
        return "small"
    if abs_value < 0.474:
        return "medium"
    return "large"


def _feature_pairs(angles: list[float], *, adjacent_only: bool) -> list[tuple[float, float]]:
    angles = [float(angle) for angle in angles]
    if adjacent_only:
        return list(zip(angles[:-1], angles[1:]))
    return [(left, right) for idx, left in enumerate(angles) for right in angles[idx + 1 :]]


def feature_pair_effects(
    features: pd.DataFrame,
    angles: list[float],
    feature_names: list[str],
    *,
    adjacent_only: bool = True,
) -> pd.DataFrame:
    try:
        from scipy.stats import ks_2samp, wasserstein_distance
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise ImportError("scipy is required for statistical distance analysis") from exc

    rows = []
    for left, right in _feature_pairs(angles, adjacent_only=adjacent_only):
        left_group = features[features["angle_value"] == float(left)]
        right_group = features[features["angle_value"] == float(right)]
        for feature in feature_names:
            a = numeric_values(left_group, feature)
            b = numeric_values(right_group, feature)
            if a.size == 0 or b.size == 0:
                continue
            ks = ks_2samp(a, b)
            a25, a75 = np.percentile(a, [25, 75])
            b25, b75 = np.percentile(b, [25, 75])
            scale = max(float(np.percentile(np.concatenate([a, b]), 75) - np.percentile(np.concatenate([a, b]), 25)), 1e-12)
            cliffs = cliffs_delta(a, b)
            rows.append(
                {
                    "angle_a": left,
                    "angle_b": right,
                    "angle_pair": f"{left:g}-{right:g}",
                    "feature": feature,
                    "n_a": int(a.size),
                    "n_b": int(b.size),
                    "ks_statistic": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                    "wasserstein_distance": float(wasserstein_distance(a, b)),
                    "wasserstein_distance_normalized": float(wasserstein_distance(a, b) / scale),
                    "cliffs_delta": cliffs,
                    "median_a": float(np.median(a)),
                    "median_b": float(np.median(b)),
                    "median_difference": float(np.median(b) - np.median(a)),
                    "iqr_a_low": float(a25),
                    "iqr_a_high": float(a75),
                    "iqr_b_low": float(b25),
                    "iqr_b_high": float(b75),
                    "iqr_overlap_ratio": iqr_overlap_ratio(a, b),
                    "effect_size_interpretation": f"{ks_interpretation(float(ks.statistic))}; cliffs_delta_{cliffs_interpretation(cliffs)}",
                }
            )
    return pd.DataFrame(rows)


def feature_distance_summary(effect_df: pd.DataFrame) -> pd.DataFrame:
    if effect_df.empty:
        return pd.DataFrame()
    grouped = effect_df.groupby("feature", as_index=False).agg(
        max_ks=("ks_statistic", "max"),
        mean_ks=("ks_statistic", "mean"),
        max_wasserstein=("wasserstein_distance", "max"),
        mean_wasserstein=("wasserstein_distance", "mean"),
        max_abs_cliffs_delta=("cliffs_delta", lambda x: float(np.max(np.abs(x)))),
        mean_iqr_overlap=("iqr_overlap_ratio", "mean"),
    )
    return grouped.sort_values(["max_ks", "max_wasserstein"], ascending=False)


def pivot_metric(effect_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if effect_df.empty:
        return pd.DataFrame()
    return effect_df.pivot_table(index="feature", columns="angle_pair", values=metric, aggfunc="max").reset_index()
