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


def feature_pair_effects(features: pd.DataFrame, angles: list[float], feature_names: list[str]) -> pd.DataFrame:
    try:
        from scipy.stats import ks_2samp, wasserstein_distance
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise ImportError("scipy is required for statistical distance analysis") from exc

    rows = []
    for left, right in zip(angles[:-1], angles[1:]):
        left_group = features[features["angle_value"] == float(left)]
        right_group = features[features["angle_value"] == float(right)]
        for feature in feature_names:
            a = numeric_values(left_group, feature)
            b = numeric_values(right_group, feature)
            if a.size == 0 or b.size == 0:
                continue
            ks = ks_2samp(a, b)
            rows.append(
                {
                    "angle_left": left,
                    "angle_right": right,
                    "angle_pair": f"{left:g}-{right:g}",
                    "feature": feature,
                    "n_left": int(a.size),
                    "n_right": int(b.size),
                    "ks_statistic": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                    "wasserstein_distance": float(wasserstein_distance(a, b)),
                    "cliffs_delta": cliffs_delta(a, b),
                    "median_difference": float(np.median(b) - np.median(a)),
                    "iqr_overlap_ratio": iqr_overlap_ratio(a, b),
                    "left_median": float(np.median(a)),
                    "right_median": float(np.median(b)),
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

