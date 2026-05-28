#!/usr/bin/env python3
"""Stage-3a conservative source-label cleaning audit.

This stage does not export the final cleaned training dataset. It proposes a
conservative keep/reject audit for source-label training, where labels remain
the radiation sources (`Am`, `Co60`, `Sr`) rather than automatic beta/gamma
cluster labels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DEFAULT_STAGE1_ROOT = Path(r"E:\TimepixData\particle\stage1_single_particle_candidates_100x100")
DEFAULT_STAGE2A_ROOT = Path(r"E:\TimepixData\particle\stage2_cluster_features_v1")
DEFAULT_OUTPUT_ROOT = Path(r"E:\TimepixData\particle\stage3a_source_cleaning_audit_v1")

PARTICLE_ORDER = ["Am", "Co60", "Sr"]
FEATURE_COLUMNS = ["Npix", "S_total_ToT", "Pmax", "Rg", "E_pca", "Fbox"]
PCA_FEATURE_COLUMNS = ["Npix", "S_total_ToT", "Pmax", "Rg", "E_pca", "Fbox"]


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.dpi": 120,
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")


def angle_key(value: object) -> str:
    if pd.isna(value):
        return "NA"
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:g}"
    except (TypeError, ValueError):
        return str(value)


def otsu_threshold_1d(values: Iterable[float]) -> float:
    """Return a one-dimensional Otsu threshold over observed values."""
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return 0.0
    unique = np.unique(arr)
    if unique.size == 1:
        return float(unique[0])

    best_threshold = float(unique[0])
    best_score = -math.inf
    total_mean = float(arr.mean())
    for threshold in unique[:-1]:
        left = arr[arr <= threshold]
        right = arr[arr > threshold]
        if left.size == 0 or right.size == 0:
            continue
        weight_left = left.size / arr.size
        weight_right = right.size / arr.size
        score = (
            weight_left * (float(left.mean()) - total_mean) ** 2
            + weight_right * (float(right.mean()) - total_mean) ** 2
        )
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _quantile(values: pd.Series, q: float) -> float:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    return float(clean.quantile(q))


def build_group_thresholds(
    features: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build conservative per-source/per-angle threshold table."""
    group_cols = group_cols or ["particle", "angle"]
    work = features.copy()
    if "angle" in group_cols:
        work["angle_key"] = work["angle"].map(angle_key)
        effective_group_cols = ["particle", "angle_key"]
    else:
        effective_group_cols = group_cols

    rows: list[dict[str, object]] = []
    for keys, sub in work.groupby(effective_group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(effective_group_cols, keys))
        row: dict[str, object] = {**key_values, "n": int(len(sub))}
        for feature in FEATURE_COLUMNS:
            row[f"{feature}_median"] = _quantile(sub[feature], 0.50)
            row[f"{feature}_iqr"] = _quantile(sub[feature], 0.75) - _quantile(sub[feature], 0.25)
        row.update(
            {
                "Npix_low_q05": _quantile(sub["Npix"], 0.05),
                "S_total_ToT_low_q05": _quantile(sub["S_total_ToT"], 0.05),
                "Npix_high_q995": _quantile(sub["Npix"], 0.995),
                "S_total_ToT_high_q995": _quantile(sub["S_total_ToT"], 0.995),
                "Rg_high_q995": _quantile(sub["Rg"], 0.995),
                "E_pca_high_q995": _quantile(sub["E_pca"], 0.995),
                "Fbox_low_q005": _quantile(sub["Fbox"], 0.005),
                "Pmax_high_q995": _quantile(sub["Pmax"], 0.995),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def suggest_am_npix_threshold(features: pd.DataFrame) -> float:
    am = features[features["particle"] == "Am"]
    if am.empty:
        return 0.0
    return otsu_threshold_1d(am["Npix"].to_numpy(dtype=float))


def _merge_thresholds(features: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    work = features.copy()
    work["angle_key"] = work["angle"].map(angle_key)
    if "angle_key" not in thresholds.columns:
        thresholds = thresholds.copy()
        thresholds["angle_key"] = thresholds["angle"].map(angle_key)
    return work.merge(thresholds, on=["particle", "angle_key"], how="left", suffixes=("", "_thr"))


def _robust_score(value: float, median: float, iqr: float) -> float:
    scale = iqr if abs(iqr) > 1e-9 else 1.0
    return abs(float(value) - float(median)) / scale


def apply_cleaning_flags(
    features: pd.DataFrame,
    thresholds: pd.DataFrame,
    am_npix_threshold: float,
) -> pd.DataFrame:
    """Apply conservative source-label cleaning flags."""
    merged = _merge_thresholds(features, thresholds)
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        reasons: list[str] = []
        review_flags: list[str] = []
        particle = str(row["particle"])

        if particle == "Am":
            if float(row["Npix"]) < float(am_npix_threshold):
                reasons.append("am_low_npix")
        else:
            low_npix_limit = max(1.0, float(row.get("Npix_low_q05", 1.0)))
            low_tot_limit = float(row.get("S_total_ToT_low_q05", 0.0))
            if float(row["Npix"]) <= low_npix_limit and float(row["S_total_ToT"]) <= low_tot_limit:
                reasons.append("low_signal_noise_like")

            if (
                float(row["Npix"]) >= float(row.get("Npix_high_q995", np.inf))
                or float(row["S_total_ToT"]) >= float(row.get("S_total_ToT_high_q995", np.inf))
                or float(row["Rg"]) >= float(row.get("Rg_high_q995", np.inf))
            ):
                review_flags.append("extreme_large_component")

            if (
                float(row["Fbox"]) <= float(row.get("Fbox_low_q005", -np.inf))
                and float(row["Npix"]) >= float(row.get("Npix_median", 0.0))
            ):
                review_flags.append("extreme_sparse_shape")

            robust_extremes = 0
            for feature in FEATURE_COLUMNS:
                robust_extremes += int(
                    _robust_score(
                        float(row[feature]),
                        float(row.get(f"{feature}_median", 0.0)),
                        float(row.get(f"{feature}_iqr", 1.0)),
                    )
                    >= 8.0
                )
            if robust_extremes >= 3:
                review_flags.append("multi_feature_outlier")

        out = row[features.columns].to_dict()
        out["recommended_keep"] = len(reasons) == 0
        out["reject_reasons"] = ";".join(reasons) if reasons else "keep"
        out["review_flags"] = ";".join(review_flags) if review_flags else "none"
        out["am_npix_threshold"] = float(am_npix_threshold)
        rows.append(out)
    return pd.DataFrame(rows)


def write_rule_summary(thresholds: pd.DataFrame, am_npix_threshold: float, output_path: Path) -> None:
    summary = thresholds.copy()
    summary.insert(0, "am_global_npix_threshold", float(am_npix_threshold))
    summary.to_csv(output_path, index=False)


def write_counts(audit: pd.DataFrame, output_path: Path) -> None:
    rows: list[dict[str, object]] = []
    group_cols = ["particle", "angle"]
    for keys, sub in audit.groupby(group_cols, dropna=False, sort=True):
        particle, angle = keys
        total = len(sub)
        kept = int(sub["recommended_keep"].sum())
        rejected = total - kept
        base = {
            "particle": particle,
            "angle": angle,
            "total": int(total),
            "kept": kept,
            "rejected": rejected,
            "reject_rate": float(rejected / total) if total else 0.0,
        }
        reason_counts: dict[str, int] = {}
        for reason_text in sub.loc[~sub["recommended_keep"], "reject_reasons"]:
            for reason in str(reason_text).split(";"):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reason_counts:
            for reason, count in sorted(reason_counts.items()):
                rows.append({**base, "reason": reason, "reason_count": int(count)})
        else:
            rows.append({**base, "reason": "none", "reason_count": 0})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def plot_feature_keep_reject(audit: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(
        len(FEATURE_COLUMNS),
        len(PARTICLE_ORDER),
        figsize=(2.4 * len(PARTICLE_ORDER), 1.25 * len(FEATURE_COLUMNS)),
        constrained_layout=True,
    )
    for row_idx, feature in enumerate(FEATURE_COLUMNS):
        for col_idx, particle in enumerate(PARTICLE_ORDER):
            ax = axes[row_idx, col_idx]
            sub = audit[audit["particle"] == particle]
            if sub.empty:
                ax.axis("off")
                continue
            values = sub[feature].astype(float)
            lo, hi = np.nanpercentile(values, [0.5, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = float(values.min()), float(values.max())
            bins = np.linspace(lo, hi, 60)
            kept = sub[sub["recommended_keep"]]
            rejected = sub[~sub["recommended_keep"]]
            ax.hist(kept[feature].astype(float), bins=bins, color="#4E79A7", alpha=0.45, label="keep")
            if not rejected.empty:
                ax.hist(
                    rejected[feature].astype(float),
                    bins=bins,
                    color="#E15759",
                    alpha=0.60,
                    label="reject",
                )
            if particle == "Am" and feature == "Npix":
                threshold = float(sub["am_npix_threshold"].iloc[0])
                ax.axvline(threshold, color="black", linewidth=0.8, linestyle="--")
            if row_idx == 0:
                ax.set_title(particle)
            if col_idx == 0:
                ax.set_ylabel(f"{feature}\ncount")
            if row_idx == len(FEATURE_COLUMNS) - 1:
                ax.set_xlabel("feature value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Stage-3a conservative cleaning audit: keep/reject feature histograms")
    save_figure(fig, output_dir / "stage3a_keep_reject_feature_histograms")
    plt.close(fig)


def transform_for_pca(sub: pd.DataFrame) -> np.ndarray:
    transformed = pd.DataFrame(index=sub.index)
    transformed["Npix"] = np.log1p(sub["Npix"].astype(float))
    transformed["S_total_ToT"] = np.log1p(sub["S_total_ToT"].astype(float))
    transformed["Pmax"] = sub["Pmax"].astype(float)
    transformed["Rg"] = np.log1p(sub["Rg"].astype(float))
    transformed["E_pca"] = np.log1p(np.maximum(sub["E_pca"].astype(float) - 1.0, 0.0))
    transformed["Fbox"] = sub["Fbox"].astype(float)

    values = transformed.to_numpy(dtype=float)
    med = np.nanmedian(values, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    return (values - med) / iqr


def plot_pca_keep_reject(audit: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(PARTICLE_ORDER), figsize=(3.0 * len(PARTICLE_ORDER), 2.7), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    for ax, particle in zip(axes, PARTICLE_ORDER):
        sub = audit[audit["particle"] == particle].copy()
        if len(sub) < 3:
            ax.axis("off")
            continue
        x = transform_for_pca(sub)
        pcs = PCA(n_components=2, random_state=42).fit_transform(x)
        keep = sub["recommended_keep"].to_numpy(dtype=bool)
        ax.scatter(pcs[keep, 0], pcs[keep, 1], s=2, alpha=0.20, c="#4E79A7", label="keep", linewidths=0)
        ax.scatter(pcs[~keep, 0], pcs[~keep, 1], s=5, alpha=0.60, c="#E15759", label="reject", linewidths=0)
        ax.set_title(particle)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Stage-3a conservative cleaning audit: PCA keep/reject overlay")
    save_figure(fig, output_dir / "stage3a_keep_reject_pca_overlay")
    plt.close(fig)


def center_nonzero_to_canvas(matrix: np.ndarray, size: int = 10) -> np.ndarray:
    coords = np.argwhere(matrix > 0)
    canvas = np.zeros((size, size), dtype=np.float32)
    if coords.size == 0:
        return canvas
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    crop = matrix[r0:r1, c0:c1]
    height, width = crop.shape
    if height > size or width > size:
        crop = crop[:size, :size]
        height, width = crop.shape
    top = (size - height) // 2
    left = (size - width) // 2
    canvas[top : top + height, left : left + width] = crop
    return canvas


def sample_rejected_by_reason(audit: pd.DataFrame, samples_per_reason: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rejected = audit[~audit["recommended_keep"]].copy()
    parts: list[pd.DataFrame] = []
    for (particle, reason), sub in rejected.assign(
        primary_reason=rejected["reject_reasons"].str.split(";").str[0]
    ).groupby(["particle", "primary_reason"], sort=True):
        take = min(samples_per_reason, len(sub))
        random_state = int(rng.integers(0, 2**31 - 1))
        picked = sub.sample(n=take, random_state=random_state).copy()
        picked["primary_reason"] = reason
        picked["reason_total_count"] = int(len(sub))
        parts.append(picked)
    if not parts:
        return pd.DataFrame(columns=list(audit.columns) + ["primary_reason", "reason_total_count"])
    return pd.concat(parts, ignore_index=True)


def plot_rejected_samples(sampled: pd.DataFrame, stage1_root: Path, output_dir: Path, crop_size: int) -> None:
    if sampled.empty:
        return
    for particle, sub_particle in sampled.groupby("particle", sort=True):
        reasons = list(sub_particle["primary_reason"].drop_duplicates())
        max_cols = int(sub_particle.groupby("primary_reason").size().max())
        fig, axes = plt.subplots(
            len(reasons),
            max_cols,
            figsize=(1.45 * max_cols, 1.55 * len(reasons)),
            constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(len(reasons), max_cols)
        for ax in axes.ravel():
            ax.axis("off")
        values_for_scale: list[float] = []
        canvases: list[tuple[pd.Series, np.ndarray]] = []
        for _, row in sub_particle.iterrows():
            matrix = np.loadtxt(stage1_root / "dataset" / str(row["tot_path"]), dtype=np.float32)
            canvas = center_nonzero_to_canvas(matrix, size=crop_size)
            canvases.append((row, canvas))
            nonzero = canvas[canvas > 0]
            if nonzero.size:
                values_for_scale.extend(np.log1p(nonzero).tolist())
        vmax = float(np.quantile(values_for_scale, 0.98)) if values_for_scale else 1.0

        for row_idx, reason in enumerate(reasons):
            entries = [(row, canvas) for row, canvas in canvases if row["primary_reason"] == reason]
            for col_idx, (row, canvas) in enumerate(entries):
                ax = axes[row_idx, col_idx]
                ax.imshow(np.log1p(canvas), cmap="magma", interpolation="nearest", vmin=0, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(
                    f"N={int(row['Npix'])}, ToT={row['S_total_ToT']:.0f}\n"
                    f"P={row['Pmax']:.2f}, F={row['Fbox']:.2f}",
                    fontsize=6,
                )
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{reason}\nn={int(row['reason_total_count'])}",
                        rotation=0,
                        labelpad=35,
                        va="center",
                    )
        fig.suptitle(f"{particle} rejected-event samples by Stage-3a reason")
        save_figure(fig, output_dir / f"{particle}_stage3a_rejected_samples_{crop_size}x{crop_size}")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--stage2a-root", type=Path, default=DEFAULT_STAGE2A_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples-per-reason", type=int, default=10)
    parser.add_argument("--crop-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_style()
    args.output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_root / "figures"

    features_path = args.stage2a_root / "features_raw.csv"
    if not features_path.is_file():
        raise FileNotFoundError(f"Missing Stage-2a feature table: {features_path}")
    features = pd.read_csv(features_path)
    missing = [column for column in FEATURE_COLUMNS + ["particle", "angle", "tot_path"] if column not in features.columns]
    if missing:
        raise ValueError(f"Missing required columns in {features_path}: {missing}")

    thresholds = build_group_thresholds(features, group_cols=["particle", "angle"])
    am_threshold = suggest_am_npix_threshold(features)
    audit = apply_cleaning_flags(features, thresholds, am_threshold)
    rejected = audit[~audit["recommended_keep"]].copy()
    sampled_rejected = sample_rejected_by_reason(rejected, samples_per_reason=args.samples_per_reason, seed=args.seed)

    audit.to_csv(args.output_root / "source_cleaning_audit.csv", index=False)
    rejected.to_csv(args.output_root / "rejected_candidate_audit.csv", index=False)
    sampled_rejected.to_csv(args.output_root / "rejected_sample_manifest.csv", index=False)
    write_rule_summary(thresholds, am_threshold, args.output_root / "cleaning_rule_summary.csv")
    write_counts(audit, args.output_root / "cleaning_counts_by_source_angle.csv")

    plot_feature_keep_reject(audit, figures_dir)
    plot_pca_keep_reject(audit, figures_dir)
    plot_rejected_samples(sampled_rejected, args.stage1_root, figures_dir, crop_size=args.crop_size)

    summary = {
        "stage": "stage3a_source_cleaning_audit_v1",
        "stage1_root": str(args.stage1_root),
        "stage2a_root": str(args.stage2a_root),
        "output_root": str(args.output_root),
        "policy": "source-label conservative cleaning audit only; no final cleaned dataset is exported.",
        "am_rule": "Am uses a simple global Npix threshold estimated by 1D Otsu over Am Npix.",
        "co_sr_rule": "Co60/Sr keep source labels and only reject low-signal candidates. Extreme-large, sparse-shape, or multi-feature outlier candidates are retained and written as review_flags.",
        "am_npix_threshold": float(am_threshold),
        "total_candidates": int(len(audit)),
        "kept_candidates": int(audit["recommended_keep"].sum()),
        "rejected_candidates": int((~audit["recommended_keep"]).sum()),
        "reject_rate": float((~audit["recommended_keep"]).mean()) if len(audit) else 0.0,
        "outputs": {
            "source_cleaning_audit.csv": "All candidates with recommended_keep and reject_reasons.",
            "rejected_candidate_audit.csv": "Rejected-only audit rows.",
            "cleaning_rule_summary.csv": "Per-source/per-angle thresholds plus Am global threshold.",
            "cleaning_counts_by_source_angle.csv": "Counts and rejection reasons by source and angle.",
            "rejected_sample_manifest.csv": "Traceable rejected samples used in the crop figures.",
            "figures/*": "Keep/reject histograms, PCA overlay, and rejected crop samples.",
        },
    }
    (args.output_root / "stage3a_notes.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Stage-3a audit written to: {args.output_root}")
    print(f"Total candidates: {summary['total_candidates']}")
    print(f"Rejected candidates: {summary['rejected_candidates']} ({summary['reject_rate']:.2%})")
    print(f"Am Npix threshold: {am_threshold:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
