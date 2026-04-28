"""Plotting helpers for analysis outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_matrix


PNG_DPI = 300
CORE_FEATURES = ["active_count", "active_sum", "bbox_aspect_ratio", "rms_radius", "energy_entropy"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig, path_without_suffix: str | Path, *, dpi: int = PNG_DPI) -> tuple[Path, Path]:
    path = Path(path_without_suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    png = path.with_suffix(".png")
    pdf = path.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def plot_preprocessing_pipeline(out_path: str | Path) -> tuple[Path, Path]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    labels = [
        "Raw 256x256 detector frame\nmany hits in one time window",
        "Connected-component\ntrajectory extraction",
        "ToT-statistic cleaning\nactive_count / active_sum",
        "Final supervised datasets\nAlpha_100 / Proton_C",
    ]
    xs = np.linspace(0.08, 0.92, len(labels))
    for x, label in zip(xs, labels):
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f4f6f8", edgecolor="#4b5563"),
            transform=ax.transAxes,
        )
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate(
            "",
            xy=(right - 0.12, 0.55),
            xytext=(left + 0.12, 0.55),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#374151"),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
        )
    ax.set_title("Timepix Data Processing Pipeline", fontsize=13)
    return save_figure(fig, out_path)


def plot_class_counts(counts: pd.DataFrame, out_path: str | Path, title: str) -> tuple[Path, Path]:
    plt = _plt()
    if counts.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return save_figure(fig, out_path)
    pivot = counts.pivot_table(index="angle", columns="modality", values="count", aggfunc="sum").fillna(0)
    pivot = pivot.loc[sorted(pivot.index, key=lambda x: float(x))]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Sample count")
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, out_path)


def _display_matrix(array: np.ndarray) -> np.ndarray:
    x = np.asarray(array, dtype=float)
    if np.nanmax(x) > 0:
        x = np.log1p(np.clip(x, 0, None))
    return x


def plot_representative_grid(representatives: pd.DataFrame, out_path: str | Path, title: str, *, max_cols: int = 6) -> tuple[Path, Path]:
    plt = _plt()
    reps = representatives.sort_values(["angle_value", "modality", "sample_key"]) if "angle_value" in representatives else representatives
    n = len(reps)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No representative samples", ha="center", va="center")
        return save_figure(fig, out_path)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, item in zip(axes.ravel(), reps.itertuples(index=False)):
        array = read_matrix(item.path)
        ax.imshow(_display_matrix(array), cmap="viridis", interpolation="nearest")
        ax.set_title(f"{item.angle} deg\n{item.modality}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    return save_figure(fig, out_path)


def plot_alpha_pair_grid(tot_reps: pd.DataFrame, toa_features: pd.DataFrame, out_path: str | Path) -> tuple[Path, Path]:
    plt = _plt()
    rows = []
    toa_lookup = {
        (row.dataset, row.sample_key): row.path
        for row in toa_features.itertuples(index=False)
        if getattr(row, "modality", None) == "ToA"
    }
    for row in tot_reps.itertuples(index=False):
        toa_path = toa_lookup.get((row.dataset, row.sample_key))
        if toa_path:
            rows.append((row.angle, row.path, toa_path))
    if not rows:
        return plot_representative_grid(tot_reps, out_path, "Alpha_100 representative ToT samples")
    fig, axes = plt.subplots(len(rows), 2, figsize=(5.2, 2.3 * len(rows)), squeeze=False)
    for idx, (angle, tot_path, toa_path) in enumerate(rows):
        axes[idx, 0].imshow(_display_matrix(read_matrix(tot_path)), cmap="viridis", interpolation="nearest")
        axes[idx, 0].set_title(f"{angle} deg ToT", fontsize=8)
        axes[idx, 1].imshow(_display_matrix(read_matrix(toa_path)), cmap="magma", interpolation="nearest")
        axes[idx, 1].set_title(f"{angle} deg ToA", fontsize=8)
        for ax in axes[idx]:
            ax.axis("off")
    fig.suptitle("Alpha_100 paired representative samples")
    return save_figure(fig, out_path)


def plot_feature_violin(features: pd.DataFrame, out_path: str | Path, title: str, feature_cols: list[str] | None = None) -> tuple[Path, Path]:
    plt = _plt()
    feature_cols = [f for f in (feature_cols or CORE_FEATURES) if f in features.columns]
    if not feature_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No feature data", ha="center", va="center")
        ax.set_title(title)
        return save_figure(fig, out_path)
    fig, axes = plt.subplots(1, len(feature_cols), figsize=(4 * len(feature_cols), 4), squeeze=False)
    for ax, feature in zip(axes.ravel(), feature_cols):
        data = []
        labels = []
        for angle, group in features.sort_values("angle_value").groupby("angle"):
            values = pd.to_numeric(group[feature], errors="coerce").dropna().to_numpy()
            if len(values) > 0:
                data.append(values)
                labels.append(str(angle))
        if data:
            ax.violinplot(data, showmeans=False, showmedians=True)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45)
        ax.set_title(feature)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    return save_figure(fig, out_path)


def plot_feature_kde(features: pd.DataFrame, out_path: str | Path, title: str, feature_cols: list[str] | None = None) -> tuple[Path, Path]:
    plt = _plt()
    feature_cols = [f for f in (feature_cols or CORE_FEATURES[:3]) if f in features.columns]
    if not feature_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No feature data", ha="center", va="center")
        ax.set_title(title)
        return save_figure(fig, out_path)
    fig, axes = plt.subplots(1, len(feature_cols), figsize=(4 * len(feature_cols), 4), squeeze=False)
    for ax, feature in zip(axes.ravel(), feature_cols):
        for angle, group in features.sort_values("angle_value").groupby("angle"):
            values = pd.to_numeric(group[feature], errors="coerce").dropna().to_numpy()
            if len(values) < 3:
                continue
            ax.hist(values, bins=60, density=True, histtype="step", label=str(angle), alpha=0.8)
        ax.set_title(feature)
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(title)
    return save_figure(fig, out_path)


def plot_feature_scatter(features: pd.DataFrame, out_path: str | Path, title: str) -> tuple[Path, Path]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 5.2))
    plotted = False
    for angle, group in features.sort_values("angle_value").groupby("angle"):
        ax.scatter(group["active_count"], group["active_sum"], s=5, alpha=0.35, label=str(angle))
        plotted = True
    ax.set_xlabel("active_count")
    ax.set_ylabel("active_sum")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(markerscale=2, fontsize=8)
    return save_figure(fig, out_path)


def plot_box_by_angle(features: pd.DataFrame, feature: str, out_path: str | Path, title: str) -> tuple[Path, Path]:
    plt = _plt()
    data = []
    labels = []
    for angle, group in features.sort_values("angle_value").groupby("angle"):
        values = pd.to_numeric(group[feature], errors="coerce").dropna().to_numpy()
        if len(values):
            data.append(values)
            labels.append(str(angle))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if data:
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel(feature)
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, out_path)


def plot_heatmap(matrix: pd.DataFrame, out_path: str | Path, title: str) -> tuple[Path, Path]:
    plt = _plt()
    if matrix.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return save_figure(fig, out_path)
    data = matrix.set_index(matrix.columns[0])
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * data.shape[1]), max(6, 0.18 * data.shape[0])))
    image = ax.imshow(data.to_numpy(dtype=float), aspect="auto", cmap="mako" if False else "viridis")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    step = max(1, data.shape[0] // 25)
    y_ticks = list(range(0, data.shape[0], step))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(data.index[i]) for i in y_ticks], fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.75)
    return save_figure(fig, out_path)


def plot_embedding(points: np.ndarray, labels: np.ndarray, out_path: str | Path, title: str) -> tuple[Path, Path]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for label in sorted(set(labels), key=lambda x: float(x)):
        mask = labels == label
        ax.scatter(points[mask, 0], points[mask, 1], s=5, alpha=0.45, label=str(label))
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.grid(alpha=0.2)
    ax.legend(markerscale=2, fontsize=8)
    return save_figure(fig, out_path)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: str | Path, title: str) -> tuple[Path, Path]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5.2))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.75)
    return save_figure(fig, out_path)


def plot_metric_by_gap(df: pd.DataFrame, metric: str, out_path: str | Path, title: str) -> tuple[Path, Path]:
    plt = _plt()
    grouped = df.groupby("angle_gap", as_index=False)[metric].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(grouped["angle_gap"], grouped["mean"], yerr=grouped["std"].fillna(0.0), marker="o", capsize=4)
    ax.set_xlabel("Angle gap (deg)")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    return save_figure(fig, out_path)


def plot_mean_images_by_angle(features: pd.DataFrame, out_path: str | Path, title: str, *, sample_cap: int = 1000) -> tuple[Path, Path]:
    plt = _plt()
    groups = list(features.sort_values("angle_value").groupby("angle"))
    if not groups:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return save_figure(fig, out_path)
    fig, axes = plt.subplots(1, len(groups), figsize=(2.4 * len(groups), 2.6), squeeze=False)
    means = []
    for (_, group), ax in zip(groups, axes.ravel()):
        paths = list(group["path"].head(sample_cap))
        arrays = [read_matrix(path) for path in paths]
        means.append(np.mean(arrays, axis=0))
        ax.imshow(_display_matrix(means[-1]), cmap="viridis", interpolation="nearest")
        ax.set_title(f"{group['angle'].iloc[0]} deg", fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    return save_figure(fig, out_path)


def plot_adjacent_difference_maps(features: pd.DataFrame, angles: list[float], out_path: str | Path, *, sample_cap: int = 1000) -> tuple[Path, Path]:
    plt = _plt()
    means = {}
    for angle in angles:
        group = features[features["angle_value"] == float(angle)]
        paths = list(group["path"].head(sample_cap))
        if paths:
            means[angle] = np.mean([read_matrix(path) for path in paths], axis=0)
    pairs = [(a, b) for a, b in zip(angles[:-1], angles[1:]) if a in means and b in means]
    if not pairs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return save_figure(fig, out_path)
    fig, axes = plt.subplots(1, len(pairs), figsize=(2.7 * len(pairs), 2.8), squeeze=False)
    vmax = max(float(np.max(np.abs(means[b] - means[a]))) for a, b in pairs)
    for ax, (a, b) in zip(axes.ravel(), pairs):
        im = ax.imshow(means[b] - means[a], cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{b:g} - {a:g} deg", fontsize=8)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    fig.suptitle("Adjacent mean ToT difference maps")
    return save_figure(fig, out_path)
