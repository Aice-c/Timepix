"""Traditional machine-learning baselines for analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .progress import iter_progress
from .representative import deterministic_sample


def numeric_feature_columns(features: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    default_exclude = {"dataset", "angle", "angle_value", "modality", "sample_key", "path", "feature_error"}
    exclude = exclude | default_exclude
    cols = []
    for col in features.columns:
        if col in exclude:
            continue
        values = pd.to_numeric(features[col], errors="coerce")
        if values.notna().sum() > 0:
            cols.append(col)
    return cols


def _prepare_xy(features: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    clean = features.copy()
    x = clean[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = x.notna().all(axis=1)
    x = x.loc[keep].to_numpy(dtype=float)
    y_angles = clean.loc[keep, "angle_value"].to_numpy(dtype=float)
    labels = sorted(float(v) for v in np.unique(y_angles))
    label_to_idx = {angle: idx for idx, angle in enumerate(labels)}
    y = np.asarray([label_to_idx[float(angle)] for angle in y_angles], dtype=int)
    return x, y, labels


def _angle_mae(y_true: np.ndarray, y_pred: np.ndarray, labels: list[float]) -> float:
    angles = np.asarray(labels, dtype=float)
    return float(np.mean(np.abs(angles[y_true] - angles[y_pred])))


def _build_models(seed: int, include_rbf: bool):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC, SVC

    models = [
        ("dummy_most_frequent", DummyClassifier(strategy="most_frequent")),
        ("dummy_stratified", DummyClassifier(strategy="stratified", random_state=seed)),
        (
            "logistic_regression",
            make_pipeline(
                StandardScaler(),
                OneVsRestClassifier(
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, solver="liblinear")
                ),
            ),
        ),
        ("linear_svm", make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000))),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=-1),
        ),
        (
            "mlp",
            make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, early_stopping=True, random_state=seed),
            ),
        ),
    ]
    if include_rbf:
        models.append(("rbf_svm", make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", random_state=seed))))
    return models


def run_ml_baselines(
    features: pd.DataFrame,
    *,
    feature_cols: list[str],
    sample_cap: int,
    seeds: list[int],
    include_rbf: bool = True,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str]]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split

    rows = []
    confusion_by_model: dict[str, np.ndarray] = {}
    labels_out: list[str] = []
    for seed in seeds:
        sampled = deterministic_sample(features, sample_cap, seed, stratify="angle")
        x, y, labels = _prepare_xy(sampled, feature_cols)
        labels_out = [f"{angle:g}" for angle in labels]
        if len(np.unique(y)) < 2:
            continue
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=seed)
        rbf_ok = include_rbf and len(x_train) <= 5000
        models = _build_models(seed, rbf_ok)
        for name, model in iter_progress(models, total=len(models), desc=f"ML baselines seed={seed}", unit="model"):
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            cm = confusion_matrix(y_test, y_pred, labels=list(range(len(labels))))
            if seed == seeds[0] and name in {"logistic_regression", "random_forest", "mlp"}:
                confusion_by_model[name] = cm
            rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                    "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                    "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                    "mae_in_degrees": _angle_mae(y_test, y_pred, labels),
                }
            )
    return pd.DataFrame(rows), confusion_by_model, labels_out


def aggregate_ml_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mae_in_degrees"]
    rows = []
    for model, group in results.groupby("model"):
        row = {"model": model, "runs": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("balanced_accuracy_mean", ascending=False)


def pairwise_auc_by_gap(features: pd.DataFrame, feature_cols: list[str], seeds: list[int], sample_cap: int) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    angles = sorted(float(v) for v in features["angle_value"].dropna().unique())
    tasks = [(left, right, seed) for i, left in enumerate(angles) for right in angles[i + 1 :] for seed in seeds]
    rows = []
    for left, right, seed in iter_progress(tasks, total=len(tasks), desc="Pairwise AUC", unit="fit"):
        gap = right - left
        subset = features[features["angle_value"].isin([left, right])]
        sampled = deterministic_sample(subset, sample_cap, seed, stratify="angle")
        x, y, labels = _prepare_xy(sampled, feature_cols)
        if len(np.unique(y)) != 2:
            continue
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=seed)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, solver="liblinear"),
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        if hasattr(model, "decision_function"):
            score = model.decision_function(x_test)
        else:
            score = model.predict_proba(x_test)[:, 1]
        rows.append(
            {
                "seed": seed,
                "angle_left": left,
                "angle_right": right,
                "angle_pair": f"{left:g}-{right:g}",
                "angle_gap": gap,
                "auc": float(roc_auc_score(y_test, score)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            }
        )
    return pd.DataFrame(rows)


def auc_by_gap(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    return (
        pairwise.groupby("angle_gap", as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            pairs=("angle_pair", "nunique"),
        )
        .sort_values("angle_gap")
    )
