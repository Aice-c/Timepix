"""Metrics for classification and regression."""

from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix


def p90_error(abs_errors: np.ndarray) -> float:
    """Return the 90th percentile of absolute angle errors in degrees."""
    if abs_errors.size == 0:
        return 0.0
    return float(np.percentile(abs_errors, 90))


def _classification_core(logits: np.ndarray, y_true: np.ndarray, class_names: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    num_classes = len(class_names)
    if y_true.size == 0:
        return (
            {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "confusion_matrix": np.zeros((num_classes, num_classes), dtype=int).tolist(),
                "per_class": [],
            },
            np.asarray([], dtype=int),
            np.zeros((num_classes, num_classes), dtype=int),
        )

    probs = _softmax(logits)
    y_pred = probs.argmax(axis=1)
    accuracy = float(np.mean(y_pred == y_true))
    cm = confusion_matrix(y_true, y_pred, num_classes)

    per_class = []
    f1_values = []
    recall_values = []
    weighted_f1_sum = 0.0
    total_support = int(cm.sum())
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        support = int(cm[cls, :].sum())
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
        f1_values.append(f1)
        recall_values.append(recall)
        weighted_f1_sum += f1 * support
        per_class.append(
            {
                "class_index": cls,
                "class_name": class_names[cls],
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return (
        {
            "accuracy": accuracy,
            "balanced_accuracy": float(np.mean(recall_values)) if recall_values else 0.0,
            "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
            "weighted_f1": float(weighted_f1_sum / max(total_support, 1)),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
        },
        y_pred,
        cm,
    )


def categorical_classification_metrics(logits: np.ndarray, y_true: np.ndarray, class_names: list[str]) -> dict:
    metrics, _y_pred, _cm = _classification_core(logits, y_true, class_names)
    return metrics


def angle_classification_metrics(logits: np.ndarray, y_true: np.ndarray, angle_values: list[float]) -> dict:
    class_names = [f"{angle:g}" for angle in angle_values]
    if y_true.size == 0:
        metrics = categorical_classification_metrics(logits, y_true, class_names)
        metrics.update(
            {
                "mae_argmax": 0.0,
                "mae_weighted": 0.0,
                "p90_error": 0.0,
                "p90_error_weighted": 0.0,
                "high_angle_macro_f1": 0.0,
                "confusion_45_50": 0,
                "confusion_60_70": 0,
                "far_error_rate_abs_ge_20": 0.0,
            }
        )
        return metrics

    probs = _softmax(logits)
    y_pred = probs.argmax(axis=1)
    metrics, _y_pred, cm = _classification_core(logits, y_true, class_names)
    angles = np.asarray(angle_values, dtype=float)
    true_angles = angles[y_true.astype(int)]
    pred_angles = angles[y_pred.astype(int)]
    weighted_angles = probs @ angles
    abs_errors_argmax = np.abs(pred_angles - true_angles)
    abs_errors_weighted = np.abs(weighted_angles - true_angles)

    def pair_confusions(a: float, b: float) -> int:
        matches_a = np.where(np.isclose(angles, a))[0]
        matches_b = np.where(np.isclose(angles, b))[0]
        if matches_a.size == 0 or matches_b.size == 0:
            return 0
        i = int(matches_a[0])
        j = int(matches_b[0])
        return int(cm[i, j] + cm[j, i])

    f1_values = [item["f1"] for item in metrics["per_class"]]
    high_angle_f1 = [f1 for f1, angle in zip(f1_values, angles) if angle >= 45.0]
    metrics.update(
        {
            "mae_argmax": float(np.mean(abs_errors_argmax)),
            "mae_weighted": float(np.mean(abs_errors_weighted)),
            "p90_error": p90_error(abs_errors_argmax),
            "p90_error_weighted": p90_error(abs_errors_weighted),
            "high_angle_macro_f1": float(np.mean(high_angle_f1)) if high_angle_f1 else 0.0,
            "confusion_45_50": pair_confusions(45.0, 50.0),
            "confusion_60_70": pair_confusions(60.0, 70.0),
            "far_error_rate_abs_ge_20": float(np.mean(abs_errors_argmax >= 20.0)),
        }
    )
    return metrics


def classification_metrics(
    logits: np.ndarray,
    y_true: np.ndarray,
    angle_values: list[float] | None = None,
    *,
    label_type: str = "angle_folder",
    class_names: list[str] | None = None,
) -> dict:
    if label_type == "categorical_folder":
        if class_names is None:
            num_classes = int(logits.shape[1]) if logits.ndim == 2 else 0
            class_names = [str(i) for i in range(num_classes)]
        return categorical_classification_metrics(logits, y_true, class_names)
    if angle_values is None:
        raise ValueError("angle_values is required for angle_folder classification metrics")
    return angle_classification_metrics(logits, y_true, angle_values)


def regression_metrics(predictions: np.ndarray, targets: np.ndarray, max_angle: float) -> dict:
    pred_angles = predictions * max_angle
    true_angles = targets * max_angle
    errors = pred_angles - true_angles
    abs_errors = np.abs(errors)
    return {
        "mae": float(np.mean(abs_errors)) if errors.size else 0.0,
        "rmse": float(np.sqrt(np.mean(errors**2))) if errors.size else 0.0,
        "p90_error": p90_error(abs_errors),
    }
