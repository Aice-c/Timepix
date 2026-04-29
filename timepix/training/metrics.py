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


def classification_metrics(logits: np.ndarray, y_true: np.ndarray, angle_values: list[float]) -> dict:
    num_classes = len(angle_values)
    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "mae_argmax": 0.0,
            "mae_weighted": 0.0,
            "p90_error": 0.0,
            "p90_error_weighted": 0.0,
            "macro_f1": 0.0,
        }

    probs = _softmax(logits)
    y_pred = probs.argmax(axis=1)
    accuracy = float(np.mean(y_pred == y_true))
    angles = np.asarray(angle_values, dtype=float)
    true_angles = angles[y_true.astype(int)]
    pred_angles = angles[y_pred.astype(int)]
    weighted_angles = probs @ angles
    abs_errors_argmax = np.abs(pred_angles - true_angles)
    abs_errors_weighted = np.abs(weighted_angles - true_angles)
    mae_argmax = float(np.mean(abs_errors_argmax))
    mae_weighted = float(np.mean(abs_errors_weighted))
    cm = confusion_matrix(y_true, y_pred, num_classes)

    per_class = []
    f1_values = []
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
        f1_values.append(f1)
        per_class.append({"class_index": cls, "precision": precision, "recall": recall, "f1": f1})

    def pair_confusions(a: float, b: float) -> int:
        matches_a = np.where(np.isclose(angles, a))[0]
        matches_b = np.where(np.isclose(angles, b))[0]
        if matches_a.size == 0 or matches_b.size == 0:
            return 0
        i = int(matches_a[0])
        j = int(matches_b[0])
        return int(cm[i, j] + cm[j, i])

    high_angle_f1 = [f1 for f1, angle in zip(f1_values, angles) if angle >= 45.0]

    return {
        "accuracy": accuracy,
        "mae_argmax": mae_argmax,
        "mae_weighted": mae_weighted,
        "p90_error": p90_error(abs_errors_argmax),
        "p90_error_weighted": p90_error(abs_errors_weighted),
        "macro_f1": float(np.mean(f1_values)),
        "high_angle_macro_f1": float(np.mean(high_angle_f1)) if high_angle_f1 else 0.0,
        "confusion_45_50": pair_confusions(45.0, 50.0),
        "confusion_60_70": pair_confusions(60.0, 70.0),
        "far_error_rate_abs_ge_20": float(np.mean(abs_errors_argmax >= 20.0)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


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
