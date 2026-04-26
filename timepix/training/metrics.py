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


def classification_metrics(logits: np.ndarray, y_true: np.ndarray, angle_values: list[float]) -> dict:
    num_classes = len(angle_values)
    if y_true.size == 0:
        return {"accuracy": 0.0, "mae_argmax": 0.0, "mae_weighted": 0.0, "macro_f1": 0.0}

    probs = _softmax(logits)
    y_pred = probs.argmax(axis=1)
    accuracy = float(np.mean(y_pred == y_true))
    angles = np.asarray(angle_values, dtype=float)
    true_angles = angles[y_true.astype(int)]
    pred_angles = angles[y_pred.astype(int)]
    weighted_angles = probs @ angles
    mae_argmax = float(np.mean(np.abs(pred_angles - true_angles)))
    mae_weighted = float(np.mean(np.abs(weighted_angles - true_angles)))
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

    return {
        "accuracy": accuracy,
        "mae_argmax": mae_argmax,
        "mae_weighted": mae_weighted,
        "macro_f1": float(np.mean(f1_values)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def regression_metrics(predictions: np.ndarray, targets: np.ndarray, max_angle: float) -> dict:
    pred_angles = predictions * max_angle
    true_angles = targets * max_angle
    errors = pred_angles - true_angles
    return {
        "mae": float(np.mean(np.abs(errors))) if errors.size else 0.0,
        "rmse": float(np.sqrt(np.mean(errors**2))) if errors.size else 0.0,
    }

