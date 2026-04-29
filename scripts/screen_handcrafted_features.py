#!/usr/bin/env python
"""Screen handcrafted scalar features without training a CNN."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timepix.config import load_experiment_config, resolve_project_path
from timepix.config_validation import validate_experiment_config


def _load_sklearn() -> None:
    global RandomForestClassifier, LogisticRegression, accuracy_score, f1_score, permutation_importance
    try:
        from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier
        from sklearn.inspection import permutation_importance as _permutation_importance
        from sklearn.linear_model import LogisticRegression as _LogisticRegression
        from sklearn.metrics import accuracy_score as _accuracy_score
        from sklearn.metrics import f1_score as _f1_score
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "scikit-learn is required for handcrafted feature screening. "
            "Install the analysis/local environment dependencies first."
        ) from exc

    RandomForestClassifier = _RandomForestClassifier
    LogisticRegression = _LogisticRegression
    accuracy_score = _accuracy_score
    f1_score = _f1_score
    permutation_importance = _permutation_importance


def _load_timepix_data() -> None:
    global ALPHA_A5_FEATURES, TOT_ONLY_A5_FEATURES
    global HandcraftedFeatureExtractor, HandcraftedFeatureScaler, build_dataloaders, load_feature_arrays
    try:
        from timepix.data.builders import build_dataloaders as _build_dataloaders
        from timepix.data.features import ALPHA_A5_FEATURES as _ALPHA_A5_FEATURES
        from timepix.data.features import TOT_ONLY_A5_FEATURES as _TOT_ONLY_A5_FEATURES
        from timepix.data.features import HandcraftedFeatureExtractor as _HandcraftedFeatureExtractor
        from timepix.data.features import HandcraftedFeatureScaler as _HandcraftedFeatureScaler
        from timepix.data.features import load_feature_arrays as _load_feature_arrays
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "torch and the Timepix data dependencies are required for handcrafted feature extraction. "
            "Run this script inside the project training/analysis environment."
        ) from exc

    build_dataloaders = _build_dataloaders
    ALPHA_A5_FEATURES = _ALPHA_A5_FEATURES
    TOT_ONLY_A5_FEATURES = _TOT_ONLY_A5_FEATURES
    HandcraftedFeatureExtractor = _HandcraftedFeatureExtractor
    HandcraftedFeatureScaler = _HandcraftedFeatureScaler
    load_feature_arrays = _load_feature_arrays


FEATURE_GROUPS = {
    "geometry": [
        "active_pixel_count",
        "bbox_long",
        "bbox_short",
        "bbox_fill_ratio",
        "pca_major_axis",
        "pca_minor_axis",
    ],
    "tot": [
        "total_ToT",
        "ToT_density",
    ],
    "toa": [
        "ToA_span",
        "ToA_p90_minus_p10",
    ],
    "axis_interaction": [
        "ToA_major_axis_slope_abs",
        "ToA_major_axis_corr_abs",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen A5 handcrafted scalar features")
    parser.add_argument("--config", required=True, help="A5a YAML config")
    parser.add_argument("--data-root", default=None, help="Override dataset.root for this machine")
    parser.add_argument("--out-dir", default="outputs/a5_feature_screening", help="Output directory")
    parser.add_argument("--name", default=None, help="Output run name; defaults to experiment_name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for traditional models")
    parser.add_argument("--n-repeats", type=int, default=20, help="Permutation repeats on validation")
    parser.add_argument("--include-test-metrics", action="store_true", help="Also report test metrics for diagnostics")
    return parser.parse_args()


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_as_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _collect_split_features(dataset) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    extractor: HandcraftedFeatureExtractor | None = dataset.feature_extractor
    scaler: HandcraftedFeatureScaler | None = dataset.feature_scaler
    if extractor is None or extractor.dim == 0:
        raise ValueError("A5a requires handcrafted_features.enabled=true with at least one feature")

    feature_names = extractor.feature_names
    xs: list[np.ndarray] = []
    ys: list[int] = []
    rows: list[dict] = []
    for record in dataset.records:
        arrays = load_feature_arrays(
            record,
            extractor.required_modalities,
            data_dtype=dataset.data_dtype,
            crop_size=dataset.crop_size,
        )
        vec = extractor.extract(arrays)
        if vec is None:
            continue
        if scaler is not None:
            vec = scaler.apply(vec)
        values = vec.numpy().astype(float)
        xs.append(values)
        ys.append(int(record.label))
        row = {
            "sample_key": record.key,
            "label": int(record.label),
            "angle": dataset.label_map[int(record.label)],
        }
        row.update({name: float(value) for name, value in zip(feature_names, values)})
        rows.append(row)
    if not xs:
        raise ValueError("No handcrafted features were extracted")
    return np.stack(xs, axis=0), np.asarray(ys, dtype=int), rows


def _angle_mae(y_true: np.ndarray, y_pred: np.ndarray, label_map: dict[int, str]) -> float:
    angles = {int(k): float(v) for k, v in label_map.items()}
    true_angles = np.asarray([angles[int(y)] for y in y_true], dtype=float)
    pred_angles = np.asarray([angles[int(y)] for y in y_pred], dtype=float)
    return float(np.mean(np.abs(true_angles - pred_angles)))


def _metrics(model, x: np.ndarray, y: np.ndarray, label_map: dict[int, str]) -> dict[str, float]:
    pred = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "mae_argmax": _angle_mae(y, pred, label_map),
    }


def _feature_summary(rows: list[dict], feature_names: list[str]) -> list[dict]:
    summary = []
    for name in feature_names:
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
        summary.append(
            {
                "feature": name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "nan_count": int(np.isnan(values).sum()),
            }
        )
    return summary


def _group_columns(feature_names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for group, names in FEATURE_GROUPS.items():
        idxs = [feature_names.index(name) for name in names if name in feature_names]
        if idxs:
            groups[group] = idxs
    return groups


def _group_permutation_importance(
    model,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    rng: np.random.Generator,
    repeats: int,
) -> list[dict]:
    baseline = float(accuracy_score(y_val, model.predict(x_val)))
    rows = []
    for group, idxs in _group_columns(feature_names).items():
        drops = []
        for _ in range(repeats):
            x_perm = x_val.copy()
            order = rng.permutation(x_perm.shape[0])
            x_perm[:, idxs] = x_perm[order][:, idxs]
            score = float(accuracy_score(y_val, model.predict(x_perm)))
            drops.append(baseline - score)
        rows.append(
            {
                "group": group,
                "features": ";".join(feature_names[idx] for idx in idxs),
                "baseline_accuracy": baseline,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
            }
        )
    return rows


def _run_model_suite(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    label_map: dict[int, str],
    seed: int,
) -> dict[str, Any]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        ).fit(x_train, y_train),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        ).fit(x_train, y_train),
    }


def main() -> int:
    args = parse_args()
    _load_sklearn()
    _load_timepix_data()
    cfg = load_experiment_config(args.config)
    validate_experiment_config(cfg)

    loaders, data_info = build_dataloaders(cfg, data_root_override=args.data_root, eval_mode=True)
    label_map = {int(k): str(v) for k, v in data_info["label_map"].items()}
    feature_names = list(data_info.get("handcrafted_features", []))
    if not feature_names:
        raise ValueError("A5a config produced zero handcrafted features")

    run_name = args.name or str(cfg.get("experiment_name") or "a5a_handcrafted_screening")
    out_dir = resolve_project_path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    split_payload = {}
    for split_name, loader in loaders.items():
        x, y, rows = _collect_split_features(loader.dataset)
        split_payload[split_name] = {"x": x, "y": y, "rows": rows}
        _write_rows(
            out_dir / f"features_{split_name}.csv",
            rows,
            ["sample_key", "label", "angle", *feature_names],
        )

    x_train = split_payload["train"]["x"]
    y_train = split_payload["train"]["y"]
    x_val = split_payload["val"]["x"]
    y_val = split_payload["val"]["y"]

    models = _run_model_suite(x_train, y_train, x_val, y_val, label_map, args.seed)
    metrics_rows = []
    permutation_rows = []
    group_rows = []
    rng = np.random.default_rng(args.seed)
    for model_name, model in models.items():
        for split_name in ("train", "val"):
            metrics = _metrics(model, split_payload[split_name]["x"], split_payload[split_name]["y"], label_map)
            metrics_rows.append({"model": model_name, "split": split_name, **metrics})
        if args.include_test_metrics:
            metrics = _metrics(model, split_payload["test"]["x"], split_payload["test"]["y"], label_map)
            metrics_rows.append({"model": model_name, "split": "test", **metrics})

        pi = permutation_importance(
            model,
            x_val,
            y_val,
            n_repeats=args.n_repeats,
            random_state=args.seed,
            scoring="accuracy",
            n_jobs=-1,
        )
        for idx, name in enumerate(feature_names):
            permutation_rows.append(
                {
                    "model": model_name,
                    "feature": name,
                    "importance_mean": float(pi.importances_mean[idx]),
                    "importance_std": float(pi.importances_std[idx]),
                }
            )
        for row in _group_permutation_importance(model, x_val, y_val, feature_names, rng, args.n_repeats):
            group_rows.append({"model": model_name, **row})

    _write_rows(
        out_dir / "model_metrics.csv",
        metrics_rows,
        ["model", "split", "accuracy", "macro_f1", "mae_argmax"],
    )
    _write_rows(
        out_dir / "permutation_importance_val.csv",
        sorted(permutation_rows, key=lambda row: (row["model"], -row["importance_mean"])),
        ["model", "feature", "importance_mean", "importance_std"],
    )
    _write_rows(
        out_dir / "group_permutation_importance_val.csv",
        sorted(group_rows, key=lambda row: (row["model"], -row["importance_mean"])),
        ["model", "group", "features", "baseline_accuracy", "importance_mean", "importance_std"],
    )
    _write_rows(
        out_dir / "feature_summary_train.csv",
        _feature_summary(split_payload["train"]["rows"], feature_names),
        ["feature", "mean", "std", "min", "max", "nan_count"],
    )

    corr = np.corrcoef(x_train, rowvar=False)
    with (out_dir / "feature_correlation_train.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", *feature_names])
        for name, row in zip(feature_names, corr):
            writer.writerow([name, *[float(value) for value in row]])

    scaler = loaders["train"].dataset.feature_scaler
    _write_json(
        out_dir / "feature_metadata.json",
        {
            "config_path": str(cfg.get("_config_path")),
            "dataset": cfg.get("dataset", {}),
            "split_path": data_info.get("split_path"),
            "split_counts": data_info.get("split_counts"),
            "feature_names": feature_names,
            "feature_source_modalities": data_info.get("feature_source_modalities"),
            "tot_only_a5_features": [name for name in TOT_ONLY_A5_FEATURES if name in feature_names],
            "alpha_a5_features": [name for name in ALPHA_A5_FEATURES if name in feature_names],
            "standardized": scaler is not None,
            "feature_means": scaler.means.numpy() if scaler is not None else None,
            "feature_stds": scaler.stds.numpy() if scaler is not None else None,
            "test_metrics_included": bool(args.include_test_metrics),
            "note": "Feature selection must use train/validation only. Test rows are saved for later final reporting.",
        },
    )

    print(f"A5 handcrafted feature screening written to: {out_dir}")
    print(f"Features: {', '.join(feature_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
