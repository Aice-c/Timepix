#!/usr/bin/env python
"""Search training hyperparameters with Optuna."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timepix.config import load_experiment_config, parse_override, resolve_project_path, set_by_dotted_key
from timepix.config_validation import validate_experiment_config
from timepix.training.logger import write_json, write_yaml
from timepix.utils.paths import slugify


DEFAULT_SEARCH_ROOT = Path("outputs/hparam_search")
DEFAULT_OBJECTIVE = "validation.accuracy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search")
    parser.add_argument("--config", required=True, help="Search YAML file")
    parser.add_argument("--data-root", default=None, help="Override dataset.root for this machine")
    parser.add_argument("--output-root", default=None, help="Override experiment output root")
    parser.add_argument("--search-root", default=None, help="Directory for search logs and best config")
    parser.add_argument("--study-name", default=None, help="Override Optuna study name")
    parser.add_argument("--storage", default=None, help="Override Optuna storage, e.g. sqlite:///outputs/optuna/study.db")
    parser.add_argument("--n-trials", type=int, default=None, help="Override search.n_trials")
    parser.add_argument("--timeout", type=int, default=None, help="Override search.timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the planned search without training")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop the study when a trial raises an exception")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value before search, e.g. --set training.epochs=5",
    )
    return parser.parse_args()


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _search_parameters(search_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    params = search_cfg.get("parameters")
    if not isinstance(params, Mapping) or not params:
        raise ValueError("search.parameters must be a non-empty mapping")
    return params


def _validate_parameter_spec(name: str, spec: Any) -> None:
    spec = _require_mapping(spec, f"search.parameters.{name}")
    kind = spec.get("type")
    if kind not in {"float", "int", "categorical"}:
        raise ValueError(f"search.parameters.{name}.type must be one of: float, int, categorical")
    if kind in {"float", "int"}:
        for key in ("low", "high"):
            if key not in spec:
                raise ValueError(f"search.parameters.{name}.{key} is required")
        low = spec["low"]
        high = spec["high"]
        try:
            valid_range = float(low) <= float(high)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"search.parameters.{name}.low/high must be numeric") from exc
        if not valid_range:
            raise ValueError(f"search.parameters.{name}.low must be <= high")
        if kind == "int":
            int(low)
            int(high)
    if kind == "categorical":
        choices = spec.get("choices", spec.get("values"))
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"search.parameters.{name}.choices must be a non-empty list")


def _validate_search_config(search_cfg: Mapping[str, Any]) -> None:
    if search_cfg.get("sampler", "tpe") not in {"tpe", "random"}:
        raise ValueError("search.sampler must be one of: tpe, random")
    if search_cfg.get("direction", "maximize") not in {"maximize", "minimize"}:
        raise ValueError("search.direction must be maximize or minimize")
    if "n_trials" in search_cfg and int(search_cfg["n_trials"]) <= 0:
        raise ValueError("search.n_trials must be > 0")
    if "timeout" in search_cfg and search_cfg["timeout"] is not None and int(search_cfg["timeout"]) <= 0:
        raise ValueError("search.timeout must be > 0")
    for name, spec in _search_parameters(search_cfg).items():
        if not isinstance(name, str) or not name:
            raise ValueError("search.parameters keys must be non-empty dotted config paths")
        _validate_parameter_spec(name, spec)


def _infer_direction(metric: str) -> str:
    lower = metric.lower()
    return "minimize" if any(token in lower for token in ("loss", "mae", "error", "rmse")) else "maximize"


def _sample_value(trial, name: str, spec: Mapping[str, Any]):
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, spec.get("choices", spec.get("values")))
    if kind == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    return trial.suggest_float(
        name,
        float(spec["low"]),
        float(spec["high"]),
        step=spec.get("step"),
        log=bool(spec.get("log", False)),
    )


def _preview_value(spec: Mapping[str, Any]):
    kind = spec["type"]
    if kind == "categorical":
        return spec.get("choices", spec.get("values"))[0]
    return spec["low"]


def _metric_from_metadata(metadata: Mapping[str, Any], metric: str) -> float:
    aliases = {
        "val_accuracy": "validation.accuracy",
        "test_accuracy": "test.accuracy",
        "val_mae_argmax": "validation.mae_argmax",
        "test_mae_argmax": "test.mae_argmax",
        "val_p90_error": "validation.p90_error",
        "test_p90_error": "test.p90_error",
        "best_score": "best_score",
    }
    metric = aliases.get(metric, metric)
    root: Any = metadata.get("metrics", {})
    if metric.startswith("metrics."):
        root = metadata
    cur = root
    for part in metric.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(f"Metric not found in metadata: {metric}")
    return float(cur)


def _sqlite_storage_parent(storage: str | None) -> Path | None:
    if not storage or not storage.startswith("sqlite:///"):
        return None
    db_path = storage.removeprefix("sqlite:///")
    return Path(db_path).parent


def _default_storage(study_name: str) -> str:
    return f"sqlite:///outputs/optuna/{slugify(study_name)}.db"


def _search_output_dir(search_cfg: Mapping[str, Any], study_name: str, override: str | None) -> Path:
    root = Path(override) if override else Path(search_cfg.get("output_root", DEFAULT_SEARCH_ROOT))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return resolve_project_path(root / f"{timestamp}_{slugify(study_name)}")


def _write_trials_csv(path: Path, rows: list[dict[str, Any]], param_names: list[str]) -> None:
    fields = [
        "trial",
        "state",
        "value",
        "objective_metric",
        "experiment_name",
        "experiment_dir",
        "error",
        *param_names,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _best_trial_config(base_cfg: dict[str, Any], study, base_name: str) -> dict[str, Any] | None:
    best_trial = _best_trial_or_none(study)
    if best_trial is None:
        return None
    cfg = copy.deepcopy(base_cfg)
    for key, value in best_trial.params.items():
        set_by_dotted_key(cfg, key, value)
    cfg["experiment_name"] = f"{base_name}_best_hparams"
    return cfg


def _best_trial_or_none(study):
    try:
        return study.best_trial
    except ValueError:
        return None


def _load_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise SystemExit("Optuna is required for hyperparameter search. Install it with: pip install optuna") from exc
    return optuna


def _make_sampler(optuna, search_cfg: Mapping[str, Any]):
    seed = int(search_cfg.get("seed", 42))
    sampler = search_cfg.get("sampler", "tpe")
    if sampler == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    n_startup_trials = int(search_cfg.get("n_startup_trials", 5))
    return optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials, multivariate=True)


def main() -> int:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item}")
        key, value = item.split("=", 1)
        set_by_dotted_key(cfg, key, parse_override(value))

    search_cfg = _require_mapping(cfg.pop("search", None), "search")
    if args.n_trials is not None:
        search_cfg = dict(search_cfg)
        search_cfg["n_trials"] = args.n_trials
    if args.timeout is not None:
        search_cfg = dict(search_cfg)
        search_cfg["timeout"] = args.timeout
    if args.study_name is not None:
        search_cfg = dict(search_cfg)
        search_cfg["study_name"] = args.study_name
    if args.storage is not None:
        search_cfg = dict(search_cfg)
        search_cfg["storage"] = args.storage

    _validate_search_config(search_cfg)
    params = _search_parameters(search_cfg)
    for name, spec in params.items():
        preview_cfg = copy.deepcopy(cfg)
        set_by_dotted_key(preview_cfg, name, _preview_value(spec))
        validate_experiment_config(preview_cfg)
    validate_experiment_config(cfg)

    metric = str(search_cfg.get("objective", search_cfg.get("metric", DEFAULT_OBJECTIVE)))
    direction = str(search_cfg.get("direction") or _infer_direction(metric))
    n_trials = int(search_cfg.get("n_trials", 20))
    timeout = search_cfg.get("timeout")
    timeout = int(timeout) if timeout is not None else None
    base_name = str(cfg.get("experiment_name", "hparam_search"))
    study_name = str(search_cfg.get("study_name", base_name))
    storage = search_cfg.get("storage", _default_storage(study_name))

    if args.dry_run:
        print(f"Study: {study_name}")
        print(f"Sampler: {search_cfg.get('sampler', 'tpe')}")
        print(f"Direction: {direction}")
        print(f"Objective: {metric}")
        print(f"Trials: {n_trials}")
        print("Search parameters:")
        for name, spec in params.items():
            print(f"  {name}: {json.dumps(spec, ensure_ascii=False)}")
        return 0

    storage_parent = _sqlite_storage_parent(str(storage))
    if storage_parent is not None:
        resolve_project_path(storage_parent).mkdir(parents=True, exist_ok=True)

    optuna = _load_optuna()
    from timepix.training.runner import run_experiment

    search_dir = _search_output_dir(search_cfg, study_name, args.search_root)
    search_dir.mkdir(parents=True, exist_ok=False)
    write_yaml(search_dir / "search_config.yaml", {"search": dict(search_cfg), "base_experiment": cfg})

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=bool(search_cfg.get("load_if_exists", True)),
        sampler=_make_sampler(optuna, search_cfg),
    )

    trial_rows: list[dict[str, Any]] = []
    param_names = list(params.keys())

    def objective(trial) -> float:
        run_cfg = copy.deepcopy(cfg)
        sampled = {}
        for name, spec in params.items():
            value = _sample_value(trial, name, spec)
            sampled[name] = value
            set_by_dotted_key(run_cfg, name, value)
        run_cfg["experiment_name"] = f"{base_name}_trial_{trial.number:04d}"
        validate_experiment_config(run_cfg)

        row = {
            "trial": trial.number,
            "state": "running",
            "value": "",
            "objective_metric": metric,
            "experiment_name": run_cfg["experiment_name"],
            "experiment_dir": "",
            "error": "",
            **sampled,
        }
        trial_rows.append(row)
        _write_trials_csv(search_dir / "trials.csv", trial_rows, param_names)

        try:
            metadata = run_experiment(
                run_cfg,
                output_root=args.output_root,
                data_root_override=args.data_root,
                experiment_name=run_cfg["experiment_name"],
            )
            value = _metric_from_metadata(metadata, metric)
            row["state"] = "complete"
            row["value"] = value
            row["experiment_dir"] = metadata["experiment_dir"]
            trial.set_user_attr("experiment_dir", metadata["experiment_dir"])
            trial.set_user_attr("experiment_name", run_cfg["experiment_name"])
            trial.set_user_attr("metrics", metadata.get("metrics", {}))
            return value
        except Exception as exc:
            row["state"] = "failed"
            row["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            raise
        finally:
            _write_trials_csv(search_dir / "trials.csv", trial_rows, param_names)

    catch = () if args.stop_on_fail else (Exception,)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=catch)

    best_cfg = _best_trial_config(cfg, study, base_name)
    if best_cfg is not None:
        validate_experiment_config(best_cfg)
        write_yaml(search_dir / "best_config.yaml", best_cfg)

    best_trial = _best_trial_or_none(study)
    summary = {
        "study_name": study.study_name,
        "direction": direction,
        "objective": metric,
        "n_trials_requested": n_trials,
        "n_trials_finished": len(study.trials),
        "storage": str(storage),
        "search_dir": str(search_dir),
        "best_trial": best_trial.number if best_trial is not None else None,
        "best_value": best_trial.value if best_trial is not None else None,
        "best_params": dict(best_trial.params) if best_trial is not None else {},
    }
    write_json(search_dir / "study_summary.json", summary)
    write_json(search_dir / "best_params.json", summary["best_params"])

    print(f"Search finished: {search_dir}")
    if best_trial is not None:
        print(f"Best trial: {best_trial.number}")
        print(f"Best value ({metric}): {best_trial.value}")
        print(f"Best params: {best_trial.params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
