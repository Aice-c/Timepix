"""Run one Timepix experiment from a config dictionary."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from timepix.config import PROJECT_ROOT, resolve_project_path
from timepix.data import build_dataloaders
from timepix.losses import build_loss
from timepix.models import build_model
from timepix.training.logger import CsvLogger, write_json, write_yaml
from timepix.training.metrics import classification_metrics, regression_metrics
from timepix.training.trainer import evaluate, train_one_epoch
from timepix.utils.paths import make_experiment_dir, slugify
from timepix.utils.seed import set_seed


def _count_parameters(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


def _clone_state_dict(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def _sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_info() -> dict[str, Any]:
    status = _run_git(["status", "--porcelain"])
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["branch", "--show-current"]),
        "dirty": bool(status),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment_info(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def _mixed_precision_dtype(name: str) -> tuple[str, torch.dtype]:
    normalized = str(name).strip().lower().replace("torch.", "")
    aliases = {
        "fp16": "float16",
        "float16": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
    }
    dtype_name = aliases.get(normalized)
    if dtype_name is None:
        raise ValueError("training.mixed_precision_dtype must be one of: float16, fp16, bfloat16, bf16")
    return dtype_name, torch.float16 if dtype_name == "float16" else torch.bfloat16


def _cuda_autocast_factory(dtype: torch.dtype):
    def factory():
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            try:
                return torch.amp.autocast("cuda", dtype=dtype)
            except TypeError:
                pass
        return torch.cuda.amp.autocast(dtype=dtype)

    return factory


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _mixed_precision_setup(training_cfg: dict, device: torch.device):
    requested = bool(training_cfg.get("mixed_precision", False))
    dtype_name, dtype = _mixed_precision_dtype(training_cfg.get("mixed_precision_dtype", "float16"))
    info = {
        "requested": requested,
        "enabled": False,
        "dtype": dtype_name,
        "grad_scaler_enabled": False,
        "reason": "",
    }
    if not requested:
        return None, None, info
    if device.type != "cuda":
        info["reason"] = "disabled because CUDA is not available"
        print(f"[AMP] mixed_precision requested but {info['reason']}.")
        return None, None, info
    if dtype == torch.bfloat16 and hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        raise ValueError("training.mixed_precision_dtype=bfloat16 was requested, but this CUDA device does not support BF16")

    scaler = _make_grad_scaler(enabled=dtype == torch.float16)
    info["enabled"] = True
    info["grad_scaler_enabled"] = bool(scaler.is_enabled())
    print(f"[AMP] enabled with dtype={dtype_name}, grad_scaler={info['grad_scaler_enabled']}")
    return _cuda_autocast_factory(dtype), scaler, info


def load_config_from_checkpoint(path: str | Path) -> dict:
    checkpoint_path = resolve_project_path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(f"Checkpoint does not contain a config: {checkpoint_path}")
    return copy.deepcopy(cfg)


def _checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint_path = resolve_project_path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, Mapping) and "model_state" in checkpoint:
        state = checkpoint["model_state"]
    else:
        state = checkpoint
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint does not contain a state dict: {checkpoint_path}")
    return dict(state)


def _matches_filter(value: Any, expected: Any) -> bool:
    if expected is None:
        return True
    if isinstance(expected, list):
        return list(value or []) == expected
    return value == expected


def _find_checkpoint_from_metadata(search_cfg: Mapping[str, Any], *, seed: int, output_root: Path) -> Path:
    groups = search_cfg.get("groups", search_cfg.get("group"))
    if groups is None:
        raise ValueError("checkpoint search requires group or groups")
    if isinstance(groups, str):
        groups = [groups]

    checkpoint_name = str(search_cfg.get("checkpoint_name", "best_model.pth"))
    candidates: list[tuple[float, Path, Path]] = []
    for group in groups:
        group_root = output_root / str(group)
        for metadata_path in sorted(group_root.glob("*/metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            training = metadata.get("training", {})
            if int(training.get("seed", -1)) != int(seed):
                continue
            if "dataset_name" in search_cfg and not _matches_filter(metadata.get("dataset", {}).get("name"), search_cfg.get("dataset_name")):
                continue
            if "modalities" in search_cfg and not _matches_filter(metadata.get("dataset", {}).get("modalities"), search_cfg.get("modalities")):
                continue
            if "model" in search_cfg and not _matches_filter(metadata.get("model", {}).get("name"), search_cfg.get("model")):
                continue
            data_filters = search_cfg.get("data", {})
            if isinstance(data_filters, Mapping):
                data = metadata.get("data", {})
                data_info = metadata.get("data_info", {})
                if any(
                    not _matches_filter(data_info.get(key, data.get(key)), expected)
                    for key, expected in data_filters.items()
                ):
                    continue
            checkpoint_path = metadata_path.parent / checkpoint_name
            if checkpoint_path.exists():
                val = metadata.get("metrics", {}).get("validation", {})
                score = float(val.get("accuracy", 0.0) or 0.0)
                candidates.append((score, metadata_path, checkpoint_path))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint matched search config for seed={seed}: {dict(search_cfg)}")
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return candidates[0][2]


def _select_checkpoint_path(source: Any, *, seed: int, label: str, output_root: Path) -> Path:
    if isinstance(source, str):
        return resolve_project_path(source)
    if isinstance(source, Mapping):
        if "path" in source:
            return resolve_project_path(source["path"])
        if "paths" in source:
            paths = source["paths"]
            if not isinstance(paths, Mapping):
                raise ValueError(f"model.expert_gate.{label}.paths must be a mapping")
            value = paths.get(str(seed), paths.get(seed))
            if value is None:
                raise ValueError(f"model.expert_gate.{label}.paths has no entry for seed={seed}")
            return resolve_project_path(value)
        return _find_checkpoint_from_metadata(source, seed=seed, output_root=output_root)
    raise ValueError(f"model.expert_gate.{label} must be a path, mapping, or metadata search config")


def _initialize_model_from_config(model, cfg: dict, *, seed: int, output_root: Path) -> dict[str, Any]:
    model_cfg = cfg.get("model", {})
    expert_gate_cfg = model_cfg.get("expert_gate", {})
    info: dict[str, Any] = {}
    if hasattr(model, "load_expert_states") and expert_gate_cfg:
        primary_source = expert_gate_cfg.get("primary_checkpoint") or expert_gate_cfg.get("primary_search")
        candidate_source = expert_gate_cfg.get("candidate_checkpoint") or expert_gate_cfg.get("candidate_search")
        if primary_source is None or candidate_source is None:
            raise ValueError("warm_started_expert_gate requires primary and candidate checkpoint sources")
        primary_path = _select_checkpoint_path(primary_source, seed=seed, label="primary", output_root=output_root)
        candidate_path = _select_checkpoint_path(candidate_source, seed=seed, label="candidate", output_root=output_root)
        strict = bool(expert_gate_cfg.get("strict_checkpoint_load", True))
        load_info = model.load_expert_states(
            _checkpoint_state_dict(primary_path),
            _checkpoint_state_dict(candidate_path),
            strict=strict,
        )
        freeze_experts = bool(expert_gate_cfg.get("freeze_experts", False))
        if hasattr(model, "set_experts_trainable"):
            model.set_experts_trainable(not freeze_experts)
        info.update(
            {
                "type": "warm_started_expert_gate",
                "primary_checkpoint": str(primary_path),
                "candidate_checkpoint": str(candidate_path),
                "strict_checkpoint_load": strict,
                "freeze_experts": freeze_experts,
                "load_info": load_info,
            }
        )
        print(f"[Init] Loaded primary expert: {primary_path}")
        print(f"[Init] Loaded candidate expert: {candidate_path}")
        print(f"[Init] freeze_experts={freeze_experts}")
    return info


def _metrics_from_payload(payload: dict, task: str, angle_values: list[float], max_angle: float) -> dict:
    if task == "regression":
        return regression_metrics(payload["regression"], payload["labels"], max_angle)
    return classification_metrics(payload["logits"], payload["labels"].astype(int), angle_values)


def _primary_score(metrics: dict, task: str, primary_metric: str) -> float:
    lower_is_better = any(token in primary_metric for token in ("mae", "error", "rmse", "loss"))
    if primary_metric in metrics:
        value = float(metrics[primary_metric])
        if lower_is_better:
            value = -value
    elif task == "classification" and primary_metric == "val_accuracy":
        value = float(metrics["accuracy"])
    elif task == "regression":
        value = -float(metrics.get("mae", 0.0))
    else:
        value = float(metrics.get("accuracy", 0.0))
    return value


def _save_predictions(path: Path, payload: dict, task: str, angle_values: list[float], max_angle: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if task == "regression":
            writer = csv.DictWriter(f, fieldnames=["row", "true_angle", "pred_angle", "abs_error"])
            writer.writeheader()
            for idx, (true, pred) in enumerate(zip(payload["labels"], payload["regression"])):
                true_angle = float(true) * max_angle
                pred_angle = float(pred) * max_angle
                writer.writerow(
                    {
                        "row": idx,
                        "true_angle": true_angle,
                        "pred_angle": pred_angle,
                        "abs_error": abs(pred_angle - true_angle),
                    }
                )
        else:
            logits = payload["logits"]
            labels = payload["labels"].astype(int)
            shifted = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            angles = np.asarray(angle_values, dtype=float)
            preds = probs.argmax(axis=1)
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "row",
                    "true_label",
                    "pred_label",
                    "true_angle",
                    "pred_angle_argmax",
                    "pred_angle_weighted",
                    "abs_error_argmax",
                    "abs_error_weighted",
                ],
            )
            writer.writeheader()
            for idx, (true, pred, prob) in enumerate(zip(labels, preds, probs)):
                true_angle = float(angles[true])
                pred_angle_argmax = float(angles[pred])
                pred_angle_weighted = float(prob @ angles)
                writer.writerow(
                    {
                        "row": idx,
                        "true_label": int(true),
                        "pred_label": int(pred),
                        "true_angle": true_angle,
                        "pred_angle_argmax": pred_angle_argmax,
                        "pred_angle_weighted": pred_angle_weighted,
                        "abs_error_argmax": abs(pred_angle_argmax - true_angle),
                        "abs_error_weighted": abs(pred_angle_weighted - true_angle),
                    }
                )


def _save_last_checkpoint(
    path: Path,
    *,
    epoch: int,
    model,
    optimizer,
    scheduler,
    best_score: float,
    best_epoch: int,
    best_model_state: dict[str, torch.Tensor] | None,
    best_val_metrics: dict[str, Any],
    patience_counter: int,
    grad_scaler,
    experiment_dir: Path,
    cfg: dict,
    data_root_override: str | None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_model_state": best_model_state,
        "best_val_metrics": best_val_metrics,
        "patience_counter": patience_counter,
        "scaler_state": grad_scaler.state_dict() if grad_scaler is not None and grad_scaler.is_enabled() else None,
        "experiment_dir": str(experiment_dir),
        "config": cfg,
        "data_root_override": data_root_override,
    }
    _atomic_torch_save(checkpoint, path)


def run_experiment(
    cfg: dict,
    output_root: str | Path | None = None,
    data_root_override: str | None = None,
    experiment_name: str | None = None,
) -> dict:
    run_started_at = time.perf_counter()
    training_cfg = cfg.get("training", {})
    task_cfg = cfg.get("task", {})
    model_cfg = cfg.get("model", {})
    task = task_cfg.get("type", "classification")
    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)

    name = experiment_name or cfg.get("experiment_name") or "timepix_experiment"
    experiment_group = cfg.get("experiment_group") or "default"
    experiment_group = slugify(str(experiment_group))
    output_root = output_root or cfg.get("output", {}).get("root", "outputs/experiments")
    output_root = resolve_project_path(output_root)
    resume_from = training_cfg.get("resume_from")
    resume_checkpoint = None
    if resume_from:
        resume_path = resolve_project_path(resume_from)
        resume_checkpoint = torch.load(resume_path, map_location="cpu")
        exp_dir = Path(resume_checkpoint.get("experiment_dir", resume_path.parent)).resolve()
        exp_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_data_root = resume_checkpoint.get("data_root_override")
        if data_root_override is None and checkpoint_data_root:
            data_root_override = str(checkpoint_data_root)
            print(f"[Resume] Using checkpoint data root: {data_root_override}")
        print(f"[Resume] Loading checkpoint: {resume_path}")
        print(f"[Resume] Experiment directory: {exp_dir}")
    else:
        exp_dir = make_experiment_dir(output_root / experiment_group, name)
        write_yaml(exp_dir / "config.yaml", cfg)
    best_model_path = exp_dir / "best_model.pth"

    loaders, data_info = build_dataloaders(cfg, data_root_override=data_root_override)
    label_map = data_info["label_map"]
    num_classes = data_info["num_classes"]
    angle_values = [float(label_map[i]) for i in range(num_classes)]
    max_angle = float(task_cfg.get("max_angle", 90.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_factory, grad_scaler, mixed_precision_info = _mixed_precision_setup(training_cfg, device)
    model = build_model(
        cfg,
        input_channels=int(data_info.get("input_channels", len(data_info["modalities"]))),
        num_classes=num_classes,
        task=task,
        handcrafted_dim=int(data_info["handcrafted_dim"]),
    ).to(device)
    model_initialization_info = _initialize_model_from_config(model, cfg, seed=seed, output_root=output_root)
    param_count = _count_parameters(model)

    criterion = build_loss(cfg, num_classes, label_map).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    scheduler = None
    epochs = int(training_cfg.get("epochs", 20))
    if training_cfg.get("scheduler", "none") == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(training_cfg.get("eta_min", 1e-6)))

    log_fields = [
        "epoch",
        "lr",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "train_mae_argmax",
        "val_mae_argmax",
        "train_p90_error",
        "val_p90_error",
        "train_macro_f1",
        "val_macro_f1",
        "epoch_seconds",
    ]
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    best_val_metrics = {}
    patience = int(training_cfg.get("early_stopping_patience", 0))
    patience_counter = 0
    primary_metric = task_cfg.get("primary_metric", "val_accuracy")
    start_epoch = 1
    show_progress = bool(training_cfg.get("progress_bar", True))
    save_last_checkpoint = bool(training_cfg.get("save_last_checkpoint", True))
    aux_loss_cfg = model_cfg.get("aux_loss", {})
    best_val_diagnostics = {}

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if scheduler is not None and resume_checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_score = float(resume_checkpoint.get("best_score", best_score))
        best_epoch = int(resume_checkpoint.get("best_epoch", best_epoch))
        best_val_metrics = dict(resume_checkpoint.get("best_val_metrics", {}))
        patience_counter = int(resume_checkpoint.get("patience_counter", 0))
        if grad_scaler is not None and resume_checkpoint.get("scaler_state") is not None:
            grad_scaler.load_state_dict(resume_checkpoint["scaler_state"])
        if resume_checkpoint.get("best_model_state") is not None:
            best_state = resume_checkpoint["best_model_state"]
        elif best_model_path.exists():
            best_state = torch.load(best_model_path, map_location="cpu")
        else:
            best_state = _clone_state_dict(model)
        print(f"[Resume] Continue from epoch {start_epoch}/{epochs}")

    log = CsvLogger(
        exp_dir / "training_log.csv",
        log_fields,
        append=resume_checkpoint is not None,
        resume_from_epoch=start_epoch if resume_checkpoint is not None else None,
    )

    if start_epoch > epochs:
        print(f"[Resume] Checkpoint already reached epoch {start_epoch - 1}; skipping training loop.")

    stopped_epoch = start_epoch - 1
    early_stopped = False
    fit_started_at = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        epoch_started_at = time.perf_counter()
        stopped_epoch = epoch
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{epochs} | lr={lr:.6g}")
        train_payload = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            task,
            progress_bar=show_progress,
            desc=f"train {epoch}/{epochs}",
            autocast_factory=autocast_factory,
            grad_scaler=grad_scaler,
            aux_loss_cfg=aux_loss_cfg,
        )
        val_payload = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            task,
            progress_bar=show_progress,
            desc=f"val   {epoch}/{epochs}",
            autocast_factory=autocast_factory,
            aux_loss_cfg=aux_loss_cfg,
        )
        if scheduler is not None:
            scheduler.step()

        train_metrics = _metrics_from_payload(train_payload, task, angle_values, max_angle)
        val_metrics = _metrics_from_payload(val_payload, task, angle_values, max_angle)
        score = _primary_score(val_metrics, task, primary_metric)
        epoch_seconds = time.perf_counter() - epoch_started_at

        log.write(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_payload["loss"],
                "val_loss": val_payload["loss"],
                "train_accuracy": train_metrics.get("accuracy", ""),
                "val_accuracy": val_metrics.get("accuracy", ""),
                "train_mae_argmax": train_metrics.get("mae_argmax", train_metrics.get("mae", "")),
                "val_mae_argmax": val_metrics.get("mae_argmax", val_metrics.get("mae", "")),
                "train_p90_error": train_metrics.get("p90_error", ""),
                "val_p90_error": val_metrics.get("p90_error", ""),
                "train_macro_f1": train_metrics.get("macro_f1", ""),
                "val_macro_f1": val_metrics.get("macro_f1", ""),
                "epoch_seconds": epoch_seconds,
            }
        )

        is_better = score > best_score
        if is_better:
            best_score = score
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_val_diagnostics = val_payload.get("diagnostics", {})
            best_state = _clone_state_dict(model)
            patience_counter = 0
            _atomic_torch_save(best_state, best_model_path)
        else:
            patience_counter += 1

        print(
            "Epoch summary | "
            f"train_loss={train_payload['loss']:.5f} "
            f"val_loss={val_payload['loss']:.5f} "
            f"val_acc={val_metrics.get('accuracy', 0):.4f} "
            f"val_mae={val_metrics.get('mae_argmax', val_metrics.get('mae', 0)):.3f} "
            f"val_p90={val_metrics.get('p90_error', 0):.3f} "
            f"time={epoch_seconds:.1f}s"
            + (" | best" if is_better else "")
        )

        if save_last_checkpoint:
            _save_last_checkpoint(
                exp_dir / "last_checkpoint.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_score=best_score,
                best_epoch=best_epoch,
                best_model_state=best_state,
                best_val_metrics=best_val_metrics,
                patience_counter=patience_counter,
                grad_scaler=grad_scaler,
                experiment_dir=exp_dir,
                cfg=cfg,
                data_root_override=str(data_info["data_root"]) if data_root_override is not None else None,
            )

        if patience > 0 and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            early_stopped = True
            break

    fit_seconds = time.perf_counter() - fit_started_at
    if best_state is not None:
        _atomic_torch_save(best_state, best_model_path)
        model.load_state_dict(best_state)

    test_started_at = time.perf_counter()
    test_payload = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        task,
        autocast_factory=autocast_factory,
        aux_loss_cfg=aux_loss_cfg,
    )
    test_seconds = time.perf_counter() - test_started_at
    test_metrics = _metrics_from_payload(test_payload, task, angle_values, max_angle)
    _save_predictions(exp_dir / "predictions.csv", test_payload, task, angle_values, max_angle)
    total_seconds = time.perf_counter() - run_started_at

    metrics = {
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "max_epochs": epochs,
        "early_stopped": early_stopped,
        "best_score": best_score,
        "fit_seconds": fit_seconds,
        "test_seconds": test_seconds,
        "total_seconds": total_seconds,
        "validation": best_val_metrics,
        "test": test_metrics,
    }
    if best_val_diagnostics:
        metrics["validation_diagnostics"] = best_val_diagnostics
    if test_payload.get("diagnostics"):
        metrics["test_diagnostics"] = test_payload["diagnostics"]
    write_json(exp_dir / "metrics.json", metrics)
    if "confusion_matrix" in test_metrics:
        np.savetxt(exp_dir / "confusion_matrix.csv", np.asarray(test_metrics["confusion_matrix"], dtype=int), fmt="%d", delimiter=",")

    metadata = {
        "experiment_name": name,
        "experiment_group": experiment_group,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_dir": str(exp_dir),
        "config_path": cfg.get("_config_path"),
        "dataset": cfg.get("dataset", {}),
        "data": cfg.get("data", {}),
        "data_info": data_info,
        "task": task,
        "model": cfg.get("model", {}),
        "model_initialization": model_initialization_info,
        "loss": cfg.get("loss", {}),
        "training": training_cfg,
        "mixed_precision": mixed_precision_info,
        "timing": {
            "fit_seconds": fit_seconds,
            "test_seconds": test_seconds,
            "total_seconds": total_seconds,
        },
        "param_count": param_count,
        "primary_metric": primary_metric,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "device": str(device),
        "environment": _environment_info(device),
        "git": _git_info(),
        "command": " ".join(sys.argv),
        "split_manifest_hash": _sha256_file(data_info.get("split_path", "")),
    }
    write_json(exp_dir / "metadata.json", metadata)
    return metadata
