"""Training and evaluation loops."""

from __future__ import annotations

import sys
import time

import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    tqdm = None


def _unpack_batch(batch):
    if len(batch) == 3:
        images, labels, handcrafted = batch
        return images, labels, handcrafted
    images, labels = batch
    return images, labels, None


class _SimpleProgress:
    def __init__(self, iterable, desc: str | None) -> None:
        self.iterable = iterable
        self.desc = desc or "progress"
        try:
            self.total = len(iterable)
        except TypeError:
            self.total = None
        self.count = 0
        self.postfix: dict[str, str] = {}
        self.start_time = time.monotonic()
        self.last_print = 0.0

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.count += 1
            self._print()
        self._print(force=True, done=True)
        print(file=sys.stderr)

    def set_postfix(self, **kwargs) -> None:
        self.postfix = {str(k): str(v) for k, v in kwargs.items()}

    def _print(self, force: bool = False, done: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_print < 1.0:
            return
        self.last_print = now
        if self.total:
            pct = self.count / self.total * 100
            progress = f"{self.count}/{self.total} ({pct:5.1f}%)"
        else:
            progress = str(self.count)
        elapsed = now - self.start_time
        postfix = " ".join(f"{k}={v}" for k, v in self.postfix.items())
        suffix = " done" if done else ""
        message = f"\r{self.desc}: {progress} elapsed={elapsed:.0f}s {postfix}{suffix}"
        print(message, end="", file=sys.stderr, flush=True)


def _progress(iterable, enabled: bool, desc: str | None):
    if not enabled or tqdm is None:
        return _SimpleProgress(iterable, desc) if enabled else iterable
    return tqdm(iterable, desc=desc, unit="batch", leave=False, dynamic_ncols=True)


def _set_postfix(iterator, loss: float) -> None:
    if hasattr(iterator, "set_postfix"):
        iterator.set_postfix(loss=f"{loss:.4f}")


def _scaler_enabled(grad_scaler) -> bool:
    return grad_scaler is not None and grad_scaler.is_enabled()


def _auxiliary_loss(output, labels, criterion, task: str, aux_loss_cfg: dict | None):
    if task != "classification" or not aux_loss_cfg or not bool(aux_loss_cfg.get("enabled", False)):
        return None
    aux_logits = getattr(output, "aux_logits", None) or {}
    if not aux_logits:
        return None

    total = None
    default_weight = float(aux_loss_cfg.get("weight", 0.0))
    for name, logits in aux_logits.items():
        weight = float(aux_loss_cfg.get(f"weight_{name}", default_weight))
        if weight <= 0.0:
            continue
        component = criterion(logits, labels.long()) * weight
        total = component if total is None else total + component
    return total


def _record_diagnostics(output, labels, diagnostics: dict[str, list[torch.Tensor]]) -> None:
    output_diagnostics = getattr(output, "diagnostics", None) or {}
    if not output_diagnostics:
        return

    batch_size = int(labels.shape[0])
    for key, value in output_diagnostics.items():
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        tensor = value.detach().float().cpu()
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        elif tensor.shape[0] == batch_size:
            if tensor.ndim > 1:
                tensor = tensor.reshape(batch_size, -1).mean(dim=1)
        else:
            tensor = tensor.reshape(-1).mean().reshape(1)
        diagnostics.setdefault(str(key), []).append(tensor)


def _summarize_diagnostics(
    diagnostics: dict[str, list[torch.Tensor]],
    labels: torch.Tensor | None,
) -> dict[str, dict]:
    if not diagnostics:
        return {}
    labels_np = labels.numpy().astype(int) if labels is not None else None
    summary: dict[str, dict] = {}
    for key, chunks in diagnostics.items():
        if not chunks:
            continue
        values = torch.cat(chunks).numpy()
        item = {"mean": float(values.mean()) if values.size else 0.0}
        if labels_np is not None and values.shape[0] == labels_np.shape[0]:
            by_class = {}
            for cls in sorted(set(int(v) for v in labels_np.tolist())):
                mask = labels_np == cls
                by_class[str(cls)] = float(values[mask].mean()) if mask.any() else 0.0
            item["by_class"] = by_class
        summary[key] = item
    return summary


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    task: str,
    progress_bar: bool = False,
    desc: str | None = None,
    autocast_factory=None,
    grad_scaler=None,
    aux_loss_cfg: dict | None = None,
):
    model.train()
    total_loss = 0.0
    total_count = 0
    logits_list = []
    labels_list = []
    regression_list = []
    diagnostics_list: dict[str, list[torch.Tensor]] = {}
    use_scaler = _scaler_enabled(grad_scaler)

    iterator = _progress(loader, progress_bar, desc)
    for batch in iterator:
        images, labels, handcrafted = _unpack_batch(batch)
        images = images.to(device)
        labels = labels.to(device)
        handcrafted = handcrafted.to(device) if handcrafted is not None else None

        optimizer.zero_grad(set_to_none=True)
        if autocast_factory is None:
            output = model(images, handcrafted)
            if task == "regression":
                loss = criterion(output.regression, labels.float())
            else:
                loss = criterion(output.logits, labels.long())
                aux_loss = _auxiliary_loss(output, labels, criterion, task, aux_loss_cfg)
                if aux_loss is not None:
                    loss = loss + aux_loss
        else:
            with autocast_factory():
                output = model(images, handcrafted)
                if task == "regression":
                    loss = criterion(output.regression, labels.float())
                else:
                    loss = criterion(output.logits, labels.long())
                    aux_loss = _auxiliary_loss(output, labels, criterion, task, aux_loss_cfg)
                    if aux_loss is not None:
                        loss = loss + aux_loss
        if task == "regression":
            regression_list.append(output.regression.detach().float().cpu())
        else:
            logits_list.append(output.logits.detach().float().cpu())
        labels_list.append(labels.detach().cpu())
        _record_diagnostics(output, labels, diagnostics_list)
        if use_scaler:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        _set_postfix(iterator, total_loss / max(total_count, 1))

    labels_tensor = torch.cat(labels_list) if labels_list else None
    payload = {
        "loss": total_loss / max(total_count, 1),
        "labels": labels_tensor.numpy() if labels_tensor is not None else None,
    }
    if task == "regression":
        payload["regression"] = torch.cat(regression_list).numpy() if regression_list else None
    else:
        payload["logits"] = torch.cat(logits_list).numpy() if logits_list else None
    diagnostics = _summarize_diagnostics(diagnostics_list, labels_tensor)
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return payload


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    task: str,
    progress_bar: bool = False,
    desc: str | None = None,
    autocast_factory=None,
    aux_loss_cfg: dict | None = None,
):
    model.eval()
    total_loss = 0.0
    total_count = 0
    logits_list = []
    labels_list = []
    regression_list = []
    diagnostics_list: dict[str, list[torch.Tensor]] = {}

    iterator = _progress(loader, progress_bar, desc)
    for batch in iterator:
        images, labels, handcrafted = _unpack_batch(batch)
        images = images.to(device)
        labels = labels.to(device)
        handcrafted = handcrafted.to(device) if handcrafted is not None else None

        if autocast_factory is None:
            output = model(images, handcrafted)
            if task == "regression":
                loss = criterion(output.regression, labels.float())
            else:
                loss = criterion(output.logits, labels.long())
                aux_loss = _auxiliary_loss(output, labels, criterion, task, aux_loss_cfg)
                if aux_loss is not None:
                    loss = loss + aux_loss
        else:
            with autocast_factory():
                output = model(images, handcrafted)
                if task == "regression":
                    loss = criterion(output.regression, labels.float())
                else:
                    loss = criterion(output.logits, labels.long())
                    aux_loss = _auxiliary_loss(output, labels, criterion, task, aux_loss_cfg)
                    if aux_loss is not None:
                        loss = loss + aux_loss
        if task == "regression":
            regression_list.append(output.regression.detach().float().cpu())
        else:
            logits_list.append(output.logits.detach().float().cpu())
        labels_list.append(labels.detach().cpu())
        _record_diagnostics(output, labels, diagnostics_list)

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        _set_postfix(iterator, total_loss / max(total_count, 1))

    labels_tensor = torch.cat(labels_list) if labels_list else None
    payload = {
        "loss": total_loss / max(total_count, 1),
        "labels": labels_tensor.numpy() if labels_tensor is not None else None,
    }
    if task == "regression":
        payload["regression"] = torch.cat(regression_list).numpy() if regression_list else None
    else:
        payload["logits"] = torch.cat(logits_list).numpy() if logits_list else None
    diagnostics = _summarize_diagnostics(diagnostics_list, labels_tensor)
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return payload
