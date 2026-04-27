"""Training and evaluation loops."""

from __future__ import annotations

from contextlib import nullcontext
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


def _autocast_context(autocast_factory):
    return autocast_factory() if autocast_factory is not None else nullcontext()


def _scaler_enabled(grad_scaler) -> bool:
    return grad_scaler is not None and grad_scaler.is_enabled()


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
):
    model.train()
    total_loss = 0.0
    total_count = 0
    logits_list = []
    labels_list = []
    regression_list = []

    iterator = _progress(loader, progress_bar, desc)
    for batch in iterator:
        images, labels, handcrafted = _unpack_batch(batch)
        images = images.to(device)
        labels = labels.to(device)
        handcrafted = handcrafted.to(device) if handcrafted is not None else None

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(autocast_factory):
            output = model(images, handcrafted)
            if task == "regression":
                loss = criterion(output.regression, labels.float())
            else:
                loss = criterion(output.logits, labels.long())
        if task == "regression":
            regression_list.append(output.regression.detach().float().cpu())
        else:
            logits_list.append(output.logits.detach().float().cpu())
        labels_list.append(labels.detach().cpu())
        if _scaler_enabled(grad_scaler):
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

    payload = {
        "loss": total_loss / max(total_count, 1),
        "labels": torch.cat(labels_list).numpy() if labels_list else None,
    }
    if task == "regression":
        payload["regression"] = torch.cat(regression_list).numpy() if regression_list else None
    else:
        payload["logits"] = torch.cat(logits_list).numpy() if logits_list else None
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
):
    model.eval()
    total_loss = 0.0
    total_count = 0
    logits_list = []
    labels_list = []
    regression_list = []

    iterator = _progress(loader, progress_bar, desc)
    for batch in iterator:
        images, labels, handcrafted = _unpack_batch(batch)
        images = images.to(device)
        labels = labels.to(device)
        handcrafted = handcrafted.to(device) if handcrafted is not None else None

        with _autocast_context(autocast_factory):
            output = model(images, handcrafted)
            if task == "regression":
                loss = criterion(output.regression, labels.float())
            else:
                loss = criterion(output.logits, labels.long())
        if task == "regression":
            regression_list.append(output.regression.detach().float().cpu())
        else:
            logits_list.append(output.logits.detach().float().cpu())
        labels_list.append(labels.detach().cpu())

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        _set_postfix(iterator, total_loss / max(total_count, 1))

    payload = {
        "loss": total_loss / max(total_count, 1),
        "labels": torch.cat(labels_list).numpy() if labels_list else None,
    }
    if task == "regression":
        payload["regression"] = torch.cat(regression_list).numpy() if regression_list else None
    else:
        payload["logits"] = torch.cat(logits_list).numpy() if logits_list else None
    return payload
