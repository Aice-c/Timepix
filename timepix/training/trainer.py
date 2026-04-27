"""Training and evaluation loops."""

from __future__ import annotations

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


def _progress(iterable, enabled: bool, desc: str | None):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit="batch", leave=False, dynamic_ncols=True)


def _set_postfix(iterator, loss: float) -> None:
    if hasattr(iterator, "set_postfix"):
        iterator.set_postfix(loss=f"{loss:.4f}")


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    task: str,
    progress_bar: bool = False,
    desc: str | None = None,
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

        optimizer.zero_grad()
        output = model(images, handcrafted)
        if task == "regression":
            loss = criterion(output.regression, labels.float())
            regression_list.append(output.regression.detach().cpu())
        else:
            loss = criterion(output.logits, labels.long())
            logits_list.append(output.logits.detach().cpu())
        labels_list.append(labels.detach().cpu())
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

        output = model(images, handcrafted)
        if task == "regression":
            loss = criterion(output.regression, labels.float())
            regression_list.append(output.regression.detach().cpu())
        else:
            loss = criterion(output.logits, labels.long())
            logits_list.append(output.logits.detach().cpu())
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
