"""Stratified split helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path


def stratified_split(records, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    by_label: dict[int, list[int]] = {}
    for idx, record in enumerate(records):
        by_label.setdefault(record.label, []).append(idx)

    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for label in sorted(by_label):
        idxs = list(by_label[label])
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            k_train, k_val = 1, 0
        elif n == 2:
            k_train, k_val = 1, 1
        else:
            k_train = max(1, int(n * train_ratio))
            k_val = max(1, int(n * val_ratio))
            k_train = min(k_train, n - 2)
            k_val = min(k_val, n - k_train - 1)
        train_idx.extend(idxs[:k_train])
        val_idx.extend(idxs[k_train : k_train + k_val])
        test_idx.extend(idxs[k_train + k_val :])

    return train_idx, val_idx, test_idx


def save_split_manifest(path: str | Path, records, train_idx, val_idx, test_idx) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": [records[i].key for i in train_idx],
        "val": [records[i].key for i in val_idx],
        "test": [records[i].key for i in test_idx],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_split_manifest(path: str | Path, records):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_key = {record.key: idx for idx, record in enumerate(records)}

    def resolve(keys):
        missing = [key for key in keys if key not in by_key]
        if missing:
            raise ValueError(f"Split manifest does not match dataset. Missing key: {missing[0]}")
        return [by_key[key] for key in keys]

    return resolve(payload["train"]), resolve(payload["val"]), resolve(payload["test"])

