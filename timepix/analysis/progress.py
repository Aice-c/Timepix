"""Progress-bar helpers for long-running analysis loops."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


def iter_progress(iterable: Iterable[T], *, total: int | None = None, desc: str | None = None, unit: str = "it") -> Iterator[T]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield from iterable
        return

    yield from tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)
