"""Lightweight helpers for small batches of concurrent tasks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")
R = TypeVar("R")

_DEFAULT_MAX_WORKERS: int | None = None


def set_default_max_workers(max_workers: int | None) -> None:
    """Configure how many worker threads ``run_tasks`` will use by default."""

    global _DEFAULT_MAX_WORKERS
    _DEFAULT_MAX_WORKERS = _sanitize_max_workers(max_workers)


def run_tasks(
    items: Iterable[T],
    func: Callable[[T], R],
    *,
    max_workers: int | None = None,
) -> list[R]:
    """Apply ``func`` to each item, fanning out with a thread pool if useful."""

    item_list = list(items)
    if not item_list:
        return []

    worker_cap = _pick_worker_count(len(item_list), max_workers)
    if worker_cap <= 1 or len(item_list) <= 1:
        return [func(item) for item in item_list]

    with ThreadPoolExecutor(max_workers=worker_cap) as executor:
        futures = [executor.submit(func, item) for item in item_list]
        return [future.result() for future in futures]


def _pick_worker_count(n_items: int, override: int | None) -> int:
    if n_items <= 1:
        return 1

    if override is None:
        override = _DEFAULT_MAX_WORKERS

    # Default to one worker per item if nothing configured.
    if override is None:
        override = n_items

    return max(1, min(_sanitize_max_workers(override) or n_items, n_items))


def _sanitize_max_workers(value: int | None) -> int | None:
    if value is None:
        return None
    return value if value > 0 else 1

