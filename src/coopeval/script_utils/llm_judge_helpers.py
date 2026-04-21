"""Shared helpers for LLM judge reporting scripts."""

from __future__ import annotations

from typing import Any


def unique_labels(value: Any) -> list[str]:
    """Return deduped label list from a raw list or comma-separated string."""
    if isinstance(value, list):
        labels = [x for x in value if isinstance(x, str) and x.strip()]
    elif isinstance(value, str):
        labels = [x.strip() for x in value.split(",") if x.strip()]
    else:
        labels = []

    out: list[str] = []
    seen = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def classification_labels(row: dict[str, Any]) -> list[str]:
    """Return preferred normalized classification labels from a judge row."""
    return unique_labels(
        row.get("classification_labels_normalized")
        or row.get("classification_labels")
    )


def pct(count: int, denom: int) -> float:
    """Compute percentage share with a zero-denominator guard."""
    return 0.0 if denom == 0 else (100.0 * count / denom)
