"""JSON and JSONL file helpers shared by package code and scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator


def extract_json_object_from_end(response: str) -> dict[str, Any]:
    """Extract the last valid JSON object embedded in an LLM response.

    This is intentionally stricter than a brace regex: it uses JSONDecoder so
    nested objects are handled correctly, while surrounding prose/code fences are
    ignored.
    """
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []

    start_idx = 0
    while start_idx < len(response):
        start_idx = response.find("{", start_idx)
        if start_idx == -1:
            break
        try:
            parsed, end_offset = decoder.raw_decode(response[start_idx:])
        except json.JSONDecodeError:
            start_idx += 1
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)
        start_idx += max(end_offset, 1)

    if not candidates:
        raise ValueError(f"No JSON object found in the response {response!r}")

    return candidates[-1]


def clean_path(value: str) -> Path:
    """Clean up the path directly from command line args."""
    p = Path(value).expanduser().resolve()
    return p


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Write a JSON file with a trailing newline."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False)
        handle.write("\n")


def iter_jsonl(path: Path, *, skip_invalid: bool = False) -> Iterator[Any]:
    """Yield parsed records from a JSONL file."""
    for _line_no, record in iter_jsonl_with_line_numbers(
        path, skip_invalid=skip_invalid
    ):
        yield record


def iter_jsonl_with_line_numbers(
    path: Path,
    *,
    skip_invalid: bool = False,
    on_invalid: Callable[[int, json.JSONDecodeError], None] | None = None,
) -> Iterator[tuple[int, Any]]:
    """Yield ``(line_number, record)`` pairs from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                if on_invalid is not None:
                    on_invalid(line_no, exc)
                if skip_invalid:
                    continue
                raise ValueError(
                    f"Failed to parse {path}:{line_no}: {exc}"
                ) from exc


def write_jsonl_record(handle: Any, record: Any) -> None:
    """Write one JSONL record to an open text file."""
    json.dump(record, handle, ensure_ascii=False)
    handle.write("\n")

