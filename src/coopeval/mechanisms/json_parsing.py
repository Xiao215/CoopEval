"""Shared parsers for mechanism-specific LLM JSON responses."""

from __future__ import annotations

from coopeval.utils.json_io import extract_json_object_from_end


def parse_bool_votes(
    response: str,
    *,
    prefix: str,
    count: int,
) -> dict[int, bool]:
    """Parse keyed boolean votes like ``{"M1": true}`` or ``{"C1": false}``."""
    json_obj = extract_json_object_from_end(response)
    votes: dict[int, bool] = {}
    expected_keys = {f"{prefix}{i}" for i in range(1, count + 1)}
    got_keys = set(json_obj.keys())

    missing = expected_keys - got_keys
    extra = got_keys - expected_keys
    if missing:
        raise ValueError(f"Missing vote for {sorted(missing)[0]}")
    if extra:
        raise ValueError(f"Unexpected vote keys: {sorted(extra)}")

    for i in range(1, count + 1):
        key = f"{prefix}{i}"
        value = json_obj[key]
        if not isinstance(value, bool):
            raise ValueError(
                f"Vote for {key} must be boolean, got {value!r}"
            )
        votes[i] = value

    return votes


def parse_action_value_map(
    response: str,
    *,
    num_actions: int,
) -> dict[int, int]:
    """Parse an integer value for every action token ``A0..A{n}``."""
    json_obj = extract_json_object_from_end(response)
    values: dict[int, int] = {}
    expected_keys = {f"A{i}" for i in range(num_actions)}
    got_keys = set(json_obj.keys())

    missing = expected_keys - got_keys
    extra = got_keys - expected_keys
    if missing:
        raise ValueError(f"Action key mismatch. Missing: {sorted(missing)}")
    if extra:
        raise ValueError(f"Action key mismatch. Extra: {sorted(extra)}")

    for idx in range(num_actions):
        key = f"A{idx}"
        if isinstance(json_obj[key], bool):
            raise ValueError(
                f"Value for {key} must be an integer, got {json_obj[key]!r}"
            )
        try:
            values[idx] = int(json_obj[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Value for {key} must be an integer, got {json_obj[key]!r}"
            ) from exc

    return values

