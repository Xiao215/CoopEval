#!/usr/bin/env python3
"""
Convert tournament JSONL logs into a readable JSON structure.

Usage:
    python post_processing/jsonl_to_readable_json.py \
        --input outputs/2025/11/03/14:32/Repetition_PrisonersDilemma.jsonl \
        --match-index 6 \
        --output outputs/2025/11/03/14:32/deepseek_self_play.json

By default all matches contained in the JSONL file are exported.  Use
``--match-index`` (1-based) to select a single match.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


PROBABILITY_REGEX = re.compile(
    r'\{\s*"A0"\s*:\s*(\d+)[^0-9]+"A1"\s*:\s*(\d+)\s*\}', re.IGNORECASE
)


def iter_json_lines(path: Path) -> Iterable[list[list[dict[str, Any]]]]:
    """Yield parsed JSON objects from a newline-delimited log."""
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("Total output lines"):
                continue
            yield json.loads(line)


def extract_probabilities(response: str | None) -> dict[str, int] | None:
    """Attempt to extract the submitted probability distribution."""
    if not response:
        return None
    match = PROBABILITY_REGEX.search(response)
    if not match:
        return None
    return {"A0": int(match.group(1)), "A1": int(match.group(2))}


def build_match_payload(
    match: list[list[dict[str, Any]]],
    *,
    source_path: Path,
    match_index: int,
) -> dict[str, Any]:
    """Convert a single match (list of rounds) into a readable structure."""
    rounds: list[dict[str, Any]] = []
    for round_idx, round_moves in enumerate(match, start=1):
        players: list[dict[str, Any]] = []
        for move in round_moves:
            player_slot = f"{move['player_name']}#{move['uid']}"
            players.append(
                {
                    "uid": move["uid"],
                    "player_name": move["player_name"],
                    "player_slot": player_slot,
                    "action": move.get("action"),
                    "points": move.get("points"),
                    "probabilities": extract_probabilities(move.get("response")),
                    "response": move.get("response"),
                }
            )
        rounds.append({"round": round_idx, "players": players})

    unique_players = sorted(
        {move["player_name"] for round_moves in match for move in round_moves}
    )

    return {
        "description": "Tournament match parsed from JSONL log",
        "source_log": str(source_path),
        "match_index": match_index,
        "players": unique_players,
        "rounds": rounds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to JSONL log.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSON file. Defaults to stdout if omitted.",
    )
    parser.add_argument(
        "--match-index",
        type=int,
        help="1-based index of the match to export. Exports all matches when absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    matches = list(iter_json_lines(args.input))
    if args.match_index is not None:
        index = args.match_index
        if not (1 <= index <= len(matches)):
            raise ValueError(
                f"match-index {index} out of range (found {len(matches)} matches)."
            )
        payload = build_match_payload(
            matches[index - 1], source_path=args.input, match_index=index
        )
    else:
        payload = [
            build_match_payload(match, source_path=args.input, match_index=i + 1)
            for i, match in enumerate(matches)
        ]

    output_json = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json + "\n", encoding="utf-8")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
