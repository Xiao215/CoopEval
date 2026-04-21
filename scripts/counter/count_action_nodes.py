#!/usr/bin/env python3
"""Aggregate action-node counts per game/mechanism/model from CoopEval records."""

from __future__ import annotations

import argparse
from collections import Counter

from coopeval.script_utils.display_helper import extract_model_name
from coopeval.script_utils.result_loader import (
    DEFAULT_SKIP_GAMES,
    ExperimentInfo,
    iter_action_nodes,
    iter_experiments,
    load_json_lines,
)
from coopeval.utils.json_io import clean_path


def parse_args() -> argparse.Namespace:
    """Build the CLI for selecting experiment paths to scan."""
    parser = argparse.ArgumentParser(
        description=(
            "Walk one or more directories, find CoopEval experiments "
            "(config.json + records.jsonl), and report how many action nodes "
            "each model produced per (game, mechanism)."
        )
    )
    parser.add_argument(
        "--tournament_result_dirs",
        nargs="+",
        type=clean_path,
        required=True,
        help="Tournament result batch to scan.",
    )
    parser.add_argument(
        "--skip-games",
        nargs="*",
        default=DEFAULT_SKIP_GAMES,
        help=(
            "Games to skip when scanning runs "
            "(default: %(default)s; pass with no values to keep all games)."
        ),
    )
    return parser.parse_args()


def process_run(
    info: ExperimentInfo, counter: Counter[tuple[str, str, str]]
) -> None:
    """Count action nodes per model for a single experiment directory."""

    for payload in load_json_lines(info.path / "records.jsonl"):
        for node in iter_action_nodes(payload, require_trace_id=True):
            player = str(node.get("player", ""))
            model = extract_model_name(player)
            counter[(info.game, info.mechanism, model)] += 1


def main() -> None:
    """Aggregate action-node counts for every supplied experiment path."""
    args = parse_args()
    counter: Counter[tuple[str, str, str]] = Counter()
    runs_seen = 0

    skip_games = tuple(args.skip_games)

    for search_path in args.tournament_result_dirs:
        for info in iter_experiments(search_path, skip_games=skip_games):
            process_run(info, counter)
            runs_seen += 1

    if runs_seen == 0:
        print("No experiment directories found.")
        return

    print(f"Processed {runs_seen} experiment(s).\n")
    header = f"{'Game':<25} {'Mechanism':<18} {'Model':<50} {'ActionNodes':>12}"
    print(header)
    print("-" * len(header))
    for (game, mechanism, player), count in sorted(
        counter.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        print(f"{game:<25} {mechanism:<18} {player:<50} {count:>12}")

    totals_by_model: Counter[str] = Counter()
    for (_, _, model), count in counter.items():
        totals_by_model[model] += count

    if totals_by_model:
        print("\nTotal action nodes per model")
        print("-" * 32)
        for model, total in sorted(totals_by_model.items()):
            print(f"{model:<40} {total:>12}")


if __name__ == "__main__":
    main()
