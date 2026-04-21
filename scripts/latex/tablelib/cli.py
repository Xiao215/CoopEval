#!/usr/bin/env python3
"""Shared CLI helpers for LaTeX table scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from coopeval.script_utils.result_loader import (
    DEFAULT_SKIP_GAMES,
    should_skip_game_name,
)
from coopeval.utils.json_io import clean_path


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach shared tournament-result arguments to a parser."""
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
            "Games to skip before aggregation "
            "(default: %(default)s; pass with no values to include all)."
        ),
    )


def resolve_paths(paths: Iterable[Path]) -> list[Path]:
    """Expand and resolve a sequence of filesystem paths."""
    return [p.expanduser().resolve() for p in paths]


def filter_skipped_games(
    games: Iterable[str], skip_games: Iterable[str] | None
) -> list[str]:
    """Return games after applying the shared skip-game list."""
    return [
        game for game in games if not should_skip_game_name(game, skip_games)
    ]


def require_selected_games(games: list[str]) -> list[str]:
    """Return filtered games or raise a descriptive error."""
    if games:
        return games

    raise ValueError("No games remaining after applying --skip-games.")
