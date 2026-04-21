#!/usr/bin/env python3
"""Generate the combined normalized per-game LaTeX tables."""

from __future__ import annotations

import argparse

from coopeval.config import LATEX_DIR
from tablelib.cli import (
    add_common_arguments,
    filter_skipped_games,
    require_selected_games,
    resolve_paths,
)
from tablelib.data_loader import (
    collect_grouped_experiments,
    resolve_tournament_result_dirs,
    validate_experiment_groups,
)
from tablelib.game_table import generate_game_table
from tablelib.metrics import build_data_structure, save_table


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the combined normalized per-game LaTeX tables.",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["mean", "rd", "dr"],
        default=["mean", "rd", "dr"],
        help="Metrics to include (default: mean rd dr).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal places for normalized scores (default: 3).",
    )
    parser.add_argument(
        "--no-stderr",
        dest="show_stderr",
        action="store_false",
        default=True,
        help="Hide standard errors in per-game tables.",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        default=False,
        help="Add LaTeX heatmap cell coloring to the LLM Average column.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    tournament_result_dirs = resolve_tournament_result_dirs(
        resolve_paths(args.tournament_result_dirs)
    )

    grouped, failures = collect_grouped_experiments(
        tournament_result_dirs, args.metrics
    )
    print(
        f"\nTotal: {sum(len(exps) for exps in grouped.values())} successful, "
        f"{len(failures)} failed"
    )
    print(f"Grouped into {len(grouped)} game-mechanism combinations")

    mechanisms, games, models, _ = validate_experiment_groups(grouped)
    print(
        f"Validation passed: {len(games)} games, "
        f"{len(mechanisms)} mechanisms, {len(models)} models\n"
    )

    games = require_selected_games(filter_skipped_games(games, args.skip_games))
    print(f"Selected games ({len(games)}): {', '.join(games)}\n")
    print(
        f"Selected mechanisms ({len(mechanisms)}): "
        f"{', '.join(mechanisms)}\n"
    )

    game_configs = build_data_structure(grouped, models, args.metrics)

    per_game_tables = []
    for game in games:
        table_latex = generate_game_table(
            game,
            mechanisms,
            models,
            args.precision,
            args.metrics,
            game_configs[game],
            tournament_result_dirs,
            show_stderr=args.show_stderr,
            colorize_cells=args.color,
        )
        per_game_tables.append(table_latex)

    combined_output = "\n\n".join(per_game_tables)
    combined_path = LATEX_DIR / "table_all_games.tex"
    save_table(combined_output, combined_path)
    print(f"Saved combined per-game tables: {combined_path}")

    if failures:
        print(f"\n{'=' * 80}")
        print(f"Failed to parse {len(failures)} experiment(s):")
        for folder, error in failures:
            print(f"  - {folder}")
            print(f"    Error: {type(error).__name__}: {error}")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
