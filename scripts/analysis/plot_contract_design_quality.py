#!/usr/bin/env python3
"""Plot CoopEval contracting design-quality analysis."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from coopeval.config import FIGURE_DIR
from coopeval.analysis.contract_design_quality import CONTRACT_ANALYZER
from coopeval.analysis.mechanism_design import (
    collect_mechanism_designs,
    compute_metrics,
    plot_tiered_rates,
    print_metric_table,
    print_summary,
)
from coopeval.script_utils.result_loader import DEFAULT_SKIP_GAMES
from coopeval.utils.json_io import clean_path
from coopeval.script_utils.figure_exports import save_matplotlib_figure


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Group CoopEval contract designs per game/agent."
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
            "Games to skip before aggregation "
            "(default: %(default)s; pass with no values to include all)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=clean_path,
        default=FIGURE_DIR,
        help="Directory where analysis figures are written.",
    )
    return parser


def main() -> None:
    """Run the contracting design-quality plotting workflow."""

    args = _build_parser().parse_args()
    design_index, games = collect_mechanism_designs(
        args.tournament_result_dirs,
        CONTRACT_ANALYZER,
        skip_games=args.skip_games,
    )

    print_summary(design_index, CONTRACT_ANALYZER)
    if not design_index:
        return

    metrics = compute_metrics(design_index, games, CONTRACT_ANALYZER)
    print_metric_table(metrics, CONTRACT_ANALYZER)

    fig = plot_tiered_rates(
        metrics, title_prefix=CONTRACT_ANALYZER.figure_title_prefix or ""
    )
    analysis_dir = args.output_dir / CONTRACT_ANALYZER.figure_subdir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    base_path = analysis_dir / CONTRACT_ANALYZER.figure_stem
    saved_paths = save_matplotlib_figure(
        fig, base_path, ("png",), dpi=300, format_subdirs=False
    )
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
