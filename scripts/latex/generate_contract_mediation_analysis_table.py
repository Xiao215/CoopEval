#!/usr/bin/env python3
"""Generate a LaTeX table for mediation and contracting design analysis."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

from coopeval.config import LATEX_DIR
from coopeval.script_utils.display_helper import (
    format_mechanism_name,
    format_model_name,
    sort_agents,
    sort_games,
)
from coopeval.analysis.contract_design_quality import CONTRACT_ANALYZER
from coopeval.analysis.mechanism_design import (
    AgentMetrics,
    DesignAnalyzer,
    compute_metrics,
    extract_mechanism_entries,
)
from coopeval.analysis.mediation_design_quality import MEDIATION_ANALYZER
from tablelib.cli import (
    add_common_arguments,
    require_selected_games,
    resolve_paths,
)
from tablelib.data_loader import resolve_tournament_result_dirs
from tablelib.game_table import apply_ranking_format
from tablelib.metrics import compute_mean_stderr, save_table

RATE_ROWS = (
    ("NS", "nash_equilibrium_rate"),
    ("WD", "weak_dominance_rate"),
)
MECHANISMS = (
    ("Mediation", MEDIATION_ANALYZER),
    ("Contracting", CONTRACT_ANALYZER),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the mediation/contracting design-analysis table.",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--precision",
        type=int,
        default=1,
        help="Decimal places for percentages (default: 1).",
    )
    parser.add_argument(
        "--no-stderr",
        dest="show_stderr",
        action="store_false",
        default=True,
        help="Hide standard errors in the table.",
    )
    parser.add_argument(
        "--per-game",
        action="store_true",
        default=False,
        help=(
            "Emit one table per selected game instead of averaging across games."
        ),
    )
    return parser


def _merge_agent_metrics(
    target: dict[str, dict[str, AgentMetrics]],
    source: dict[str, dict[str, AgentMetrics]],
) -> None:
    for game_name, agent_map in source.items():
        if game_name not in target:
            target[game_name] = {}

        for agent_name, stats in agent_map.items():
            existing = target[game_name].get(agent_name)
            if existing is None:
                target[game_name][agent_name] = AgentMetrics(
                    total=stats.total,
                    weak_dominance=stats.weak_dominance,
                    nash_equilibrium=stats.nash_equilibrium,
                )
                continue

            target[game_name][agent_name] = AgentMetrics(
                total=existing.total + stats.total,
                weak_dominance=existing.weak_dominance + stats.weak_dominance,
                nash_equilibrium=existing.nash_equilibrium
                + stats.nash_equilibrium,
            )


def _collect_metrics_for_analyzer(
    tournament_result_dirs: Sequence[Path],
    analyzer: DesignAnalyzer,
    skip_games: Sequence[str] | None,
) -> dict[str, dict[str, AgentMetrics]]:
    merged_metrics: dict[str, dict[str, AgentMetrics]] = {}

    for tournament_result_dir in tournament_result_dirs:
        design_index, games = extract_mechanism_entries(
            tournament_result_dir,
            analyzer,
            skip_games=skip_games,
        )
        if not design_index:
            continue

        result_metrics = compute_metrics(
            design_index,
            games,
            analyzer,
        )
        _merge_agent_metrics(merged_metrics, result_metrics)

    return merged_metrics


def _aggregate_rate(
    mechanism_metrics: dict[str, dict[str, AgentMetrics]],
    games: Sequence[str],
    model: str,
    rate_attr: str,
) -> tuple[float | None, float | None, int]:
    values: list[float] = []
    for game in games:
        agent_map = mechanism_metrics.get(game)
        if not agent_map or model not in agent_map:
            continue
        values.append(getattr(agent_map[model], rate_attr))

    if not values:
        return None, None, 0

    mean_val, stderr_val = compute_mean_stderr(values)
    return mean_val, stderr_val, len(values)


def _compute_game_rate(
    mechanism_metrics: dict[str, dict[str, AgentMetrics]],
    game: str,
    model: str,
    rate_attr: str,
) -> tuple[float | None, float | None, int]:
    agent_map = mechanism_metrics.get(game)
    if not agent_map or model not in agent_map:
        return None, None, 0

    stats = agent_map[model]
    mean_val = getattr(stats, rate_attr)
    if stats.total <= 1:
        return mean_val, 0.0, stats.total

    stderr_val = math.sqrt((mean_val * (1 - mean_val)) / stats.total)
    return mean_val, stderr_val, stats.total


def _format_percent_cell(
    mean_val: float | None,
    stderr_val: float | None,
    precision: int,
    show_stderr: bool,
    sample_count: int,
) -> tuple[str, str | None]:
    if mean_val is None:
        return "N/A", None

    mean_str = f"{mean_val * 100:.{precision}f}"
    if show_stderr and stderr_val is not None and sample_count > 1:
        return mean_str, f"$\\pm$ {stderr_val * 100:.{precision}f}"
    return mean_str, None


def _build_table_header(
    lines: list[str],
    *,
    tournament_result_dirs: Sequence[Path],
    caption: str,
    label: str,
    models: Sequence[str],
) -> None:
    lines.append("% Source folders:")
    for folder in tournament_result_dirs:
        lines.append(f"%   {folder}")

    col_spec = "ll||" + "r" * len(models)
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\scalebox{0.78}{")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header_parts = [r"\textbf{Mechanism}", r"\textbf{Metric}"]
    header_parts.extend(format_model_name(model) for model in models)
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")


def _append_mechanism_rows(
    lines: list[str],
    *,
    models: Sequence[str],
    precision: int,
    show_stderr: bool,
    mechanism_results: Sequence[tuple[str, dict[str, dict[str, AgentMetrics]]]],
    value_getter,
) -> None:
    for mech_idx, (mechanism_name, mechanism_metrics) in enumerate(
        mechanism_results
    ):
        display_name = format_mechanism_name(mechanism_name)

        for row_idx, (metric_label, rate_attr) in enumerate(RATE_ROWS):
            row_parts: list[str] = []
            if row_idx == 0:
                row_parts.append(
                    f"\\multirow{{{len(RATE_ROWS)}}}{{*}}"
                    f"{{\\textbf{{{display_name}}}}}"
                )
            else:
                row_parts.append("")

            row_parts.append(metric_label)

            model_values: list[
                tuple[int, float | None, float | None, str, str | None]
            ] = []
            for model_idx, model in enumerate(models):
                mean_val, stderr_val, sample_count = value_getter(
                    mechanism_metrics,
                    model,
                    rate_attr,
                )
                mean_str, stderr_str = _format_percent_cell(
                    mean_val,
                    stderr_val,
                    precision,
                    show_stderr,
                    sample_count,
                )
                model_values.append(
                    (
                        model_idx,
                        mean_val,
                        (
                            stderr_val
                            if show_stderr and sample_count > 1
                            else None
                        ),
                        mean_str,
                        stderr_str,
                    )
                )

            row_parts.extend(apply_ranking_format(model_values, "maximize"))
            lines.append(" & ".join(row_parts) + r" \\")

        if mech_idx < len(mechanism_results) - 1:
            lines.append(r"\midrule")


def _finish_table(lines: list[str]) -> None:
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")


def _build_aggregate_table(
    tournament_result_dirs: Sequence[Path],
    models: Sequence[str],
    selected_games: Sequence[str],
    mechanism_results: Sequence[tuple[str, dict[str, dict[str, AgentMetrics]]]],
    precision: int,
    show_stderr: bool,
) -> str:
    lines: list[str] = []

    game_count = len(selected_games)
    if game_count == 1:
        game_phrase = "1 selected game"
    else:
        game_phrase = f"{game_count} selected games"

    caption = (
        "Design-analysis results for Mediation and Contracting, "
        f"aggregated across {game_phrase}. "
        "`NS' $(\\uparrow)$ is the percentage of submitted designs that "
        "make full delegation (Mediation) or mutual cooperation "
        "(Contracting) Nash stable. `WD' $(\\uparrow)$ is the percentage "
        "that make that target action weakly dominant. Values are shown "
        "as percentages"
        + (
            ", with standard error across games when multiple games are aggregated."
            if show_stderr
            else "."
        )
    )
    _build_table_header(
        lines,
        tournament_result_dirs=tournament_result_dirs,
        caption=caption,
        label="tab:contract_mediation_analysis",
        models=models,
    )

    def value_getter(
        mechanism_metrics: dict[str, dict[str, AgentMetrics]],
        model: str,
        rate_attr: str,
    ) -> tuple[float | None, float | None, int]:
        return _aggregate_rate(
            mechanism_metrics,
            selected_games,
            model,
            rate_attr,
        )

    _append_mechanism_rows(
        lines,
        models=models,
        precision=precision,
        show_stderr=show_stderr,
        mechanism_results=mechanism_results,
        value_getter=value_getter,
    )
    _finish_table(lines)
    return "\n".join(lines)


def _build_game_table(
    game: str,
    *,
    tournament_result_dirs: Sequence[Path],
    models: Sequence[str],
    mechanism_results: Sequence[tuple[str, dict[str, dict[str, AgentMetrics]]]],
    precision: int,
    show_stderr: bool,
) -> str:
    lines: list[str] = []

    caption = (
        f"Design-analysis results for {game}. "
        "`NS' $(\\uparrow)$ is the percentage of submitted designs that "
        "make full delegation (Mediation) or mutual cooperation "
        "(Contracting) Nash stable. `WD' $(\\uparrow)$ is the percentage "
        "that make that target action weakly dominant. Values are shown "
        "as percentages"
        + (
            ", with binomial standard error across submitted designs."
            if show_stderr
            else "."
        )
    )
    _build_table_header(
        lines,
        tournament_result_dirs=tournament_result_dirs,
        caption=caption,
        label=f"tab:contract_mediation_analysis_{game.lower()}",
        models=models,
    )

    def value_getter(
        mechanism_metrics: dict[str, dict[str, AgentMetrics]],
        model: str,
        rate_attr: str,
    ) -> tuple[float | None, float | None, int]:
        return _compute_game_rate(
            mechanism_metrics,
            game,
            model,
            rate_attr,
        )

    _append_mechanism_rows(
        lines,
        models=models,
        precision=precision,
        show_stderr=show_stderr,
        mechanism_results=mechanism_results,
        value_getter=value_getter,
    )
    _finish_table(lines)
    return "\n".join(lines)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    tournament_result_dirs = resolve_tournament_result_dirs(
        resolve_paths(args.tournament_result_dirs)
    )

    mechanism_metrics: list[tuple[str, dict[str, dict[str, AgentMetrics]]]] = []
    available_games: set[str] = set()

    for mechanism_name, analyzer in MECHANISMS:
        metrics = _collect_metrics_for_analyzer(
            tournament_result_dirs,
            analyzer,
            args.skip_games,
        )
        mechanism_metrics.append((mechanism_name, metrics))
        available_games.update(metrics)

    selected_games = require_selected_games(sort_games(available_games))

    selected_models: set[str] = set()
    for _mechanism_name, metrics in mechanism_metrics:
        for game in selected_games:
            selected_models.update(metrics.get(game, {}))
    models = sort_agents(selected_models)

    print(f"Using {len(tournament_result_dirs)} tournament result dir(s)")
    print(
        f"Selected games ({len(selected_games)}): {', '.join(selected_games)}"
    )
    print(f"Models ({len(models)}): {', '.join(models)}")

    if args.per_game:
        output_name = "table_contract_mediation_analysis_per_game.tex"
    else:
        output_name = "table_contract_mediation_analysis.tex"

    if args.per_game:
        tables = [
            _build_game_table(
                game,
                tournament_result_dirs=tournament_result_dirs,
                models=models,
                mechanism_results=mechanism_metrics,
                precision=args.precision,
                show_stderr=args.show_stderr,
            )
            for game in selected_games
        ]
        table_latex = "\n\n".join(tables)
    else:
        table_latex = _build_aggregate_table(
            tournament_result_dirs=tournament_result_dirs,
            models=models,
            selected_games=selected_games,
            mechanism_results=mechanism_metrics,
            precision=args.precision,
            show_stderr=args.show_stderr,
        )

    output_path = LATEX_DIR / output_name
    save_table(table_latex, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
