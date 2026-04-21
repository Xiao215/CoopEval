#!/usr/bin/env python3
"""Metric computations for LaTeX table generation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from coopeval.visualization.analysis_utils import NormalizeScore
from .data_loader import (
    ExperimentData,
    validate_metric_consistency,
)

METRIC_LABELS = {
    "mean": "Mean",
    "rd": "Fitness",
    "dr": "DR",
}

LLM_AVERAGE = "LLM_AVERAGE"
SOCIAL_DILEMMAS = [
    "PrisonersDilemma",
    "PublicGoods",
    "TravellersDilemma",
    "TrustGame",
]

raw_data: dict[
    str, dict[str, dict[int, dict[str, dict[str, float | None]]]]
] = {}


def compute_mean_stderr(values: Sequence[float]) -> tuple[float, float]:
    """Compute mean and standard error from a sequence of values."""
    n = len(values)
    mean = sum(values) / n

    if n == 1:
        return mean, 0.0

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    stderr = math.sqrt(variance / n)
    return mean, stderr


def build_data_structure(
    grouped_experiments: dict[tuple[str, str], list[ExperimentData]],
    models: Sequence[str],
    metrics: Sequence[str],
) -> dict[str, dict]:
    """Build the raw_data tensor at replication granularity."""
    raw_data.clear()
    game_configs: dict[str, dict] = {}

    for (game, mechanism), experiments in grouped_experiments.items():
        group_key = (game, mechanism)
        if "rd" in metrics:
            validate_metric_consistency(experiments, "rd_fitness", group_key)
        if "dr" in metrics:
            validate_metric_consistency(
                experiments, "deviation_ranks", group_key
            )

        if game not in raw_data:
            raw_data[game] = {}
            game_configs[game] = experiments[0].game_config

        if mechanism not in raw_data[game]:
            raw_data[game][mechanism] = {}

        for rep_idx, exp in enumerate(experiments):
            raw_data[game][mechanism][rep_idx] = {}
            for model in models:
                dev_rank = None
                if exp.deviation_ranks and model in exp.deviation_ranks:
                    dev_rank = float(exp.deviation_ranks[model])

                raw_data[game][mechanism][rep_idx][model] = {
                    "payoff": exp.model_scores[model],
                    "fitness": (
                        exp.rd_fitness[model] if exp.rd_fitness else None
                    ),
                    "population": (
                        exp.rd_populations[model]
                        if exp.rd_populations
                        else None
                    ),
                    "deviation_rank": dev_rank,
                }

    return game_configs


def _extract_metric_value(
    game: str, mechanism: str, rep: int, model: str, metric: str
) -> float:
    data = raw_data[game][mechanism][rep][model]

    if metric == "mean":
        return data["payoff"]
    if metric == "rd":
        return data["fitness"]
    if metric == "dr":
        return data["deviation_rank"]
    raise ValueError(f"Unknown metric: {metric}")


def get_models_for(game: str, mechanism: str) -> list[str]:
    reps = list(raw_data[game][mechanism])
    first_models = set(raw_data[game][mechanism][reps[0]])
    for rep in reps[1:]:
        if set(raw_data[game][mechanism][rep]) != first_models:
            raise ValueError("Inconsistent models across replications.")
    return list(first_models)


def _compute_llm_average_for_rep(
    game: str, mechanism: str, metric: str, rep: int, models: Sequence[str]
) -> float:
    values = [
        _extract_metric_value(game, mechanism, rep, model, metric)
        for model in models
    ]

    if metric == "rd":
        weights = [
            raw_data[game][mechanism][rep][model]["population"]
            for model in models
        ]
    else:
        weights = [1.0 / len(models)] * len(models)

    total = sum(weights)
    if not math.isclose(total, 1.0, abs_tol=0.01):
        raise ValueError(f"Weights do not sum to 1: {weights}")
    normalized_weights = [w / total for w in weights]
    return sum(
        value * weight for value, weight in zip(values, normalized_weights)
    )


def compute_per_game_cell_raw(
    game: str, mechanism: str, metric: str, model: str
) -> tuple[float, float]:
    replications = list(raw_data[game][mechanism])
    models = get_models_for(game, mechanism)

    per_rep_values = []
    for rep in replications:
        if model == LLM_AVERAGE:
            value = _compute_llm_average_for_rep(
                game, mechanism, metric, rep, models
            )
        else:
            value = _extract_metric_value(game, mechanism, rep, model, metric)
        per_rep_values.append(value)

    return compute_mean_stderr(per_rep_values)


def _normalize_mean_and_stderr(
    game: str,
    mean_val: float,
    stderr_val: float,
    game_configs: dict[str, dict],
) -> tuple[float, float]:
    """Apply a game's linear payoff normalization to mean and stderr."""
    normalizer = NormalizeScore(game, game_configs[game])
    scale = normalizer.coop_payoff - normalizer.ne_payoff
    if scale == 0:
        return 0.0, 0.0
    return normalizer.normalize(mean_val), stderr_val / abs(scale)


def compute_aggregate_cell_raw(
    mechanism: str,
    metric: str,
    model: str,
    game_configs: dict[str, dict],
    games: Sequence[str] | None = None,
    normalize: bool = True,
) -> tuple[float, float]:
    selected_games = games if games is not None else SOCIAL_DILEMMAS
    relevant_games = [
        game
        for game in selected_games
        if game in raw_data and mechanism in raw_data[game]
    ]

    if len(relevant_games) == 1:
        game = relevant_games[0]
        mean_val, stderr_val = compute_per_game_cell_raw(
            game, mechanism, metric, model
        )
        if normalize and metric in ("mean", "rd"):
            return _normalize_mean_and_stderr(
                game, mean_val, stderr_val, game_configs
            )
        return mean_val, stderr_val

    per_game_values = []
    for game in relevant_games:
        if game not in raw_data or mechanism not in raw_data[game]:
            continue

        cell_mean, _ = compute_per_game_cell_raw(game, mechanism, metric, model)
        if normalize and metric in ("mean", "rd"):
            normalizer = NormalizeScore(game, game_configs[game])
            per_game_values.append(normalizer.normalize(cell_mean))
        else:
            per_game_values.append(cell_mean)

    return compute_mean_stderr(per_game_values)


def compute_aggregate_color_value(
    mechanism: str,
    metric: str,
    model: str,
    game_configs: dict[str, dict],
    games: Sequence[str] | None = None,
) -> float | None:
    """Return the normalized heatmap value used for aggregate coloring."""
    if metric == "dr":
        mean_val, _ = compute_aggregate_cell_raw(
            mechanism,
            metric,
            model,
            game_configs,
            games,
            normalize=False,
        )
        return mean_val

    if metric not in ("mean", "rd"):
        return None

    selected_games = games if games is not None else SOCIAL_DILEMMAS
    relevant_games = [
        game
        for game in selected_games
        if game in raw_data and mechanism in raw_data[game]
    ]
    if not relevant_games:
        return None

    normalized_per_game: list[float] = []
    for game in relevant_games:
        cell_mean, _ = compute_per_game_cell_raw(game, mechanism, metric, model)
        normalized_mean, _ = _normalize_mean_and_stderr(
            game, cell_mean, 0.0, game_configs
        )
        normalized_per_game.append(normalized_mean)

    return compute_mean_stderr(normalized_per_game)[0]


def save_table(table_latex: str, output_path: Path) -> None:
    """Save LaTeX table to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{table_latex}\n", encoding="utf-8")
