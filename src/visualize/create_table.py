#!/usr/bin/env python3
"""
LaTeX Table Generator for Batch Experiment Results

This script generates LaTeX tables from batch experiment results, showing
model performance across different mechanisms and games.

Usage:
    python src/visualize/create_table.py outputs/2026/01/14/16:14/
    python src/visualize/create_table.py outputs/2026/01/14/16:14/ --output tables/
    python src/visualize/create_table.py outputs/2026/01/14/16:14/ --quiet

Required LaTeX packages:
    \\usepackage{booktabs}
"""

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.visualize.analysis_utils import (
    NormalizeScore,
    discover_experiment_subfolders,
    load_json as load_json_file,
    simplify_model_name,
    sort_games,
    sort_mechanisms,
    sort_models,
    validate_dict_consistency,
    validate_folder_count_consistency,
    validate_list_consistency,
)

# Map metric names to LaTeX display labels
METRIC_LABELS = {
    "mean": "Mean",
    "rd": "RD",
    "dr": "DR",
}


def is_reputation_mechanism(mechanism_type: str) -> bool:
    """Check if mechanism is any variant of Reputation."""
    return mechanism_type.lower() in ["reputation", "reputationfirstorder"]


@dataclass
class ExperimentData:
    """Represents a single experiment's data."""

    mechanism: str
    game: str
    model_scores: Dict[str, float]
    folder_path: Path
    game_config: dict  # Store full game config for score normalization
    eval_config: dict  # Store evaluation config for consistency checking
    # New fields for additional metrics
    rd_fitness: Optional[Dict[str, float]] = None  # Required (None only for reputation)
    deviation_ranks: Optional[Dict[str, str]] = None  # Required (None only for reputation), store as rank strings


def load_json(path: Path) -> dict:
    """Load JSON file with error handling."""
    return load_json_file(path)


def compute_deviation_ranks(ratings: Dict[str, float]) -> Dict[str, str]:
    """
    Convert deviation ratings to ranks.

    Args:
        ratings: Dict mapping model names to float ratings (higher is better)

    Returns:
        Dict mapping model names to rank strings
    """
    # Sort by rating value (descending - higher is better)
    sorted_models = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    # Handle ties using average rank
    ranks = {}
    i = 0
    while i < len(sorted_models):
        # Find all models with same rating (tie)
        rating = sorted_models[i][1]
        tied_models = [sorted_models[i][0]]
        j = i + 1
        while j < len(sorted_models) and sorted_models[j][1] == rating:
            tied_models.append(sorted_models[j][0])
            j += 1

        # Compute average rank for tied models
        # Ranks are 1-indexed: positions i+1 to j
        avg_rank = sum(range(i + 1, j + 1)) / len(tied_models)

        # Format rank as string
        if avg_rank == int(avg_rank):
            # No tie, show as integer
            rank_str = str(int(avg_rank))
        else:
            # Tie, show decimal (e.g., "2.5")
            rank_str = f"{avg_rank:.1f}"

        for model in tied_models:
            ranks[model] = rank_str

        i = j

    return ranks


def compute_mean_stderr(values: List[float]) -> Tuple[float, float]:
    """
    Compute mean and standard error from a list of values.

    Args:
        values: List of numeric values

    Returns:
        Tuple of (mean, stderr)
    """
    n = len(values)
    mean = sum(values) / n

    if n == 1:
        stderr = 0.0
    else:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        stderr = math.sqrt(variance / n)

    return mean, stderr


def validate_metric_consistency(
    experiments: List[ExperimentData],
    metric_name: str,
    group_key: Tuple[str, str]
) -> bool:
    """
    Validate that all experiments have consistent None/non-None pattern for a metric.

    Args:
        experiments: List of experiments to validate
        metric_name: "rd_fitness" or "deviation_ranks"
        group_key: (game, mechanism) for error messages

    Returns:
        True if metric exists (non-None), False if all are None

    Raises:
        ValueError: If experiments have inconsistent None patterns
    """
    if metric_name == "rd_fitness":
        values = [exp.rd_fitness for exp in experiments]
    else:  # "deviation_ranks"
        assert metric_name == "deviation_ranks"
        values = [exp.deviation_ranks for exp in experiments]

    none_count = sum(1 for v in values if v is None)

    if none_count == len(values):
        # All None - this is expected for reputation mechanisms
        return False
    elif none_count == 0:
        # All have data - expected for non-reputation mechanisms
        return True
    else:
        # Mixed - this is an error
        folder_paths = [str(exp.folder_path.name) for exp in experiments]
        none_indices = [i for i, v in enumerate(values) if v is None]
        non_none_indices = [i for i, v in enumerate(values) if v is not None]
        raise ValueError(
            f"Inconsistent {metric_name} data in {group_key}:\n"
            f"  Folders with None: {[folder_paths[i] for i in none_indices]}\n"
            f"  Folders with data: {[folder_paths[i] for i in non_none_indices]}\n"
            f"  All experiments in a group must consistently have or lack this metric."
        )


def aggregate_metric_across_folders(
    experiments: List[ExperimentData],
    metric_name: str,
    models: List[str]
) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate metric across multiple experiment folders.

    Args:
        experiments: List of ExperimentData for same (game, mechanism)
        metric_name: "mean", "rd", or "dr"
        models: Expected model list

    Returns:
        Dict mapping model → (mean, stderr)

    Raises:
        ValueError: If data is missing or inconsistent
    """
    # Collect values for each model
    model_values: Dict[str, List[float]] = {model: [] for model in models}

    for exp in experiments:
        # Extract metric data
        if metric_name == "mean":
            data_dict = exp.model_scores
        elif metric_name == "rd":
            if exp.rd_fitness is None:
                raise ValueError(
                    f"Missing RD fitness in {exp.folder_path} "
                    f"for {exp.mechanism}_{exp.game}"
                )
            data_dict = exp.rd_fitness
        else:  # "dr"
            if exp.deviation_ranks is None:
                raise ValueError(
                    f"Missing deviation ranks in {exp.folder_path} "
                    f"for {exp.mechanism}_{exp.game}"
                )
            # Convert rank strings to floats
            data_dict = {
                m: float(rank_str)
                for m, rank_str in exp.deviation_ranks.items()
            }

        # Collect values
        for model in models:
            if model not in data_dict:
                raise ValueError(
                    f"Missing {model} in {exp.folder_path} "
                    f"for {exp.mechanism}_{exp.game}"
                )
            value = data_dict[model]
            if value is None:
                raise ValueError(
                    f"None value for {model} in {exp.folder_path}"
                )
            model_values[model].append(value)

    # Compute mean and stderr for each model
    results = {}
    for model, values in model_values.items():
        results[model] = compute_mean_stderr(values)

    return results


def parse_batch_folder(
    batch_path: Path,
    metrics: List[str]
) -> tuple[
    Dict[Tuple[str, str], List[ExperimentData]],
    List[Tuple[Path, Exception]]
]:
    """
    Parse batch folder and group experiment data by (game, mechanism).

    Args:
        batch_path: Path to batch experiment folder
        metrics: List of metrics to load (subset of ["mean", "rd", "dr"])

    Returns:
        Tuple of:
        - Dict mapping (game, mechanism) → List[ExperimentData]
        - List of (failed_path, error) tuples

    Raises:
        FileNotFoundError: If batch folder doesn't exist
        NotADirectoryError: If path is not a directory
    """
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch folder not found: {batch_path}")
    if not batch_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {batch_path}")

    grouped_experiments: Dict[Tuple[str, str], List[ExperimentData]] = defaultdict(list)
    failed_experiments = []

    # Scan for subdirectories (skip configs and other non-experiment folders)
    subdirs = discover_experiment_subfolders(batch_path)

    for subdir in subdirs:
        try:
            # Check for required files
            config_path = subdir / "config.json"
            payoff_path = subdir / "agent_average_payoff.json"

            if not config_path.exists() or not payoff_path.exists():
                raise FileNotFoundError(
                    f"Missing required files in {subdir}"
                )

            # Load config
            config = load_json(config_path)

            # Load payoffs
            payoffs = load_json(payoff_path)

            # Extract mechanism and game from config
            mechanism = config["mechanism"]["type"]
            game = config["game"]["type"]
            game_config = config["game"]
            eval_config = config["evaluation"]

            # Load metrics (REQUIRED for non-reputation mechanisms, based on requested metrics)
            if is_reputation_mechanism(mechanism):
                # Reputation mechanism: explicitly use None (will display N/A)
                rd_fitness = None
                deviation_ranks = None
            else:
                # All other mechanisms: Load only requested metric files
                rd_fitness = None
                deviation_ranks = None

                # Load RD fitness if requested
                if "rd" in metrics:
                    rd_fitness_path = subdir / "replicator_dynamics_fitness.json"
                    if not rd_fitness_path.exists():
                        raise FileNotFoundError(
                            f"Missing replicator_dynamics_fitness.json in {subdir} for mechanism {mechanism}"
                        )
                    rd_fitness_data = load_json(rd_fitness_path)
                    if rd_fitness_data is None:
                        raise ValueError(f"Failed to parse replicator_dynamics_fitness.json in {subdir}")
                    rd_fitness = {
                        model: data["fitness"]
                        for model, data in rd_fitness_data.items()
                    }

                # Load deviation ratings if requested
                if "dr" in metrics:
                    deviation_ratings_path = subdir / "deviation_ratings.json"
                    if not deviation_ratings_path.exists():
                        raise FileNotFoundError(
                            f"Missing deviation_ratings.json in {subdir} for mechanism {mechanism}"
                        )
                    deviation_ratings_data = load_json(deviation_ratings_path)
                    if deviation_ratings_data is None:
                        raise ValueError(f"Failed to parse deviation_ratings.json in {subdir}")
                    deviation_ranks = compute_deviation_ranks(deviation_ratings_data)

            # Skip reputation mechanisms
            if is_reputation_mechanism(mechanism):
                print(f"Skipping reputation mechanism: {mechanism}_{game}")
                continue

            # Create experiment data
            exp_data = ExperimentData(
                mechanism=mechanism,
                game=game,
                model_scores=payoffs,
                folder_path=subdir,
                game_config=game_config,
                eval_config=eval_config,
                rd_fitness=rd_fitness,
                deviation_ranks=deviation_ranks,
            )

            # Group by (game, mechanism)
            group_key = (game, mechanism)
            grouped_experiments[group_key].append(exp_data)

        except Exception as e:
            print(f"WARNING: Failed to parse experiment in {subdir}")
            print(f"  Error: {type(e).__name__}: {e}")
            failed_experiments.append((subdir, e))

    return grouped_experiments, failed_experiments


def build_data_structure(
    grouped_experiments: Dict[Tuple[str, str], List[ExperimentData]],
    models: List[str],
    metrics: List[str]
) -> tuple[
    Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    Dict[str, Dict[str, Dict[str, Optional[Tuple[float, float]]]]],
    Dict[str, Dict[str, Dict[str, Optional[Tuple[float, float]]]]],
    Dict[str, dict]
]:
    """
    Build nested data structures with aggregated metrics.

    Args:
        grouped_experiments: Dict mapping (game, mechanism) → List[ExperimentData]
        models: Expected model list
        metrics: List of metrics to aggregate

    Returns:
        Tuple of:
        - payoffs[game][mechanism][model] = (mean, stderr)
        - rd_fitness[game][mechanism][model] = (mean, stderr) or None
        - deviation_ranks[game][mechanism][model] = (mean, stderr) or None
        - game_configs[game] = config dict
    """
    payoffs = {}
    rd_fitness = {}
    deviation_ranks = {}
    game_configs = {}

    for (game, mechanism), experiments in grouped_experiments.items():
        # Initialize
        if game not in payoffs:
            payoffs[game] = {}
            rd_fitness[game] = {}
            deviation_ranks[game] = {}
            game_configs[game] = experiments[0].game_config

        payoffs[game][mechanism] = {}
        rd_fitness[game][mechanism] = {}
        deviation_ranks[game][mechanism] = {}

        # Aggregate metrics
        if "mean" in metrics:
            payoffs[game][mechanism] = aggregate_metric_across_folders(
                experiments, "mean", models
            )

        if "rd" in metrics:
            # Validate consistency and check if metric exists
            has_rd_data = validate_metric_consistency(experiments, "rd_fitness", (game, mechanism))
            if not has_rd_data:
                # Reputation mechanism - all experiments have None
                for model in models:
                    rd_fitness[game][mechanism][model] = None
            else:
                # All experiments have data - aggregate
                rd_fitness[game][mechanism] = aggregate_metric_across_folders(
                    experiments, "rd", models
                )

        if "dr" in metrics:
            # Validate consistency and check if metric exists
            has_dr_data = validate_metric_consistency(experiments, "deviation_ranks", (game, mechanism))
            if not has_dr_data:
                # Reputation mechanism - all experiments have None
                for model in models:
                    deviation_ranks[game][mechanism][model] = None
            else:
                # All experiments have data - aggregate
                deviation_ranks[game][mechanism] = aggregate_metric_across_folders(
                    experiments, "dr", models
                )

    return payoffs, rd_fitness, deviation_ranks, game_configs


def validate_experiment_groups(
    grouped_experiments: Dict[Tuple[str, str], List[ExperimentData]]
) -> Tuple[List[str], List[str], List[str], dict]:
    """
    Validate experiment groups.

    Checks:
    1. All groups have same folder count
    2. Within each group: same models, game config, eval config
    3. Across groups: same models, eval config

    Args:
        grouped_experiments: Dict mapping (game, mechanism) → List[ExperimentData]

    Returns:
        Tuple of (mechanisms, games, models, eval_config)

    Raises:
        AssertionError: If validation fails
        ValueError: If validation fails
    """
    # Cross-group validation: folder count
    expected_folder_count = validate_folder_count_consistency(grouped_experiments)
    print(f"All groups have {expected_folder_count} folder(s) - validation passed")

    # Validate each group and extract canonical values
    canonical_mechanisms = set()
    canonical_games = set()
    canonical_models = None
    canonical_eval_config = None

    for group_key, experiments in grouped_experiments.items():
        game, mechanism = group_key
        canonical_mechanisms.add(mechanism)
        canonical_games.add(game)

        # Extract data for validation
        folder_paths = [str(exp.folder_path) for exp in experiments]
        model_lists = [sorted(exp.model_scores.keys()) for exp in experiments]
        eval_configs = [exp.eval_config for exp in experiments]
        game_configs = [exp.game_config for exp in experiments]

        # Within-group validation
        validated_models = validate_list_consistency(
            model_lists, folder_paths, group_key, "model list"
        )

        validate_dict_consistency(
            eval_configs, folder_paths, group_key, "evaluation config"
        )

        validate_dict_consistency(
            game_configs, folder_paths, group_key, "game config"
        )

        # Cross-group validation
        if canonical_models is None:
            canonical_models = validated_models
            canonical_eval_config = eval_configs[0]
        else:
            if validated_models != canonical_models:
                raise ValueError(
                    f"Model list mismatch across groups:\n"
                    f"  Expected: {canonical_models}\n"
                    f"  Got in {group_key}: {validated_models}"
                )
            if eval_configs[0] != canonical_eval_config:
                raise ValueError(
                    f"Eval config mismatch across groups:\n"
                    f"  Expected: {canonical_eval_config}\n"
                    f"  Got in {group_key}: {eval_configs[0]}"
                )

    # Sort and return
    return (
        sort_mechanisms(list(canonical_mechanisms)),
        sort_games(list(canonical_games)),
        sort_models(canonical_models),
        canonical_eval_config
    )


def validate_data_consistency(
    data: Dict[str, Dict[str, Dict[str, float]]],
    canonical_mechanisms: List[str],
    canonical_games: List[str],
    canonical_models: List[str],
) -> None:
    """
    Validate that all mechanism×game×model combinations are present.

    Args:
        data: Nested dictionary with scores
        canonical_mechanisms: Expected list of mechanisms
        canonical_games: Expected list of games
        canonical_models: Expected list of models

    Note:
        Missing data will be handled gracefully by showing "Unav." in tables
    """
    # Validation disabled - we now handle missing data gracefully in table generation
    # missing_combinations = []
    #
    # for game in canonical_games:
    #     if game not in data:
    #         for mechanism in canonical_mechanisms:
    #             for model in canonical_models:
    #                 missing_combinations.append((game, mechanism, model))
    #         continue
    #
    #     for mechanism in canonical_mechanisms:
    #         if mechanism not in data[game]:
    #             for model in canonical_models:
    #                 missing_combinations.append((game, mechanism, model))
    #             continue
    #
    #         for model in canonical_models:
    #             if model not in data[game][mechanism]:
    #                 missing_combinations.append((game, mechanism, model))
    #
    # if missing_combinations:
    #     error_msg = f"Data inconsistency detected! Missing {len(missing_combinations)} combinations:\n"
    #     # Show first 10 missing combinations
    #     for game, mechanism, model in missing_combinations[:10]:
    #         error_msg += f"  - {game} × {mechanism} × {model}\n"
    #     if len(missing_combinations) > 10:
    #         error_msg += f"  ... and {len(missing_combinations) - 10} more\n"
    #     error_msg += "\nAll mechanism×game×model combinations must be present in batch experiments."
    #     raise ValueError(error_msg)
    pass


def format_score(score: Optional[float], precision: int = 3) -> str:
    """Format score for LaTeX table, handling None/missing values."""
    if score is None:
        return "N/A"
    return f"{score:.{precision}f}"


def format_score_with_stderr(
    result: Optional[Tuple[float, float]], precision: int = 3
) -> str:
    """Format mean ± stderr for LaTeX table."""
    if result is None:
        return "N/A"
    mean, stderr = result
    return f"{mean:.{precision}f} $\\pm$ {stderr:.{precision}f}"


def compute_summary_statistic(
    data_tuples: List[Tuple[float, float]],
    precision: int,
    show_stderr: bool,
    is_rank: bool = False
) -> str:
    """
    Compute summary statistic (mean ± stderr) from list of (mean, stderr) tuples.

    Args:
        data_tuples: List of (mean, stderr) tuples from individual data points
        precision: Decimal places for formatting
        show_stderr: Whether to include stderr in output
        is_rank: If True, uses rank formatting (.1f), otherwise uses given precision

    Returns:
        Formatted string with mean or mean ± stderr
    """
    if not data_tuples:
        return "N/A"

    # Extract means from tuples (ignore within-group stderr)
    means = [t[0] for t in data_tuples]
    avg, std = compute_mean_stderr(means)

    if is_rank:
        if show_stderr and len(means) > 1:
            return f"{avg:.1f} $\\pm$ {std:.1f}"
        else:
            return f"{avg:.1f}"
    else:
        if show_stderr and len(means) > 1:
            return f"{avg:.{precision}f} $\\pm$ {std:.{precision}f}"
        else:
            return f"{avg:.{precision}f}"


def generate_game_table(
    game: str,
    mechanisms: List[str],
    models: List[str],
    payoffs: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    rd_fitness: Dict[str, Dict[str, Dict[str, Optional[Tuple[float, float]]]]],
    deviation_ranks: Dict[str, Dict[str, Dict[str, Optional[Tuple[float, float]]]]],
    precision: int = 3,
    metrics: List[str] | None = None,
    source_folders: List[Path] | None = None,
    show_stderr: bool = False,
) -> str:
    """
    Generate LaTeX table for a single game with multicolumn headers.

    Args:
        game: Game name
        mechanisms: List of mechanism names (columns)
        models: List of model names (rows)
        payoffs: Nested dictionary with payoff scores
        rd_fitness: Nested dictionary with RD fitness values
        deviation_ranks: Nested dictionary with deviation ranks
        precision: Number of decimal places
        metrics: List of metrics to include (subset of ["mean", "rd", "dr"])
        source_folders: Optional list of source folder paths

    Returns:
        Complete LaTeX table as string
    """
    if metrics is None:
        metrics = ["mean", "rd", "dr"]

    # Start building table
    lines = []

    # Add source folder comments if provided
    if source_folders:
        lines.append("% Source folders:")
        for folder in source_folders:
            lines.append(f"%   {folder}")

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Results for {game}}}")
    game_slug = game.lower().replace(" ", "_")
    lines.append(f"\\label{{tab:{game_slug}}}")

    # Determine which metrics each mechanism supports
    mech_metrics = {}
    for mech in mechanisms:
        is_reputation = is_reputation_mechanism(mech)
        if is_reputation:
            # Reputation only supports mean
            mech_metrics[mech] = ["mean"] if "mean" in metrics else []
        else:
            # All other mechanisms support all requested metrics
            mech_metrics[mech] = metrics

    # Table header with vertical bars
    # Each mechanism has different number of sub-columns based on supported metrics
    col_spec_parts = ["l"]
    for mech in mechanisms:
        num_cols = len(mech_metrics[mech])
        if num_cols > 0:
            col_spec_parts.append("r" * num_cols)
    col_spec = "|".join(col_spec_parts)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # First header row: Mechanism names with multicolumn
    header_parts = ["\\textbf{Model}"]
    for mech in mechanisms:
        num_cols = len(mech_metrics[mech])
        if num_cols > 1:
            header_parts.append(f"\\multicolumn{{{num_cols}}}{{c}}{{\\textbf{{{mech}}}}}")
        elif num_cols == 1:
            header_parts.append(f"\\textbf{{{mech}}}")
    lines.append(" & ".join(header_parts) + " \\\\")

    # Second header row: Sub-column labels (only if needed)
    show_subheaders = any(len(mech_metrics[mech]) > 1 for mech in mechanisms)
    if show_subheaders:
        subheader_parts = [""]  # Empty for model column
        for mech in mechanisms:
            mech_metric_list = mech_metrics[mech]
            if len(mech_metric_list) > 1:
                metric_cols = " & ".join([f"\\textbf{{{METRIC_LABELS[m]}}}" for m in mech_metric_list])
                subheader_parts.append(metric_cols)
            elif len(mech_metric_list) == 1:
                # Single column - show the metric label
                subheader_parts.append(f"\\textbf{{{METRIC_LABELS[mech_metric_list[0]]}}}")
        lines.append(" & ".join(subheader_parts) + " \\\\")

    lines.append(r"\midrule")

    # Summary row (average across all models)
    row_parts = ["\\textbf{Average}"]
    for mech in mechanisms:
        metric_averages = {}
        mech_metric_list = mech_metrics[mech]

        # Check if data exists for this game-mechanism combination
        if game in payoffs and mech in payoffs[game]:
            # Calculate average for each metric this mechanism supports
            for metric in mech_metric_list:
                if metric == "mean":
                    tuples = [payoffs[game][mech][model] for model in models]
                    metric_averages[metric] = compute_summary_statistic(
                        tuples, precision, show_stderr, is_rank=False
                    )
                elif metric == "rd":
                    tuples = [rd_fitness[game][mech][model] for model in models
                              if rd_fitness[game][mech][model] is not None]
                    metric_averages[metric] = compute_summary_statistic(
                        tuples, precision, show_stderr, is_rank=False
                    )
                elif metric == "dr":
                    tuples = [deviation_ranks[game][mech][model] for model in models
                              if deviation_ranks[game][mech][model] is not None]
                    metric_averages[metric] = compute_summary_statistic(
                        tuples, precision, show_stderr, is_rank=True
                    )
        else:
            # Missing data - use "Unav." for all metrics
            for metric in mech_metric_list:
                metric_averages[metric] = "Unav."

        metric_values = [metric_averages[m] for m in mech_metric_list]
        if metric_values:
            row_parts.append(" & ".join(metric_values))

    lines.append(" & ".join(row_parts) + " \\\\")
    lines.append(r"\midrule")

    # Data rows (one per model)
    for model in models:
        row_parts = [simplify_model_name(model)]
        for mech in mechanisms:
            mech_metric_list = mech_metrics[mech]

            # Safely access nested dictionaries - use "Unav." for missing data
            if game in payoffs and mech in payoffs[game]:
                payoff_tuple = payoffs[game][mech][model]
                rd_tuple = rd_fitness[game][mech][model]
                dr_tuple = deviation_ranks[game][mech][model]

                metric_data = {}

                # Mean payoff
                if show_stderr:
                    metric_data["mean"] = format_score_with_stderr(payoff_tuple, precision)
                else:
                    payoff_mean, _ = payoff_tuple
                    metric_data["mean"] = format_score(payoff_mean, precision)

                # RD fitness
                if rd_tuple is not None:
                    if show_stderr:
                        metric_data["rd"] = format_score_with_stderr(rd_tuple, precision)
                    else:
                        rd_mean, _ = rd_tuple
                        metric_data["rd"] = format_score(rd_mean, precision)
                else:
                    metric_data["rd"] = "N/A"

                # Deviation ranks
                if dr_tuple is not None:
                    dr_mean, dr_stderr = dr_tuple
                    if show_stderr:
                        metric_data["dr"] = f"{dr_mean:.1f} $\\pm$ {dr_stderr:.1f}"
                    else:
                        metric_data["dr"] = f"{dr_mean:.1f}"
                else:
                    metric_data["dr"] = "N/A"
            else:
                # Missing data - use "Unav." for all metrics
                metric_data = {
                    "mean": "Unav.",
                    "rd": "Unav.",
                    "dr": "Unav.",
                }

            # Extract only metrics this mechanism supports
            metric_values = [metric_data[m] for m in mech_metric_list]
            if metric_values:
                row_parts.append(" & ".join(metric_values))

        lines.append(" & ".join(row_parts) + " \\\\")

    # Table footer
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def compute_aggregate_metric(
    data: Dict[str, Dict[str, Dict[str, Tuple[float, float] | None]]],
    game_configs: Dict[str, dict],
    mechanism: str,
    model: str,
    metric_type: str,
) -> Optional[Tuple[float, float]]:
    """
    Unified function to compute aggregate metrics across social dilemmas.

    Handles two metric types:
    - "numeric": For payoffs and RD fitness (returns mean ± stderr, with normalization)
    - "rank": For deviation ranks (returns mean ± stderr, no normalization)

    Args:
        data: Nested dictionary with (mean, stderr) tuples for each metric
        game_configs: Game configuration dictionaries
        mechanism: Mechanism name
        model: Model name
        metric_type: Type of metric ("numeric" or "rank")

    Returns:
        Tuple of (mean, stderr) or None if no data
    """
    social_dilemmas = {
        "PrisonersDilemma",
        "PublicGoods",
        "TravellersDilemma",
        "TrustGame",
    }

    # Precompute normalizers for each game (efficient for repeated calls)
    normalizers = {}
    if metric_type == "numeric":
        for game in data:
            if game in social_dilemmas:
                normalizers[game] = NormalizeScore(game, game_configs[game])

    values = []
    for game in data:
        if game not in social_dilemmas:
            continue

        # Skip if this game-mechanism combination doesn't exist
        if mechanism not in data[game]:
            continue

        value_tuple = data[game][mechanism][model]

        # Skip None values (RD/DR for reputation mechanism)
        if value_tuple is None:
            continue

        # Extract mean from (mean, stderr) tuple (ignore within-game stderr)
        mean_value, _ = value_tuple

        if metric_type == "numeric":
            # Apply normalization using precomputed normalizer
            normalized_value = normalizers[game].normalize(mean_value)
            values.append(normalized_value)
        else:
            assert metric_type == "rank"
            # Use rank mean directly
            values.append(mean_value)

    # If no values collected (e.g., all None for reputation RD/DR), return None
    if not values:
        return None

    # Compute mean and stderr for cross-game variation
    return compute_mean_stderr(values)


def generate_aggregate_table(
    mechanisms: List[str],
    models: List[str],
    payoffs: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    rd_fitness: Dict[str, Dict[str, Dict[str, Optional[Tuple[float, float]]]]],
    deviation_ranks: Dict[str, Dict[str, Dict[str, Optional[Tuple[float, float]]]]],
    game_configs: Dict[str, dict],
    precision: int,
    metrics: List[str],
    show_stderr: bool,
    source_folders: List[Path] | None = None,
) -> str:
    """
    Generate aggregated LaTeX table across social dilemma games with selected metrics.

    Args:
        mechanisms: List of mechanism names (columns)
        models: List of model names (rows)
        payoffs: Nested dictionary with payoff scores
        rd_fitness: Nested dictionary with RD fitness values
        deviation_ranks: Nested dictionary with deviation ranks
        game_configs: Game configuration dictionaries for normalization
        precision: Number of decimal places
        metrics: List of metrics to include (subset of ["mean", "rd", "dr"])
        show_stderr: Whether to show standard errors for mean/rd metrics
        source_folders: Optional list of source folder paths

    Returns:
        Complete LaTeX table as string
    """
    # Compute only requested aggregate metrics
    aggregate_payoffs = {}
    aggregate_rd = {}
    aggregate_dr = {}
    for mech in mechanisms:
        aggregate_payoffs[mech] = {}
        aggregate_rd[mech] = {}
        aggregate_dr[mech] = {}
        for model in models:
            if "mean" in metrics:
                aggregate_payoffs[mech][model] = compute_aggregate_metric(
                    payoffs, game_configs, mech, model, metric_type="numeric"
                )
            if "rd" in metrics:
                aggregate_rd[mech][model] = compute_aggregate_metric(
                    rd_fitness, game_configs, mech, model, metric_type="numeric"
                )
            if "dr" in metrics:
                aggregate_dr[mech][model] = compute_aggregate_metric(
                    deviation_ranks, game_configs, mech, model, metric_type="rank"
                )

    # Start building table
    lines = []

    # Add source folder comments if provided
    if source_folders:
        lines.append("% Source folders:")
        for folder in source_folders:
            lines.append(f"%   {folder}")

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Aggregate Results Across Social Dilemmas (Normalized)}"
    )
    lines.append(r"\label{tab:aggregate_results}")

    # Determine which metrics each mechanism supports
    mech_metrics = {}
    for mech in mechanisms:
        is_reputation = is_reputation_mechanism(mech)
        if is_reputation:
            # Reputation only supports mean
            mech_metrics[mech] = ["mean"] if "mean" in metrics else []
        else:
            # All other mechanisms support all requested metrics
            mech_metrics[mech] = metrics

    # Table header with vertical bars
    # Each mechanism has different number of sub-columns based on supported metrics
    col_spec_parts = ["l"]
    for mech in mechanisms:
        num_cols = len(mech_metrics[mech])
        if num_cols > 0:
            col_spec_parts.append("r" * num_cols)
    col_spec = "|".join(col_spec_parts)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # First header row: Mechanism names with multicolumn
    header_parts = ["\\textbf{Model}"]
    for mech in mechanisms:
        num_cols = len(mech_metrics[mech])
        if num_cols > 1:
            header_parts.append(f"\\multicolumn{{{num_cols}}}{{c}}{{\\textbf{{{mech}}}}}")
        elif num_cols == 1:
            header_parts.append(f"\\textbf{{{mech}}}")
    lines.append(" & ".join(header_parts) + " \\\\")

    # Second header row: Sub-column labels (only if needed)
    show_subheaders = any(len(mech_metrics[mech]) > 1 for mech in mechanisms)
    if show_subheaders:
        subheader_parts = [""]  # Empty for model column
        for mech in mechanisms:
            mech_metric_list = mech_metrics[mech]
            if len(mech_metric_list) > 1:
                metric_cols = " & ".join([f"\\textbf{{{METRIC_LABELS[m]}}}" for m in mech_metric_list])
                subheader_parts.append(metric_cols)
            elif len(mech_metric_list) == 1:
                # Single column - show the metric label
                subheader_parts.append(f"\\textbf{{{METRIC_LABELS[mech_metric_list[0]]}}}")
        lines.append(" & ".join(subheader_parts) + " \\\\")

    lines.append(r"\midrule")

    # Summary row (average across all models)
    row_parts = ["\\textbf{Average}"]
    for mech in mechanisms:
        is_reputation = is_reputation_mechanism(mech)
        mech_metric_list = mech_metrics[mech]
        metric_averages = {}

        # Calculate average for each metric this mechanism supports
        for metric in mech_metric_list:
            if metric == "mean":
                tuples = [aggregate_payoffs[mech][model] for model in models]
                metric_averages[metric] = compute_summary_statistic(
                    tuples, precision, show_stderr, is_rank=False
                )
            elif metric == "rd":
                tuples = [aggregate_rd[mech][model] for model in models
                          if aggregate_rd[mech][model] is not None]
                metric_averages[metric] = compute_summary_statistic(
                    tuples, precision, show_stderr, is_rank=False
                )
            elif metric == "dr":
                tuples = [aggregate_dr[mech][model] for model in models
                          if aggregate_dr[mech][model] is not None]
                metric_averages[metric] = compute_summary_statistic(
                    tuples, precision, show_stderr, is_rank=True
                )

        metric_values = [metric_averages[m] for m in mech_metric_list]
        if metric_values:
            row_parts.append(" & ".join(metric_values))

    lines.append(" & ".join(row_parts) + " \\\\")
    lines.append(r"\midrule")

    # Data rows (one per model)
    for model in models:
        row_parts = [simplify_model_name(model)]
        for mech in mechanisms:
            is_reputation = is_reputation_mechanism(mech)
            mech_metric_list = mech_metrics[mech]
            metric_data = {}

            for metric in mech_metric_list:
                if metric == "mean":
                    payoff_result = aggregate_payoffs[mech][model]
                    if show_stderr:
                        metric_data["mean"] = format_score_with_stderr(payoff_result, precision)
                    else:
                        metric_data["mean"] = format_score(payoff_result[0], precision)

                elif metric == "rd":
                    rd_result = aggregate_rd[mech][model]
                    if rd_result is not None:
                        if show_stderr:
                            metric_data["rd"] = format_score_with_stderr(rd_result, precision)
                        else:
                            metric_data["rd"] = format_score(rd_result[0], precision)
                    else:
                        metric_data["rd"] = "N/A"

                elif metric == "dr":
                    dr_result = aggregate_dr[mech][model]
                    if dr_result is not None:
                        if show_stderr:
                            dr_mean, dr_stderr = dr_result
                            metric_data["dr"] = f"{dr_mean:.1f} $\\pm$ {dr_stderr:.1f}"
                        else:
                            dr_mean, _ = dr_result
                            metric_data["dr"] = f"{dr_mean:.1f}"
                    else:
                        metric_data["dr"] = "N/A"

            # Extract only metrics this mechanism supports
            metric_values = [metric_data[m] for m in mech_metric_list]
            if metric_values:
                row_parts.append(" & ".join(metric_values))

        lines.append(" & ".join(row_parts) + " \\\\")

    # Table footer
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def save_table(table_latex: str, output_path: Path) -> None:
    """Save LaTeX table to .tex file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(table_latex)
        f.write("\n")


def print_table(table_latex: str, title: str) -> None:
    """Print table to stdout with formatting."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    print(table_latex)
    print()


def main() -> None:
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from batch experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/visualize/create_table.py outputs/2026/01/14/16:14/
  python src/visualize/create_table.py outputs/2026/01/14/16:14/ --output tables/
  python src/visualize/create_table.py outputs/2026/01/14/16:14/ --quiet
  python src/visualize/create_table.py outputs/folder1/ outputs/folder2/ --output tables/
        """,
    )

    parser.add_argument(
        "batch_paths",
        type=Path,
        nargs="+",
        help="Path(s) to batch experiment folder(s) (e.g., outputs/2026/01/14/16:14/)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for .tex files (default: same as batch_path)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Don't print tables to stdout, only save files",
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Number of decimal places for scores (default: 3)",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["mean", "rd", "dr"],
        default=["mean", "rd", "dr"],
        help="Metrics to include in tables (default: all three)",
    )

    parser.add_argument(
        "--no-stderr",
        dest="show_stderr",
        action="store_false",
        default=True,
        help="Hide standard errors in aggregate table (default: show stderr)",
    )

    parser.add_argument(
        "--show-stderr-games",
        action="store_true",
        default=False,
        help="Show standard errors in per-game tables (from variation across runs)",
    )

    args = parser.parse_args()

    # Parse all batch folders and group
    all_grouped_experiments: Dict[Tuple[str, str], List[ExperimentData]] = defaultdict(list)
    all_failed = []

    for batch_path in args.batch_paths:
        try:
            grouped, failed = parse_batch_folder(batch_path, args.metrics)
            all_failed.extend(failed)

            # Merge groups
            for group_key, experiments in grouped.items():
                all_grouped_experiments[group_key].extend(experiments)

            print(f"Parsed {sum(len(exps) for exps in grouped.values())} experiments from {batch_path}")
        except (FileNotFoundError, NotADirectoryError) as e:
            print(f"Error parsing {batch_path}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\nTotal: {sum(len(exps) for exps in all_grouped_experiments.values())} successful, {len(all_failed)} failed")
    print(f"Grouped into {len(all_grouped_experiments)} game-mechanism combinations")

    # Validate experiment groups
    try:
        mechanisms, games, models, eval_config = validate_experiment_groups(all_grouped_experiments)
    except (ValueError, AssertionError) as e:
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Validation passed: {len(games)} games, {len(mechanisms)} mechanisms, {len(models)} models\n")

    # Build data structures
    payoffs, rd_fitness, deviation_ranks, game_configs = build_data_structure(
        all_grouped_experiments, models, args.metrics
    )

    # Determine output directory
    if args.output:
        output_dir = args.output
    elif len(args.batch_paths) == 1:
        output_dir = args.batch_paths[0]
    else:
        output_dir = args.batch_paths[0].parent

    # Store all table LaTeX for combined file
    all_tables = []

    # Generate and save aggregate table FIRST (social dilemmas only)
    aggregate_table_latex = generate_aggregate_table(
        mechanisms, models, payoffs, rd_fitness, deviation_ranks, game_configs, args.precision, args.metrics, args.show_stderr, args.batch_paths
    )

    output_path = output_dir / "table_aggregate.tex"
    save_table(aggregate_table_latex, output_path)

    if not args.quiet:
        print_table(aggregate_table_latex, "Aggregate Table")

    print(f"Saved: {output_path}")

    # Add aggregate table to combined list FIRST
    all_tables.append(aggregate_table_latex)

    # Generate and save per-game tables
    for game in games:
        table_latex = generate_game_table(
            game, mechanisms, models, payoffs, rd_fitness, deviation_ranks, args.precision, args.metrics, args.batch_paths, show_stderr=args.show_stderr_games
        )

        output_path = output_dir / f"table_{game}.tex"
        save_table(table_latex, output_path)

        if not args.quiet:
            print_table(table_latex, f"Table for {game}")

        print(f"Saved: {output_path}")

        # Add to combined list
        all_tables.append(table_latex)

    # Generate and save combined table file
    combined_latex = "\n\n".join(all_tables)
    combined_output_path = output_dir / "table_all_combined.tex"
    save_table(combined_latex, combined_output_path)
    print(f"Saved combined table: {combined_output_path}")

    print(f"\nTotal tables generated: {len(games) + 1} (plus 1 combined file)")

    # Print summary of failed experiments
    if all_failed:
        print(f"\n{'='*80}")
        print(f"Failed to parse {len(all_failed)} experiment(s):")
        for folder, error in all_failed:
            print(f"  - {folder}")
            print(f"    Error: {type(error).__name__}: {error}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
