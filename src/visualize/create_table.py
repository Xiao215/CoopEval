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
)

# Map metric names to LaTeX display labels
METRIC_LABELS = {
    "mean": "Mean",
    "rd": "RD",
    "dr": "DR",
}


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


def parse_batch_folder(batch_path: Path, metrics: List[str]) -> tuple[List[ExperimentData], List[tuple[Path, Exception]]]:
    """
    Parse batch folder and extract experiment data.

    Args:
        batch_path: Path to batch experiment folder
        metrics: List of metrics to load (subset of ["mean", "rd", "dr"])

    Returns:
        Tuple of (list of ExperimentData objects, list of (failed_path, error) tuples)

    Raises:
        ValueError: If no valid experiments found
    """
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch folder not found: {batch_path}")
    if not batch_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {batch_path}")

    experiments = []
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
            if mechanism.lower() == "reputation":
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

            # Create experiment data with new fields
            experiments.append(
                ExperimentData(
                    mechanism=mechanism,
                    game=game,
                    model_scores=payoffs,
                    folder_path=subdir,
                    game_config=game_config,
                    eval_config=eval_config,
                    rd_fitness=rd_fitness,
                    deviation_ranks=deviation_ranks,
                )
            )

        except Exception as e:
            print(f"WARNING: Failed to parse experiment in {subdir}")
            print(f"  Error: {type(e).__name__}: {e}")
            failed_experiments.append((subdir, e))

    if not experiments:
        raise ValueError(f"No valid experiments found in {batch_path}")

    return experiments, failed_experiments


def build_data_structure(
    experiments: List[ExperimentData],
) -> tuple[
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, Dict[str, Dict[str, Optional[float]]]],
    Dict[str, Dict[str, Dict[str, Optional[str]]]],
    Dict[str, dict]
]:
    """
    Build nested data structures from experiments.

    Args:
        experiments: List of ExperimentData objects

    Returns:
        Tuple of:
        - Nested dict: payoffs[game][mechanism][model] = score
        - Nested dict: rd_fitness[game][mechanism][model] = fitness or None
        - Nested dict: deviation_ranks[game][mechanism][model] = rank_str or None
        - Game configs: game_configs[game] = config dict
    """
    payoffs: Dict[str, Dict[str, Dict[str, float]]] = {}
    rd_fitness: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    deviation_ranks: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
    game_configs: Dict[str, dict] = {}

    for exp in experiments:
        if exp.game not in payoffs:
            payoffs[exp.game] = {}
            rd_fitness[exp.game] = {}
            deviation_ranks[exp.game] = {}
            game_configs[exp.game] = exp.game_config

        if exp.mechanism not in payoffs[exp.game]:
            payoffs[exp.game][exp.mechanism] = {}
            rd_fitness[exp.game][exp.mechanism] = {}
            deviation_ranks[exp.game][exp.mechanism] = {}

        # Update payoffs
        payoffs[exp.game][exp.mechanism].update(exp.model_scores)

        # Update rd_fitness and deviation_ranks
        for model in exp.model_scores.keys():
            rd_fitness[exp.game][exp.mechanism][model] = exp.rd_fitness[model] if exp.rd_fitness else None
            deviation_ranks[exp.game][exp.mechanism][model] = exp.deviation_ranks[model] if exp.deviation_ranks else None

    return payoffs, rd_fitness, deviation_ranks, game_configs


def validate_experiments(
    experiments: List[ExperimentData],
) -> tuple[List[str], List[str], List[str], dict]:
    """
    Validate experiments and extract canonical lists.

    Checks:
    1. No duplicate (mechanism, game) combinations
    2. All experiments have identical model lists
    3. All experiments have identical evaluation configs
    4. Complete mechanism×game grid coverage

    Args:
        experiments: List of ExperimentData objects

    Returns:
        Tuple of (mechanisms, games, models, eval_config)

    Raises:
        ValueError: If validation fails
    """
    # Check for duplicate (mechanism, game) combinations
    exp_map: Dict[tuple[str, str], ExperimentData] = {}
    duplicates = []
    for exp in experiments:
        key = (exp.mechanism, exp.game)
        if key in exp_map:
            duplicates.append((
                key,
                exp_map[key].folder_path,
                exp.folder_path
            ))
        else:
            exp_map[key] = exp

    if duplicates:
        error_msg = f"Duplicate experiments detected! Found {len(duplicates)} duplicate(s):\n"
        for (mech, game), path1, path2 in duplicates:
            error_msg += f"  - {mech} × {game}: {path1} and {path2}\n"
        error_msg += "\nEach (mechanism, game) combination must appear exactly once across all folders."
        raise ValueError(error_msg)

    # Validate non-empty experiments
    if not experiments:
        raise ValueError("No experiments provided")

    # Extract canonical sets and validate consistency on the fly
    mechanisms = set()
    games = set()
    canonical_models = sorted(experiments[0].model_scores.keys())
    canonical_eval = experiments[0].eval_config

    for i, exp in enumerate(experiments):
        mechanisms.add(exp.mechanism)
        games.add(exp.game)
        
        # Check model list matches
        exp_models = sorted(exp.model_scores.keys())
        if exp_models != canonical_models:
            error_msg = f"Model list mismatch detected!\n"
            error_msg += f"  Experiment 0: {canonical_models}\n"
            error_msg += f"  Experiment {i}: {exp_models}\n"
            error_msg += "\nAll experiments must test the same set of models."
            raise ValueError(error_msg)
        
        # Check eval config matches
        if exp.eval_config != canonical_eval:
            error_msg = f"Evaluation config mismatch detected!\n"
            error_msg += f"  Experiment 0: {canonical_eval}\n"
            error_msg += f"  Experiment {i}: {exp.eval_config}\n"
            error_msg += "\nAll experiments must use identical evaluation configurations."
            raise ValueError(error_msg)

    # Sort canonical lists
    canonical_mechanisms = sort_mechanisms(list(mechanisms))
    canonical_games = sort_games(list(games))
    canonical_models = sort_models(canonical_models)

    # Check for complete grid coverage (warn if incomplete)
    missing_combinations = []
    for mech in canonical_mechanisms:
        for game in canonical_games:
            if (mech, game) not in exp_map:
                missing_combinations.append((mech, game))

    if missing_combinations:
        print(f"\nWARNING: Incomplete mechanism×game grid! Missing {len(missing_combinations)} combination(s):")
        for mech, game in missing_combinations:
            print(f"  - {mech} × {game}")
        print("Tables will show 'Unav.' for missing data.\n")

    return canonical_mechanisms, canonical_games, canonical_models, canonical_eval, missing_combinations


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


def generate_game_table(
    game: str,
    mechanisms: List[str],
    models: List[str],
    payoffs: Dict[str, Dict[str, Dict[str, float]]],
    rd_fitness: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    deviation_ranks: Dict[str, Dict[str, Dict[str, Optional[str]]]],
    precision: int = 3,
    metrics: List[str] | None = None,
    source_folders: List[Path] | None = None,
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
        is_reputation = mech.lower() == "reputation"
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
                    values = [payoffs[game][mech][model] for model in models]
                    avg = sum(values) / len(values)
                    metric_averages[metric] = format_score(avg, precision)
                elif metric == "rd":
                    values = [rd_fitness[game][mech][model] for model in models if rd_fitness[game][mech][model] is not None]
                    if values:
                        avg = sum(values) / len(values)
                        metric_averages[metric] = format_score(avg, precision)
                    else:
                        metric_averages[metric] = "N/A"
                elif metric == "dr":
                    values = [float(deviation_ranks[game][mech][model]) for model in models if deviation_ranks[game][mech][model] is not None]
                    if values:
                        avg = sum(values) / len(values)
                        metric_averages[metric] = f"{avg:.1f}"
                    else:
                        metric_averages[metric] = "N/A"
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
                payoff_score = payoffs[game][mech][model]
                rd_val = rd_fitness[game][mech][model]
                dr_val = deviation_ranks[game][mech][model]

                metric_data = {
                    "mean": format_score(payoff_score, precision),
                    "rd": format_score(rd_val, precision) if rd_val is not None else "N/A",
                    "dr": dr_val if dr_val is not None else "N/A",
                }
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
    data,
    game_configs: Dict[str, dict],
    mechanism: str,
    model: str,
    metric_type: str,
) -> Optional[Tuple[float, float]] | Optional[str]:
    """
    Unified function to compute aggregate metrics across social dilemmas.

    Handles three metric types:
    - "numeric": For payoffs and RD fitness (returns mean ± stderr, with normalization)
    - "rank": For deviation ranks (returns average rank as string, no normalization)

    Args:
        data: Nested dictionary with metric values
        game_configs: Game configuration dictionaries
        mechanism: Mechanism name
        model: Model name
        metric_type: Type of metric ("numeric" or "rank")

    Returns:
        For "numeric": Tuple of (mean, stderr) or None if no data
        For "rank": Average rank string or None if no data
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

        value = data[game][mechanism][model]

        # Skip None values (RD/DR for reputation mechanism)
        if value is None:
            continue

        if metric_type == "numeric":
            # Apply normalization using precomputed normalizer
            normalized_value = normalizers[game].normalize(value)
            values.append(normalized_value)
        else:
            assert metric_type == "rank"
            # Parse rank string to float
            rank_value = float(value)
            values.append(rank_value)

    # If no values collected (e.g., all None for reputation RD/DR), return None
    if not values:
        return None

    n = len(values)
    mean = sum(values) / n

    if metric_type == "rank":
        # Return formatted rank string
        return f"{mean:.1f}"

    # For numeric metrics, compute stderr
    if n == 1:
        stderr = 0.0
    else:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        stderr = math.sqrt(variance / n)

    return mean, stderr


def generate_aggregate_table(
    mechanisms: List[str],
    models: List[str],
    payoffs: Dict[str, Dict[str, Dict[str, float]]],
    rd_fitness: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    deviation_ranks: Dict[str, Dict[str, Dict[str, Optional[str]]]],
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
        is_reputation = mech.lower() == "reputation"
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
        is_reputation = mech.lower() == "reputation"
        mech_metric_list = mech_metrics[mech]
        metric_averages = {}

        # Calculate average for each metric this mechanism supports
        for metric in mech_metric_list:
            if metric == "mean":
                values = [aggregate_payoffs[mech][model][0] for model in models]
                avg = sum(values) / len(values)
                if len(values) > 1:
                    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
                    std = math.sqrt(variance)
                else:
                    std = 0.0

                if show_stderr:
                    metric_averages["mean"] = f"{avg:.{precision}f} $\\pm$ {std:.{precision}f}"
                else:
                    metric_averages["mean"] = f"{avg:.{precision}f}"

            elif metric == "rd":
                values = [aggregate_rd[mech][model][0] for model in models if aggregate_rd[mech][model] is not None]
                if values:
                    avg = sum(values) / len(values)
                    if len(values) > 1:
                        variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
                        std = math.sqrt(variance)
                    else:
                        std = 0.0

                    if show_stderr:
                        metric_averages["rd"] = f"{avg:.{precision}f} $\\pm$ {std:.{precision}f}"
                    else:
                        metric_averages["rd"] = f"{avg:.{precision}f}"
                else:
                    metric_averages["rd"] = "N/A"

            elif metric == "dr":
                values = [float(aggregate_dr[mech][model]) for model in models if aggregate_dr[mech][model] is not None]
                if values:
                    avg = sum(values) / len(values)
                    metric_averages["dr"] = f"{avg:.1f}"
                else:
                    metric_averages["dr"] = "N/A"

        metric_values = [metric_averages[m] for m in mech_metric_list]
        if metric_values:
            row_parts.append(" & ".join(metric_values))

    lines.append(" & ".join(row_parts) + " \\\\")
    lines.append(r"\midrule")

    # Data rows (one per model)
    for model in models:
        row_parts = [simplify_model_name(model)]
        for mech in mechanisms:
            is_reputation = mech.lower() == "reputation"
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
                    dr_val = aggregate_dr[mech][model]
                    metric_data["dr"] = dr_val if dr_val is not None else "N/A"

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

    args = parser.parse_args()

    # Parse all batch folders
    all_experiments = []
    all_failed = []
    for batch_path in args.batch_paths:
        try:
            experiments, failed_experiments = parse_batch_folder(batch_path, args.metrics)
            all_experiments.extend(experiments)
            all_failed.extend(failed_experiments)
            print(f"Parsed {len(experiments)} experiments from {batch_path} ({len(failed_experiments)} failed)")
        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            print(f"Error parsing {batch_path}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\nTotal experiments: {len(all_experiments)} successful, {len(all_failed)} failed")

    # Validate experiments and extract canonical lists
    try:
        mechanisms, games, models, eval_config, missing_combinations = validate_experiments(all_experiments)
    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        print(f"NOTE: This may be due to incomplete experiments (still running)")
        sys.exit(1)

    print(f"Validation passed: {len(games)} games, {len(mechanisms)} mechanisms, {len(models)} models")
    print(f"Grid completeness: {len(mechanisms)} × {len(games)} = {len(all_experiments)} experiments\n")

    # Build data structures
    payoffs, rd_fitness, deviation_ranks, game_configs = build_data_structure(all_experiments)

    # Determine output directory
    if args.output:
        output_dir = args.output
    elif len(args.batch_paths) == 1:
        output_dir = args.batch_paths[0]
    else:
        output_dir = args.batch_paths[0].parent

    # Store all table LaTeX for combined file
    all_tables = []

    # Generate and save per-game tables
    for game in games:
        table_latex = generate_game_table(
            game, mechanisms, models, payoffs, rd_fitness, deviation_ranks, args.precision, args.metrics, args.batch_paths
        )

        output_path = output_dir / f"table_{game}.tex"
        save_table(table_latex, output_path)

        if not args.quiet:
            print_table(table_latex, f"Table for {game}")

        print(f"Saved: {output_path}")

        # Add to combined list
        all_tables.append(table_latex)

    # Generate and save aggregate table (social dilemmas only)
    aggregate_table_latex = generate_aggregate_table(
        mechanisms, models, payoffs, rd_fitness, deviation_ranks, game_configs, args.precision, args.metrics, args.show_stderr, args.batch_paths
    )

    output_path = output_dir / "table_aggregate.tex"
    save_table(aggregate_table_latex, output_path)

    if not args.quiet:
        print_table(aggregate_table_latex, "Aggregate Table")

    print(f"Saved: {output_path}")

    # Add aggregate table to combined list
    all_tables.append(aggregate_table_latex)

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
