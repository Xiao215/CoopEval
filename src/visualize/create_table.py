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
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    # New fields for additional metrics
    rd_fitness: Optional[Dict[str, float]] = None  # Required (None only for reputation)
    deviation_ranks: Optional[Dict[str, str]] = None  # Required (None only for reputation), store as rank strings


def load_json(path: Path) -> Optional[dict]:
    """Load JSON file with error handling."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def parse_batch_folder(batch_path: Path) -> List[ExperimentData]:
    """
    Parse batch folder and extract experiment data.

    Args:
        batch_path: Path to batch experiment folder

    Returns:
        List of ExperimentData objects

    Raises:
        ValueError: If no valid experiments found
    """
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch folder not found: {batch_path}")
    if not batch_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {batch_path}")

    experiments = []

    # Scan for subdirectories (skip configs and other non-experiment folders)
    subdirs = [d for d in batch_path.iterdir() if d.is_dir() and d.name != "configs"]

    for subdir in subdirs:
        # Check for required files
        config_path = subdir / "config.json"
        payoff_path = subdir / "model_average_payoff.json"

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

        # Load metrics (REQUIRED for non-reputation mechanisms)
        if mechanism.lower() == "reputation":
            # Reputation mechanism: explicitly use None (will display N/A)
            rd_fitness = None
            deviation_ranks = None
        else:
            # All other mechanisms: REQUIRE these files to exist
            rd_fitness_path = subdir / "replicator_dynamics_fitness.json"
            if not rd_fitness_path.exists():
                raise FileNotFoundError(
                    f"Missing replicator_dynamics_fitness.json in {subdir} for mechanism {mechanism}"
                )

            deviation_ratings_path = subdir / "deviation_ratings.json"
            if not deviation_ratings_path.exists():
                raise FileNotFoundError(
                    f"Missing deviation_ratings.json in {subdir} for mechanism {mechanism}"
                )

            # Load RD fitness file and extract fitness values
            rd_fitness_data = load_json(rd_fitness_path)
            if rd_fitness_data is None:
                raise ValueError(f"Failed to parse replicator_dynamics_fitness.json in {subdir}")
            rd_fitness = {
                model: data["fitness"]
                for model, data in rd_fitness_data.items()
            }

            # Load deviation ratings and convert to ranks
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
                rd_fitness=rd_fitness,
                deviation_ranks=deviation_ranks,
            )
        )

    if not experiments:
        raise ValueError(f"No valid experiments found in {batch_path}")

    return experiments


def normalize_score(
    game: str, score: float, game_config: dict
) -> float:
    """
    Normalize score from NE payoff to cooperative payoff for social dilemmas.

    Args:
        game: Game name
        score: Raw score from the game
        game_config: Game configuration dict

    Returns:
        Normalized score scaled to [0, 1] where 0 = NE, 1 = cooperative
    """

    if game == "PrisonersDilemma":
        payoff_matrix = game_config["kwargs"]["payoff_matrix"]
        ne_payoff = payoff_matrix["DD"][0]  # NE: both defect
        coop_payoff = payoff_matrix["CC"][1]  # Cooperative: both cooperate
        normalized = (score - ne_payoff) / (coop_payoff - ne_payoff)
        return normalized

    elif game == "PublicGoods":
        coop_payoff = game_config["kwargs"]["multiplier"]
        ne_payoff = 1  # NE: no one contributes
        normalized = (score - ne_payoff) / (coop_payoff - ne_payoff)
        return normalized

    elif game == "TravellersDilemma":
        ne_payoff = game_config["kwargs"]["min_claim"]
        spacing = game_config["kwargs"]["claim_spacing"]
        num_actions = game_config["kwargs"]["num_actions"]
        coop_payoff = ne_payoff + spacing * (num_actions - 1)
        normalized = (score - ne_payoff) / (coop_payoff - ne_payoff)
        return normalized

    elif game == "TrustGame":
        payoff_matrix = game_config["kwargs"]["payoff_matrix"]
        ne_payoff = payoff_matrix["KK"][0]  # NE: both keep
        coop_payoff = payoff_matrix["GG"][0]  # Cooperative: both give
        normalized = (score - ne_payoff) / (coop_payoff - ne_payoff)
        return normalized

    else:
        # For other games, return score as-is
        return score


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
        # Reputation mechanism doesn't have RD/DR metrics
        if exp.mechanism.lower() == "reputation":
            for model in exp.model_scores.keys():
                rd_fitness[exp.game][exp.mechanism][model] = None
                deviation_ranks[exp.game][exp.mechanism][model] = None
        else:
            # Non-reputation mechanisms: values must exist for all models
            for model in exp.model_scores.keys():
                rd_fitness[exp.game][exp.mechanism][model] = exp.rd_fitness[model]
                deviation_ranks[exp.game][exp.mechanism][model] = exp.deviation_ranks[model]

    return payoffs, rd_fitness, deviation_ranks, game_configs


def sort_mechanisms(mechanisms: List[str]) -> List[str]:
    """
    Sort mechanisms in the preferred order.

    Args:
        mechanisms: List of mechanism names

    Returns:
        Sorted list of mechanism names
    """
    # Define preferred order (case-insensitive matching)
    preferred_order = [
        "NoMechanism",
        "Repetition",
        "Reputation",
        "Disarmament",
        "Mediation",
        "Contracting"
    ]

    # Create a mapping for case-insensitive lookup
    order_map = {name.lower(): i for i, name in enumerate(preferred_order)}

    # Sort mechanisms by preferred order, alphabetically for any not in the list
    def sort_key(mech):
        mech_lower = mech.lower()
        if mech_lower in order_map:
            return (0, order_map[mech_lower])
        else:
            return (1, mech)  # Unknown mechanisms go last, sorted alphabetically

    return sorted(mechanisms, key=sort_key)


def extract_canonical_lists(
    experiments: List[ExperimentData],
) -> tuple[List[str], List[str], List[str]]:
    """
    Extract canonical lists of mechanisms, games, and models from experiments.

    Args:
        experiments: List of ExperimentData objects

    Returns:
        Tuple of (mechanisms, games, models) with mechanisms in preferred order,
        games and models sorted alphabetically
    """
    mechanisms = set()
    games = set()
    models = set()

    for exp in experiments:
        mechanisms.add(exp.mechanism)
        games.add(exp.game)
        models.update(exp.model_scores.keys())

    return sort_mechanisms(list(mechanisms)), sorted(games), sorted(models)


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

    Raises:
        ValueError: If any combinations are missing
    """
    missing_combinations = []

    for game in canonical_games:
        if game not in data:
            for mechanism in canonical_mechanisms:
                for model in canonical_models:
                    missing_combinations.append((game, mechanism, model))
            continue

        for mechanism in canonical_mechanisms:
            if mechanism not in data[game]:
                for model in canonical_models:
                    missing_combinations.append((game, mechanism, model))
                continue

            for model in canonical_models:
                if model not in data[game][mechanism]:
                    missing_combinations.append((game, mechanism, model))

    if missing_combinations:
        error_msg = f"Data inconsistency detected! Missing {len(missing_combinations)} combinations:\n"
        # Show first 10 missing combinations
        for game, mechanism, model in missing_combinations[:10]:
            error_msg += f"  - {game} × {mechanism} × {model}\n"
        if len(missing_combinations) > 10:
            error_msg += f"  ... and {len(missing_combinations) - 10} more\n"
        error_msg += "\nAll mechanism×game×model combinations must be present in batch experiments."
        raise ValueError(error_msg)


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
) -> str:
    """
    Generate LaTeX table for a single game with multicolumn headers.

    Args:
        game: Game name
        mechanisms: List of mechanism names (rows)
        models: List of model names (columns)
        payoffs: Nested dictionary with payoff scores
        rd_fitness: Nested dictionary with RD fitness values
        deviation_ranks: Nested dictionary with deviation ranks
        precision: Number of decimal places
        metrics: List of metrics to include (subset of ["mean", "rd", "dr"])

    Returns:
        Complete LaTeX table as string
    """
    if metrics is None:
        metrics = ["mean", "rd", "dr"]

    # Start building table
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Results for {game}}}")
    game_slug = game.lower().replace(" ", "_")
    lines.append(f"\\label{{tab:{game_slug}}}")

    # Table header with vertical bars
    # Each model has len(metrics) sub-columns
    num_models = len(models)
    num_metrics = len(metrics)
    col_spec = "l|" + "|".join(["r" * num_metrics] * num_models)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # First header row: Model names with multicolumn
    header_parts = ["\\textbf{Mechanism}"]
    for model in models:
        header_parts.append(f"\\multicolumn{{{num_metrics}}}{{c}}{{\\textbf{{{model}}}}}")
    lines.append(" & ".join(header_parts) + " \\\\")

    # Second header row: Sub-column labels
    subheader_parts = [""]  # Empty for mechanism column
    for _ in models:
        metric_cols = " & ".join([f"\\textbf{{{METRIC_LABELS[m]}}}" for m in metrics])
        subheader_parts.append(metric_cols)
    lines.append(" & ".join(subheader_parts) + " \\\\")

    lines.append(r"\midrule")

    # Data rows
    for mech in mechanisms:
        row_parts = [mech]
        for model in models:
            payoff_score = payoffs[game][mech][model]
            rd_val = rd_fitness[game][mech][model]
            dr_val = deviation_ranks[game][mech][model]
            
            metric_data = {
                "mean": format_score(payoff_score, precision),
                "rd": format_score(rd_val, precision) if rd_val is not None else "N/A",
                "dr": dr_val if dr_val is not None else "N/A",
            }

            # Extract only selected metrics
            metric_values = [metric_data[m] for m in metrics]
            row_parts.append(" & ".join(metric_values))

        lines.append(" & ".join(row_parts) + " \\\\")

    # Table footer
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def compute_aggregate_metric(
    data: Dict[str, Dict[str, Dict[str, any]]],
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

    values = []
    for game in data:
        if game not in social_dilemmas:
            continue

        value = data[game][mechanism][model]
        
        # Skip None values (RD/DR for reputation mechanism)
        if value is None:
            continue

        if metric_type == "numeric":
            # Apply normalization for numeric metrics
            game_config = game_configs[game]
            normalized_value = normalize_score(game, value, game_config)
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
) -> str:
    """
    Generate aggregated LaTeX table across social dilemma games with selected metrics.

    Args:
        mechanisms: List of mechanism names (rows)
        models: List of model names (columns)
        payoffs: Nested dictionary with payoff scores
        rd_fitness: Nested dictionary with RD fitness values
        deviation_ranks: Nested dictionary with deviation ranks
        game_configs: Game configuration dictionaries for normalization
        precision: Number of decimal places
        metrics: List of metrics to include (subset of ["mean", "rd", "dr"])
        show_stderr: Whether to show standard errors for mean/rd metrics

    Returns:
        Complete LaTeX table as string
    """
    # Compute all aggregate metrics in one pass
    aggregate_payoffs = {}
    aggregate_rd = {}
    aggregate_dr = {}
    for mech in mechanisms:
        aggregate_payoffs[mech] = {}
        aggregate_rd[mech] = {}
        aggregate_dr[mech] = {}
        for model in models:
            aggregate_payoffs[mech][model] = compute_aggregate_metric(
                payoffs, game_configs, mech, model, metric_type="numeric"
            )
            aggregate_rd[mech][model] = compute_aggregate_metric(
                rd_fitness, game_configs, mech, model, metric_type="numeric"
            )
            aggregate_dr[mech][model] = compute_aggregate_metric(
                deviation_ranks, game_configs, mech, model, metric_type="rank"
            )

    # Start building table
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Aggregate Results Across Social Dilemmas (Normalized)}"
    )
    lines.append(r"\label{tab:aggregate_results}")

    # Table header with vertical bars
    num_models = len(models)
    num_metrics = len(metrics)
    col_spec = "l|" + "|".join(["r" * num_metrics] * num_models)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # First header row: Model names with multicolumn
    header_parts = ["\\textbf{Mechanism}"]
    for model in models:
        header_parts.append(f"\\multicolumn{{{num_metrics}}}{{c}}{{\\textbf{{{model}}}}}")
    lines.append(" & ".join(header_parts) + " \\\\")

    # Second header row: Sub-column labels
    subheader_parts = [""]
    for _ in models:
        metric_cols = " & ".join([f"\\textbf{{{METRIC_LABELS[m]}}}" for m in metrics])
        subheader_parts.append(metric_cols)
    lines.append(" & ".join(subheader_parts) + " \\\\")

    lines.append(r"\midrule")

    # Data rows
    for mech in mechanisms:
        row_parts = [mech]
        for model in models:
            payoff_result = aggregate_payoffs[mech][model]
            rd_result = aggregate_rd[mech][model]
            dr_val = aggregate_dr[mech][model]

            if show_stderr:
                metric_data = {
                    "mean": format_score_with_stderr(payoff_result, precision),
                    "rd": format_score_with_stderr(rd_result, precision) if rd_result is not None else "N/A",
                    "dr": dr_val if dr_val is not None else "N/A",
                }
            else:
                metric_data = {
                    "mean": format_score(payoff_result[0], precision) if payoff_result is not None else "N/A",
                    "rd": format_score(rd_result[0], precision) if rd_result is not None else "N/A",
                    "dr": dr_val if dr_val is not None else "N/A",
                }

            # Extract only selected metrics
            metric_values = [metric_data[m] for m in metrics]
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
        """,
    )

    parser.add_argument(
        "batch_path",
        type=Path,
        help="Path to batch experiment folder (e.g., outputs/2026/01/14/16:14/)",
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

    # Parse batch folder
    try:
        experiments = parse_batch_folder(args.batch_path)
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(experiments)} valid experiments")

    # Extract canonical lists (mechanism, game, model combinations)
    mechanisms, games, models = extract_canonical_lists(experiments)
    print(f"Games: {len(games)}, Mechanisms: {len(mechanisms)}, Models: {len(models)}")

    # Build data structures (NOW WITH NEW METRICS)
    payoffs, rd_fitness, deviation_ranks, game_configs = build_data_structure(experiments)

    # Validate all combinations are present
    validate_data_consistency(payoffs, mechanisms, games, models)
    print("Data consistency validated: all mechanism×game×model combinations present")

    # Determine output directory
    output_dir = args.output if args.output else args.batch_path

    # Generate and save per-game tables
    for game in games:
        table_latex = generate_game_table(
            game, mechanisms, models, payoffs, rd_fitness, deviation_ranks, args.precision, args.metrics
        )

        output_path = output_dir / f"table_{game}.tex"
        save_table(table_latex, output_path)

        if not args.quiet:
            print_table(table_latex, f"Table for {game}")

        print(f"Saved: {output_path}")

    # Generate and save aggregate table (social dilemmas only)
    table_latex = generate_aggregate_table(
        mechanisms, models, payoffs, rd_fitness, deviation_ranks, game_configs, args.precision, args.metrics, args.show_stderr
    )

    output_path = output_dir / "table_aggregate.tex"
    save_table(table_latex, output_path)

    if not args.quiet:
        print_table(table_latex, "Aggregate Table")

    print(f"Saved: {output_path}")

    print(f"\nTotal tables generated: {len(games) + 1}")


if __name__ == "__main__":
    main()
