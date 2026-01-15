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


@dataclass
class ExperimentData:
    """Represents a single experiment's data."""

    mechanism: str
    game: str
    model_scores: Dict[str, float]
    folder_path: Path
    game_config: dict  # Store full game config for score normalization


def load_json(path: Path) -> Optional[dict]:
    """Load JSON file with error handling."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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

    # Scan for subdirectories
    subdirs = [d for d in batch_path.iterdir() if d.is_dir()]

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

        # Create experiment data
        experiments.append(
            ExperimentData(
                mechanism=mechanism,
                game=game,
                model_scores=payoffs,
                folder_path=subdir,
                game_config=game_config,
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
    Dict[str, Dict[str, Dict[str, float]]], Dict[str, dict]
]:
    """
    Build nested data structure from experiments.

    Args:
        experiments: List of ExperimentData objects

    Returns:
        Tuple of:
        - Nested dict: data[game][mechanism][model] = score
        - Game configs: game_configs[game] = config dict
    """
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    game_configs: Dict[str, dict] = {}

    for exp in experiments:
        if exp.game not in data:
            data[exp.game] = {}
            game_configs[exp.game] = exp.game_config

        if exp.mechanism not in data[exp.game]:
            data[exp.game][exp.mechanism] = {}

        data[exp.game][exp.mechanism].update(exp.model_scores)

    return data, game_configs


def extract_canonical_lists(
    experiments: List[ExperimentData],
) -> tuple[List[str], List[str], List[str]]:
    """
    Extract canonical lists of mechanisms, games, and models from experiments.

    Args:
        experiments: List of ExperimentData objects

    Returns:
        Tuple of (mechanisms, games, models) sorted alphabetically
    """
    mechanisms = set()
    games = set()
    models = set()

    for exp in experiments:
        mechanisms.add(exp.mechanism)
        games.add(exp.game)
        models.update(exp.model_scores.keys())

    return sorted(mechanisms), sorted(games), sorted(models)


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
    data: Dict[str, Dict[str, Dict[str, float]]],
    precision: int = 3,
) -> str:
    """
    Generate LaTeX table for a single game.

    Args:
        game: Game name
        mechanisms: List of mechanism names (rows)
        models: List of model names (columns)
        data: Nested dictionary with scores
        precision: Number of decimal places

    Returns:
        Complete LaTeX table as string
    """
    # Start building table
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Average Payoffs for {game}}}")
    game_slug = game.lower().replace(" ", "_")
    lines.append(f"\\label{{tab:{game_slug}}}")

    # Table header with vertical bars
    num_models = len(models)
    col_spec = "l|" + "|".join(["r"] * num_models)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Column headers
    model_headers = " & ".join([model for model in models])
    lines.append(f"Mechanism & {model_headers} \\\\")
    lines.append(r"\midrule")

    # Data rows
    for mech in mechanisms:
        row_data = []
        for model in models:
            score = data[game][mech][model]
            row_data.append(format_score(score, precision))

        row_str = " & ".join(row_data)
        lines.append(f"{mech} & {row_str} \\\\")

    # Table footer
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def compute_aggregate_mean(
    data: Dict[str, Dict[str, Dict[str, float]]],
    game_configs: Dict[str, dict],
    mechanism: str,
    model: str,
) -> Optional[Tuple[float, float]]:
    """
    Compute mean and standard error for a mechanism-model pair across games.

    Args:
        data: Nested dictionary with scores
        game_configs: Game configuration dictionaries
        mechanism: Mechanism name
        model: Model name

    Returns:
        Tuple of (mean, stderr) or None if no data
    """
    # Define the 4 social dilemmas
    social_dilemmas = {
        "PrisonersDilemma",
        "PublicGoods",
        "TravellersDilemma",
        "TrustGame",
    }

    scores = []
    for game in data:
        # Filter for social dilemmas
        if game not in social_dilemmas:
            continue

        score = data[game][mechanism][model]
        # Apply normalization based on game config
        game_config = game_configs[game]
        normalized_score = normalize_score(game, score, game_config)
        scores.append(normalized_score)

    if not scores:
        return None

    n = len(scores)
    mean = sum(scores) / n

    if n == 1:
        stderr = 0.0
    else:
        variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
        stderr = math.sqrt(variance / n)

    return mean, stderr


def generate_aggregate_table(
    mechanisms: List[str],
    models: List[str],
    data: Dict[str, Dict[str, Dict[str, float]]],
    game_configs: Dict[str, dict],
    precision: int = 3,
) -> str:
    """
    Generate aggregated LaTeX table across social dilemma games only.

    Args:
        mechanisms: List of mechanism names (rows)
        models: List of model names (columns)
        data: Nested dictionary with scores
        game_configs: Game configuration dictionaries for normalization
        precision: Number of decimal places

    Returns:
        Complete LaTeX table as string
    """
    # Compute aggregate scores (social dilemmas only, with normalization)
    aggregate_data = {}

    for mech in mechanisms:
        aggregate_data[mech] = {}
        for model in models:
            mean_score = compute_aggregate_mean(
                data, game_configs, mech, model
            )
            aggregate_data[mech][model] = mean_score

    # Start building table
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Aggregate Average Payoffs Across Social Dilemmas (Normalized)}"
    )
    lines.append(r"\label{tab:aggregate_payoffs}")

    # Table header with vertical bars
    num_models = len(models)
    col_spec = "l|" + "|".join(["r"] * num_models)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Column headers
    model_headers = " & ".join([model for model in models])
    lines.append(f"Mechanism & {model_headers} \\\\")
    lines.append(r"\midrule")

    # Data rows
    for mech in mechanisms:
        row_data = []
        for model in models:
            result = aggregate_data[mech][model]
            row_data.append(format_score_with_stderr(result, precision))

        row_str = " & ".join(row_data)
        lines.append(f"{mech} & {row_str} \\\\")

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

    # Build data structure
    data, game_configs = build_data_structure(experiments)

    # Validate all combinations are present
    validate_data_consistency(data, mechanisms, games, models)
    print("Data consistency validated: all mechanism×game×model combinations present")

    # Determine output directory
    output_dir = args.output if args.output else args.batch_path

    # Generate and save per-game tables
    for game in games:
        table_latex = generate_game_table(
            game, mechanisms, models, data, args.precision
        )

        output_path = output_dir / f"table_{game}.tex"
        save_table(table_latex, output_path)

        if not args.quiet:
            print_table(table_latex, f"Table for {game}")

        print(f"Saved: {output_path}")

    # Generate and save aggregate table (social dilemmas only)
    table_latex = generate_aggregate_table(
        mechanisms, models, data, game_configs, args.precision
    )

    output_path = output_dir / "table_aggregate.tex"
    save_table(table_latex, output_path)

    if not args.quiet:
        print_table(table_latex, "Aggregate Table")

    print(f"Saved: {output_path}")

    print(f"\nTotal tables generated: {len(games) + 1}")


if __name__ == "__main__":
    main()
