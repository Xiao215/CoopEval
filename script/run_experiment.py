import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from config import DATA_DIR
from src.config_loader import ConfigLoader
from src.ranking_evaluations.matchup_payoffs import MatchupPayoffs
from src.ranking_evaluations.reputation_payoffs import ReputationPayoffs
from src.ranking_evaluations.replicator_dynamics import DiscreteReplicatorDynamics
from src.ranking_evaluations.deviation_rating import DeviationRating
from src.registry.agent_registry import create_players_with_player_id
from src.registry.game_registry import GAME_REGISTRY
from src.registry.mechanism_registry import MECHANISM_REGISTRY
from src.logger_manager import LOGGER
from src.utils.concurrency import set_default_max_workers
from src.agents.agent_manager import Agent


# =============================================================================
# DEFAULT EVALUATION PARAMETERS
# =============================================================================

# Evolutionary Dynamics defaults
DEFAULT_EVOL_INITIAL_POPULATION = "uniform"
DEFAULT_EVOL_STEPS = 25
DEFAULT_EVOL_LR_METHOD = "constant"
DEFAULT_EVOL_LR_NU = 0.1

# Deviation Rating defaults
DEFAULT_DEVIATION_TOLERANCE = 1e-14


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def set_seed(seed=42):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(filename: str) -> dict:
    """
    Load and parse a YAML configuration file.

    Supports both legacy monolithic configs and modular configs.
    """
    loader = ConfigLoader()
    return loader.load_main_config(filename)


def setup_game_and_mechanism(config: dict):
    """
    Initialize game and mechanism from config.

    Args:
        config: Configuration dictionary

    Returns:
        tuple: (game, mechanism)
    """
    game_class = GAME_REGISTRY[config["game"]["type"]]
    mechanism_class = MECHANISM_REGISTRY[config["mechanism"]["type"]]

    game = game_class(**config["game"].get("kwargs", {}))
    mech_kwargs = (config["mechanism"].get("kwargs", {}) or {}).copy()
    mechanism = mechanism_class(base_game=game, **mech_kwargs)

    print(
        f"Running {config['game']['type']} with mechanism {config['mechanism']['type']}.\n"
    )

    return game, mechanism


def _load_matchup_payoffs_from_file(path: Path) -> MatchupPayoffs:
    """Load pre-computed matchup payoffs from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Matchup payoff file {path} was not found.")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    print(f"Loaded precomputed matchup payoffs from {path}.")
    return MatchupPayoffs.from_json(payload)


def run_mechanism(
    mechanism, players: list[Agent], args
) -> MatchupPayoffs | ReputationPayoffs:
    """
    Run the mechanism tournament or load pre-computed payoffs.

    Args:
        mechanism: Mechanism instance
        players: List of Agent instances
        args: Command-line arguments

    Returns:
        PayoffsBase instance (either MatchupPayoffs or ReputationPayoffs)
    """
    if args.matchup_payoffs:
        payoffs = _load_matchup_payoffs_from_file(
            DATA_DIR / args.matchup_payoffs
        )
    else:
        print("No precomputed matchup payoff provided; running tournament...")
        payoffs = mechanism.run_tournament(players)
        LOGGER.log_record(
            record=payoffs.to_json(),
            file_name="matchup_payoffs.json",
        )

    return payoffs


def report_model_averages(payoffs: MatchupPayoffs | ReputationPayoffs) -> None:
    """
    Report model average payoffs for all agents.

    This works for both MatchupPayoffs and ReputationPayoffs.

    Args:
        payoffs: PayoffsBase instance
    """
    print("\n" + "="*60)
    print("MODEL AVERAGE PAYOFFS")
    print("="*60)

    agent_avg = payoffs.agent_average_payoff()

    # Pretty print the results
    for agent, avg_payoff in sorted(agent_avg.items()):
        if avg_payoff is None:
            print(f"  {agent}: Never played")
        else:
            print(f"  {agent}: {avg_payoff:.4f}")
    LOGGER.log_record(agent_avg, "agent_average_payoff.json")
    print("="*60 + "\n")


def run_evolutionary_dynamics(
    payoffs: MatchupPayoffs, players: list[Agent], eval_kwargs: dict
) -> None:
    """
    Run evolutionary dynamics evaluation.

    Args:
        payoffs: MatchupPayoffs instance
        players: List of Agent instances
        eval_kwargs: Evaluation-specific kwargs from config
    """
    print("\n" + "="*60)
    print("RUNNING EVOLUTIONARY DYNAMICS")
    print("="*60 + "\n")

    replicator_dynamics = DiscreteReplicatorDynamics(
        players=players,
        matchup_payoffs=payoffs,
    )

    population_history = replicator_dynamics.run_dynamics(
        initial_population=eval_kwargs.get("initial_population", DEFAULT_EVOL_INITIAL_POPULATION),
        steps=int(eval_kwargs.get("steps", DEFAULT_EVOL_STEPS)),
        lr_method=eval_kwargs.get("lr_method", DEFAULT_EVOL_LR_METHOD),
        lr_nu=float(eval_kwargs.get("lr_nu", DEFAULT_EVOL_LR_NU)),
    )

    # Log the population history
    LOGGER.log_record(population_history, "population_history.json")

    print("\n" + "="*60 + "\n")


def run_deviation_rating(payoffs: MatchupPayoffs, eval_kwargs: dict) -> None:
    """
    Run deviation rating evaluation.

    Args:
        payoffs: MatchupPayoffs instance
        eval_kwargs: Evaluation-specific kwargs from config
    """
    print("\n" + "="*60)
    print("RUNNING DEVIATION RATING")
    print("="*60 + "\n")

    # Ensure payoff tensor is built
    if payoffs._payoff_tensor is None:
        payoffs.build_payoff_tensor()

    deviation_rating = DeviationRating(
        matchup_payoffs=payoffs,
        tolerance=float(
            eval_kwargs.get("tolerance", DEFAULT_DEVIATION_TOLERANCE)
        ),
    )

    ratings = deviation_rating.compute_ratings()

    # Pretty print the results
    print("\nDeviation Ratings:")
    for model, rating in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {rating:.6f}")

    LOGGER.log_record(ratings, "deviation_ratings.json")

    print("\n" + "="*60 + "\n")


def run_evaluations(
    payoffs: MatchupPayoffs | ReputationPayoffs,
    players: list[Agent],
    config: dict,
) -> None:
    """
    Run all configured evaluations.

    Args:
        payoffs: MatchupPayoffs instance
        players: List of Agent instances
        config: Configuration dictionary
    """
    # Always report model averages first
    report_model_averages(payoffs)

    # For reputation: ONLY model averages possible
    if isinstance(payoffs, ReputationPayoffs):
        print("\n" + "!"*60)
        print("! REPUTATION MECHANISM DETECTED")
        print("! Only model averages available (no tensor-based evaluations)")
        print("!"*60 + "\n")
        return

    # For matchup-based: run configured evaluations
    evaluation_config = config.get("evaluation", {})
    methods = evaluation_config.get("methods", [])

    if not methods:
        print("\nNo evaluation methods configured in config['evaluation']['methods']")
        print("Only model averages will be reported.\n")
        return

    # Run each evaluation method
    for eval_method in methods:
        eval_type = eval_method.get("type")
        eval_kwargs = eval_method.get("kwargs", {})

        if eval_type == "evolutionary_dynamics":
            run_evolutionary_dynamics(payoffs, players, eval_kwargs)
        elif eval_type == "deviation_rating":
            run_deviation_rating(payoffs, eval_kwargs)
        else:
            print(f"\nWARNING: Unknown evaluation type '{eval_type}' - skipping")


def main():
    """
    Main experiment pipeline:
    1. Load config
    2. Setup game and mechanism
    3. Run mechanism (tournament)
    4. Run evaluations
    """
    parser = argparse.ArgumentParser(
        description="Run game-theoretic experiments with configurable evaluations"
    )
    parser.add_argument("--config", type=str, required=True, help="Config YAML file name")
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases figure saving"
    )
    parser.add_argument(
        "--matchup-payoffs",
        type=str,
        default=None,
        help="Path to a JSON file containing precomputed matchup payoffs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for this experiment (overrides default timestamped directory)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment (used as subdirectory under output-dir)"
    )

    args = parser.parse_args()

    # 0. Setup custom logging directory if provided
    if args.output_dir:
        if args.experiment_name:
            experiment_dir = Path(args.output_dir) / args.experiment_name
        else:
            experiment_dir = Path(args.output_dir)
        LOGGER.set_log_dir(experiment_dir)
        print(f"Logging to: {experiment_dir}")

    # 1. Load config
    config = load_config(filename=args.config)

    # Setup concurrency
    concurrency_cfg = config.get("concurrency", {}) or {}
    set_default_max_workers(concurrency_cfg.get("max_workers"))

    # 2. Setup game, agents and mechanism
    game, mechanism = setup_game_and_mechanism(config)
    players = create_players_with_player_id(config["agents"], game.num_players)
    LOGGER.log_record(config, "config.json")

    # 3. Run mechanism (tournament)
    payoffs = run_mechanism(mechanism, players, args)

    # 5. Run evaluations
    run_evaluations(payoffs, players, config)


if __name__ == "__main__":
    set_seed()
    main()
