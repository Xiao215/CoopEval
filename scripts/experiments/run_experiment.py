import argparse
from pathlib import Path

from coopeval.agents.agent_manager import Agent
from coopeval.config import DATA_DIR
from coopeval.config_loader import ConfigLoader
from coopeval.logger_manager import LOGGER
from coopeval.ranking_evaluations.deviation_rating import DeviationRating
from coopeval.ranking_evaluations.matchup_payoffs import MatchupPayoffs
from coopeval.ranking_evaluations.replicator_dynamics import (
    DiscreteReplicatorDynamics,
)
from coopeval.ranking_evaluations.reputation_payoffs import ReputationPayoffs
from coopeval.registry.agent_registry import create_players_with_player_id
from coopeval.registry.game_registry import GAME_REGISTRY
from coopeval.registry.mechanism_registry import MECHANISM_REGISTRY
from coopeval.script_utils.reproducibility import set_seed
from coopeval.utils.json_io import load_json

DEFAULT_EVOL_INITIAL_POPULATION = "uniform"
DEFAULT_EVOL_STEPS = 25
DEFAULT_EVOL_LR_METHOD = "constant"
DEFAULT_EVOL_LR_NU = 0.1

DEFAULT_DEVIATION_TOLERANCE = 1e-14


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

    concurrency_cfg = config.get("concurrency", {}) or {}
    if "tournament_workers" in concurrency_cfg:
        mech_kwargs["tournament_workers"] = concurrency_cfg[
            "tournament_workers"
        ]

    mechanism = mechanism_class(base_game=game, **mech_kwargs)

    print(
        f"Running {config['game']['type']} with mechanism {config['mechanism']['type']}.\n"
    )

    return game, mechanism


def _load_matchup_payoffs_from_file(path: Path) -> MatchupPayoffs:
    """Load pre-computed matchup payoffs from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Matchup payoff file {path} was not found.")
    payload = load_json(path)
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


def report_agent_averages(payoffs: MatchupPayoffs | ReputationPayoffs) -> None:
    """
    Report agent average payoffs for all configured strategies.

    This works for both MatchupPayoffs and ReputationPayoffs.

    Args:
        payoffs: PayoffsBase instance
    """
    print("\n" + "=" * 60)
    print("AGENT AVERAGE PAYOFFS")
    print("=" * 60)

    agent_avg = payoffs.agent_average_payoff()

    for agent, avg_payoff in sorted(agent_avg.items()):
        if avg_payoff is None:
            print(f"  {agent}: Never played")
        else:
            print(f"  {agent}: {avg_payoff:.4f}")
    LOGGER.log_record(agent_avg, "agent_average_payoff.json")
    print("=" * 60 + "\n")


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
    print("\n" + "=" * 60)
    print("RUNNING EVOLUTIONARY DYNAMICS")
    print("=" * 60 + "\n")

    replicator_dynamics = DiscreteReplicatorDynamics(
        players=players,
        matchup_payoffs=payoffs,
    )

    population_history = replicator_dynamics.run_dynamics(
        initial_population=eval_kwargs.get(
            "initial_population", DEFAULT_EVOL_INITIAL_POPULATION
        ),
        steps=int(eval_kwargs.get("steps", DEFAULT_EVOL_STEPS)),
        lr_method=eval_kwargs.get("lr_method", DEFAULT_EVOL_LR_METHOD),
        lr_nu=float(eval_kwargs.get("lr_nu", DEFAULT_EVOL_LR_NU)),
    )

    LOGGER.log_record(population_history, "population_history.json")

    print("\n" + "=" * 60 + "\n")


def run_deviation_rating(payoffs: MatchupPayoffs, eval_kwargs: dict) -> None:
    """
    Run deviation rating evaluation.

    Args:
        payoffs: MatchupPayoffs instance
        eval_kwargs: Evaluation-specific kwargs from config
    """
    print("\n" + "=" * 60)
    print("RUNNING DEVIATION RATING")
    print("=" * 60 + "\n")

    payoffs.ensure_payoff_tensor()

    deviation_rating = DeviationRating(
        matchup_payoffs=payoffs,
        tolerance=float(
            eval_kwargs.get("tolerance", DEFAULT_DEVIATION_TOLERANCE)
        ),
    )

    ratings = deviation_rating.compute_ratings()

    print("\nDeviation Ratings:")
    for model, rating in sorted(
        ratings.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {model}: {rating:.6f}")

    LOGGER.log_record(ratings, "deviation_ratings.json")

    print("\n" + "=" * 60 + "\n")


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
    report_agent_averages(payoffs)

    if isinstance(payoffs, ReputationPayoffs):
        print("\n" + "!" * 60)
        print("! REPUTATION MECHANISM DETECTED")
        print("! Only model averages available (no tensor-based evaluations)")
        print("!" * 60 + "\n")
        return

    evaluation_config = config.get("evaluation", {})
    methods = evaluation_config.get("methods", [])

    if not methods:
        print(
            "\nNo evaluation methods configured in config['evaluation']['methods']"
        )
        print("Only model averages will be reported.\n")
        return

    for eval_method in methods:
        eval_type = eval_method.get("type")
        eval_kwargs = eval_method.get("kwargs", {})

        if eval_type == "evolutionary_dynamics":
            run_evolutionary_dynamics(payoffs, players, eval_kwargs)
        elif eval_type == "deviation_rating":
            run_deviation_rating(payoffs, eval_kwargs)
        else:
            print(
                f"\nWARNING: Unknown evaluation type '{eval_type}' - skipping"
            )


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
    parser.add_argument(
        "--config", type=str, required=True, help="Config YAML file name"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases figure saving",
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
        help="Custom output directory for this experiment (overrides default timestamped directory)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment (used as subdirectory under output-dir)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    if args.output_dir:
        if args.experiment_name:
            experiment_dir = Path(args.output_dir) / args.experiment_name
        else:
            experiment_dir = Path(args.output_dir)
        LOGGER.set_log_dir(experiment_dir)
        print(f"Logging to: {experiment_dir}")

    config = load_config(filename=args.config)

    game, mechanism = setup_game_and_mechanism(config)
    players = create_players_with_player_id(config["agents"], game.num_players)

    config["seed"] = args.seed
    LOGGER.log_record(config, "config.json")

    payoffs = run_mechanism(mechanism, players, args)

    run_evaluations(payoffs, players, config)


if __name__ == "__main__":
    main()
