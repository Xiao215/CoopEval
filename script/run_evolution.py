import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from config import CONFIG_DIR, DATA_DIR
from src.ranking_evaluations.matchup_payoffs import MatchupPayoffs
from src.ranking_evaluations.replicator_dynamics import DiscreteReplicatorDynamics
from src.registry.game_registry import GAME_REGISTRY
from src.registry.mechanism_registry import MECHANISM_REGISTRY
from src.logger_manager import LOGGER
from src.utils.concurrency import set_default_max_workers


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
    """
    config_path = Path(CONFIG_DIR) / filename
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found.")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_matchup_payoffs_from_file(path: Path) -> MatchupPayoffs:
    if not path.exists():
        raise FileNotFoundError(f"Matchup payoff file {path} was not found.")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    print(f"Loaded precomputed matchup payoffs from {path}.")
    return MatchupPayoffs.from_json(payload)


def _prepare_matchup_payoffs(
    mechanism,
    agent_cfgs: list[dict],
    matchup_payoffs_path: str | None,
) -> MatchupPayoffs:
    if matchup_payoffs_path:
        return _load_matchup_payoffs_from_file(DATA_DIR / matchup_payoffs_path)

    print("No precomputed matchup payoff provided; running tournament...")
    matchup_payoffs = mechanism.run_tournament(agent_cfgs=agent_cfgs)
    LOGGER.log_record(
        record=matchup_payoffs.to_json(),
        file_name="matchup_payoffs.json",
    )
    return matchup_payoffs


def main():
    """
    Build agents and run pairwise IPD matches.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    # parser.add_argument('--log', action='store_true', help='Enable logging')
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases figure saving"
    )
    parser.add_argument(
        "--population-payoffs",
        type=str,
        default=None,
        help="Path to a JSON file containing precomputed population payoffs.",
    )

    args = parser.parse_args()

    config = load_config(filename=args.config)

    concurrency_cfg = config.get("concurrency", {}) or {}
    set_default_max_workers(concurrency_cfg.get("max_workers"))

    game_class = GAME_REGISTRY[config["game"]["type"]]
    mechanism_class = MECHANISM_REGISTRY[config["mechanism"]["type"]]

    game = game_class(**config["game"].get("kwargs", {}))
    mech_kwargs = (config["mechanism"].get("kwargs", {}) or {}).copy()
    mechanism = mechanism_class(base_game=game, **mech_kwargs)

    LOGGER.log_record(config, "config.json")

    print(
        f"Running {config['game']['type']} with mechanism {config['mechanism']['type']}.\n"
    )

    matchup_payoffs = _prepare_matchup_payoffs(
        mechanism=mechanism,
        agent_cfgs=config["agents"],
        matchup_payoffs_path=args.matchup_payoffs,
    )

    replicator_dynamics = DiscreteReplicatorDynamics(
        agent_cfgs=config["agents"],
        matchup_payoffs=matchup_payoffs,
    )

    # TODO: currently initial_population can only be a string, rather than a dynamic population
    population_history = replicator_dynamics.run_dynamics(
        initial_population=config["evolution"]["initial_population"],
        steps=config["evolution"]["steps"],
    )

if __name__ == "__main__":
    set_seed()
    main()
