import argparse
import random
import re
from pathlib import Path

import numpy as np
import torch
import yaml

from config import CONFIG_DIR
from src.evolution.replicator_dynamics import DiscreteReplicatorDynamics
from src.plot import plot_probability_evolution
from src.registry.game_registry import GAME_REGISTRY
from src.registry.mechanism_registry import MECHANISM_REGISTRY
from src.logger_manager import WandBLogger, LOGGER
from src.utils.concurrency import set_default_max_workers


def set_seed(seed=42):
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


def _slugify(text: str, *, max_len: int = 40) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text.strip())
    slug = slug.strip("-")
    if max_len and len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug.lower() or "item"


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

    args = parser.parse_args()

    config = load_config(filename=args.config)

    concurrency_cfg = config.get("concurrency", {}) or {}
    set_default_max_workers(concurrency_cfg.get("max_workers"))

    game_class = GAME_REGISTRY[config["game"]["type"]]
    mechanism_class = MECHANISM_REGISTRY[config["mechanism"]["type"]]

    game = game_class(**config["game"].get("kwargs", {}))
    mech_kwargs = (config["mechanism"].get("kwargs", {}) or {}).copy()
    mechanism = mechanism_class(base_game=game, **mech_kwargs)
    tournament_cfg = config.get("tournament", {}) or {}
    if "match_workers" in tournament_cfg:
        mechanism.match_workers = max(1, int(tournament_cfg["match_workers"]))

    LOGGER.log_record(config, "config.json")

    print(
        f"Running {config['game']['type']} with mechanism {config['mechanism']['type']}.\n"
    )

    replicator_dynamics = DiscreteReplicatorDynamics(
        agent_cfgs=config["agents"],
        mechanism=mechanism,
    )

    # TODO: currently initial_population can only be a string, rather than a dynamic population
    population_history = replicator_dynamics.run_dynamics(
        initial_population=config["evolution"]["initial_population"],
        steps=config["evolution"]["steps"],
    )

if __name__ == "__main__":
    set_seed()
    main()
