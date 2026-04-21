#!/usr/bin/env python3
"""Utility to estimate how many matches each player will play for a config."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from coopeval.config import CONFIG_DIR
from coopeval.registry.game_registry import GAME_REGISTRY
from coopeval.utils.json_io import clean_path

REPUTATION_TYPES = {"Reputation", "ReputationFirstOrder"}
DEFAULT_GAME_FILENAMES = [
    "prisoners_dilemma.yaml",
    "public_goods.yaml",
    "travellers_dilemma.yaml",
    "trust_game.yaml",
]
DEFAULT_MECHANISM_FILENAMES = [
    "contracting.yaml",
    "mediation.yaml",
    "no_mechanism.yaml",
    "repetition.yaml",
    "reputation.yaml",
    "reputation_first_order.yaml",
]

GAME_CONFIG_DIR = CONFIG_DIR / "games"
MECHANISM_CONFIG_DIR = CONFIG_DIR / "mechanisms"


def load_yaml(path: Path):
    """Return parsed YAML from path, raising if missing or empty."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ValueError(f"Config file {path} is empty.")
    return data


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for counting matches."""
    parser = argparse.ArgumentParser(
        description=(
            "Given agent, game, and mechanism configs, report how many matches "
            "each player will play."
        )
    )
    parser.add_argument(
        "--agents-config",
        type=clean_path,
        required=True,
        help=(
            "Path to the agents YAML file. "
            "Relative paths are resolved first as-is, then under configs/agents/."
        ),
    )
    return parser.parse_args()


def format_summary_line(label: str, value) -> str:
    """Return a padded summary line for console output."""
    return f"{label:<28}: {value}"


def load_game_info(path: Path) -> dict:
    """Load a game config and derive metadata required for summaries."""
    game_cfg = load_yaml(path)
    game_type = game_cfg.get("type")
    if not game_type:
        raise ValueError(f"Game config {path} must set 'type'.")
    game_kwargs = game_cfg.get("kwargs") or {}
    game_class = GAME_REGISTRY[game_type]
    game = game_class(**game_kwargs)
    num_players = game.num_players
    return {
        "path": path,
        "type": game_type,
        "num_players": num_players,
    }


def load_mechanism_info(path: Path) -> dict:
    """Load a mechanism config and read the type and round count."""
    mechanism_cfg = load_yaml(path)
    mech_type = mechanism_cfg["type"]
    mech_kwargs = mechanism_cfg.get("kwargs") or {}
    configured_rounds = int(mech_kwargs.get("num_rounds", 1))
    return {
        "path": path,
        "type": mech_type,
        "rounds": configured_rounds,
    }


def load_agent_models(agent_cfgs: list[dict]) -> list[str]:
    """Return a stable identifier for each agent's underlying model."""
    models: list[str] = []
    for cfg in agent_cfgs:
        llm_cfg = cfg.get("llm") or {}
        model_name = llm_cfg.get("model", "unknown-model")
        provider = llm_cfg.get("provider")
        identifier = f"{provider}:{model_name}" if provider else str(model_name)
        models.append(identifier)
    return models


def summarize_combo(
    *,
    agents_per_seat: int,
    game_info: dict,
    mech_info: dict,
) -> tuple[list[str], dict[str, int]]:
    """Compute per-combo metrics along with formatted summary lines."""
    num_players = game_info["num_players"]
    mech_type = mech_info["type"]
    configured_rounds = mech_info["rounds"]

    lines = [
        format_summary_line("Game type", game_info["type"]),
        format_summary_line("Game config", game_info["path"]),
        format_summary_line("Mechanism type", mech_type),
        format_summary_line("Mechanism config", mech_info["path"]),
        format_summary_line("Players per match", num_players),
    ]

    total_base_game_plays: int
    if mech_type in REPUTATION_TYPES:
        matches_per_player = 1
        matches_per_round = agents_per_seat
        base_game_plays_per_player = configured_rounds
        total_base_game_plays = matches_per_round * configured_rounds
        lines.append(
            format_summary_line("Rounds in tournament", configured_rounds)
        )
        lines.append(
            format_summary_line("Matches per round", matches_per_round)
        )
        lines.append(
            format_summary_line(
                "Note", "Matches per player fixed at 1 (shared pool)"
            )
        )
    else:
        matches_per_player = agents_per_seat ** max(num_players - 1, 0)
        total_matchups = agents_per_seat**num_players
        base_game_plays_per_player = matches_per_player * configured_rounds
        total_base_game_plays = total_matchups * configured_rounds
        lines.append(
            format_summary_line("Rounds per matchup", configured_rounds)
        )
        lines.append(
            format_summary_line("Total unique matchups", total_matchups)
        )

    lines.append(format_summary_line("Matches per player", matches_per_player))
    lines.append(
        format_summary_line(
            "Base-game plays per player", base_game_plays_per_player
        )
    )
    lines.append(
        format_summary_line(
            "Total base-game plays overall", total_base_game_plays
        )
    )
    metrics = {
        "matches_per_player": matches_per_player,
        "base_game_plays_per_player": base_game_plays_per_player,
        "total_base_game_plays": total_base_game_plays,
        "players_per_match": num_players,
    }
    return lines, metrics


def main() -> None:
    """Resolve config inputs and print match-volume summaries."""
    args = parse_args()

    agents_path = args.agents_config
    game_paths = [
        (GAME_CONFIG_DIR / path).expanduser().resolve()
        for path in DEFAULT_GAME_FILENAMES
    ]
    mechanism_paths = [
        (MECHANISM_CONFIG_DIR / path).expanduser().resolve()
        for path in DEFAULT_MECHANISM_FILENAMES
    ]

    agents_cfg = load_yaml(agents_path)
    if not isinstance(agents_cfg, list) or not agents_cfg:
        raise ValueError(
            f"Agents config {agents_path} must define a non-empty list."
        )
    agents_per_seat = len(agents_cfg)
    agent_models = load_agent_models(agents_cfg)

    game_infos = [load_game_info(path) for path in game_paths]
    mechanism_infos = [load_mechanism_info(path) for path in mechanism_paths]

    heading = "Match Volume Summary"
    print(heading)
    print("-" * len(heading))
    print(format_summary_line("Agents per seat", agents_per_seat))

    combo_records: list[dict[str, int | str]] = []
    model_totals: Counter[str] = Counter()
    combo_idx = 1
    for game_info in game_infos:
        for mech_info in mechanism_infos:
            print()
            combo_header = f"[{combo_idx}] Game={game_info['type']} | Mechanism={mech_info['type']}"
            print(combo_header)
            print("-" * len(combo_header))
            lines, metrics = summarize_combo(
                agents_per_seat=agents_per_seat,
                game_info=game_info,
                mech_info=mech_info,
            )
            for line in lines:
                print(line)
            combo_records.append(
                {
                    "game": game_info["type"],
                    "mechanism": mech_info["type"],
                    **metrics,
                }
            )
            increment_per_model = (
                metrics["base_game_plays_per_player"]
                * metrics["players_per_match"]
            )
            for model_id in agent_models:
                model_totals[model_id] += increment_per_model
            combo_idx += 1

    if combo_records:
        print()
        print("Combo Metrics Summary")
        print("---------------------")
        for record in combo_records:
            print(
                f"- {record['game']} + {record['mechanism']}: "
                f"matches/player={record['matches_per_player']}, "
                f"base-game plays/player={record['base_game_plays_per_player']}, "
                f"total base-game plays={record['total_base_game_plays']}"
            )

        total_matches_per_player = sum(
            int(record["matches_per_player"]) for record in combo_records
        )
        total_base_plays_per_player = sum(
            int(record["base_game_plays_per_player"])
            for record in combo_records
        )
        total_base_plays_overall = sum(
            int(record["total_base_game_plays"]) for record in combo_records
        )

        print()
        aggregate_heading = "Aggregate Totals"
        print(aggregate_heading)
        print("-" * len(aggregate_heading))
        print(format_summary_line("Combo count", len(combo_records)))
        print(
            format_summary_line(
                "Matches per player (sum)", total_matches_per_player
            )
        )
        print(
            format_summary_line(
                "Base-game plays/player (sum)", total_base_plays_per_player
            )
        )
        print(
            format_summary_line(
                "Total base-game plays overall", total_base_plays_overall
            )
        )
        if model_totals:
            print()
            print("Total base-game plays per model")
            print("--------------------------------")
            for model, total in sorted(model_totals.items()):
                print(f"{model:<40} {total:>12}")


if __name__ == "__main__":
    main()
