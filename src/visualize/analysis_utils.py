"""Shared utilities for experiment visualization scripts."""

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_experiment_subfolders(experiment_dir: Path) -> list[Path]:
    """
    Find experiment subfolders, excluding configs and slurm directories.

    Returns subdirectories that contain actual experiments.
    """
    return [
        d for d in experiment_dir.iterdir()
        if d.is_dir() and d.name not in {"configs", "slurm"}
    ]


def clean_model_name(model_name: str, max_length: int = 25) -> str:
    """
    Shorten model name for display in visualizations.

    Removes provider prefix and truncates if too long.
    Examples:
        "google/gemini-3-flash-preview(CoT)" -> "gemini-3-flash-preview(CoT)"
        "openai/gpt-5.2(CoT)" -> "gpt-5.2(CoT)"
    """
    parts = model_name.split("/")
    if len(parts) > 1:
        model_and_type = parts[-1]
        if len(model_and_type) > max_length:
            model_and_type = model_and_type[:max_length - 3] + "..."
        return model_and_type
    return model_name


def get_num_players_from_matchup(matchup_data: dict[str, Any]) -> int:
    """
    Extract number of players from matchup_payoffs.json data.

    Args:
        matchup_data: Loaded matchup_payoffs.json content

    Returns:
        Number of players in the game
    """
    return len(matchup_data["profile"][0]["players"])
