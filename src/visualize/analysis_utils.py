"""Shared utilities for experiment visualization scripts."""

import json
from pathlib import Path
from typing import Any


class NormalizeScore:
    """Normalizes game scores to [0, 1] scale based on Nash Equilibrium and cooperative payoffs.

    Precomputes normalization parameters at initialization for efficient repeated normalization.

    Args:
        game: Game name (e.g., "PrisonersDilemma", "PublicGoods")
        game_config: Game configuration dict containing payoff structure

    Attributes:
        game: Game name
        ne_payoff: Nash Equilibrium payoff (worst case)
        coop_payoff: Cooperative payoff (best case)

    Example:
        >>> normalizer = NormalizeScore("PrisonersDilemma", config)
        >>> normalized = normalizer.normalize(raw_score)
    """

    def __init__(self, game: str, game_config: dict):
        """Initialize normalizer by precomputing NE and cooperative payoffs.

        Args:
            game: Game name
            game_config: Game configuration dict
        """
        self.game = game

        if game == "PrisonersDilemma":
            payoff_matrix = game_config["kwargs"]["payoff_matrix"]
            self.ne_payoff = payoff_matrix["DD"][0]  # NE: both defect
            self.coop_payoff = payoff_matrix["CC"][1]  # Cooperative: both cooperate

        elif game == "PublicGoods":
            self.coop_payoff = game_config["kwargs"]["multiplier"]
            self.ne_payoff = 1  # NE: no one contributes

        elif game == "TravellersDilemma":
            self.ne_payoff = game_config["kwargs"]["min_claim"]
            spacing = game_config["kwargs"]["claim_spacing"]
            num_actions = game_config["kwargs"]["num_actions"]
            self.coop_payoff = self.ne_payoff + spacing * (num_actions - 1)

        elif game == "TrustGame":
            payoff_matrix = game_config["kwargs"]["payoff_matrix"]
            self.ne_payoff = payoff_matrix["KK"][0]  # NE: both keep
            self.coop_payoff = payoff_matrix["GG"][0]  # Cooperative: both give

        elif game == "StagHunt":
            self.ne_payoff = 3
            self.coop_payoff = 5

        elif game == "MatchingPennies":
            self.ne_payoff = -1
            self.coop_payoff = 0

        else:
            # For non-social-dilemma games, use identity normalization
            self.ne_payoff = 0.0
            self.coop_payoff = 1.0

    def normalize(self, score: float) -> float:
        """Normalize a score to [0, 1] scale.

        Args:
            score: Raw score from the game

        Returns:
            Normalized score where 0 = NE payoff, 1 = cooperative payoff
        """
        if self.coop_payoff == self.ne_payoff:
            # Avoid division by zero
            return 0.0

        return (score - self.ne_payoff) / (self.coop_payoff - self.ne_payoff)


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


def simplify_model_name(model_name: str) -> str:
    """
    Simplify model name to a short, canonical version.

    Maps specific model names to their simplified versions for consistent display.
    Handles both CoT (reasoning) and IO (base) variants of models.

    Args:
        model_name: Full model name, potentially with agent type suffix like "(CoT)"

    Returns:
        Simplified model name

    Examples:
        "google/gemini-3-flash-preview(CoT)" -> "gemini-3-f-reasoning"
        "google/gemini-3-flash-preview" -> "gemini-3-f-base"
        "openai/gpt-5.2(CoT)" -> "gpt-5.2"
        "openai/gpt-4o-2024-05-13(CoT)" -> "gpt-4o"
        "anthropic/claude-sonnet-4.5(CoT)" -> "claude-sonnet-4.5"
        "qwen/qwen3-30b-a3b-instruct-2507(CoT)" -> "qwen3-30b-a3b"
    """
    # Define mappings for specific models
    model_mappings = {
        "google/gemini-3-flash-preview": {
            "with_cot": "gemini-3-f-reasoning",
            "without_cot": "gemini-3-f-base"
        },
        "openai/gpt-5.2": "gpt-5.2",
        "openai/gpt-4o-2024-05-13": "gpt-4o",
        "anthropic/claude-sonnet-4.5": "claude-sonnet-4.5",
        "qwen/qwen3-30b-a3b-instruct-2507": "qwen3-30b-a3b"
    }

    # Check if model has CoT suffix
    has_cot = "(CoT)" in model_name
    base_model = model_name.replace("(CoT)", "").replace("(IO)", "").strip()

    # Look up the mapping
    if base_model in model_mappings:
        mapping = model_mappings[base_model]
        if isinstance(mapping, dict):
            # Special handling for models with different variants
            return mapping["with_cot"] if has_cot else mapping["without_cot"]
        else:
            # Simple mapping
            return mapping

    # Fallback to basic cleaning if no mapping exists
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


def sort_mechanisms(mechanisms: list[str]) -> list[str]:
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
        "ReputationFirstOrder",
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


def sort_games(games: list[str]) -> list[str]:
    """
    Sort games in the preferred order.

    Args:
        games: List of game names

    Returns:
        Sorted list of game names
    """
    # Define preferred order (case-insensitive matching)
    preferred_order = [
        "PrisonersDilemma",
        "PublicGoods",
        "TravellersDilemma",
        "TrustGame",
        "StagHunt",
        "MatchingPennies"
    ]

    # Create a mapping for case-insensitive lookup
    order_map = {name.lower(): i for i, name in enumerate(preferred_order)}

    # Sort games by preferred order, alphabetically for any not in the list
    def sort_key(game):
        game_lower = game.lower()
        if game_lower in order_map:
            return (0, order_map[game_lower])
        else:
            return (1, game)  # Unknown games go last, sorted alphabetically

    return sorted(games, key=sort_key)


def sort_models(models: list[str]) -> list[str]:
    """
    Sort models in the preferred order.

    Preferred order: claude, gemini-reasoning, gemini-base, gpt-5.2, gpt-4o, qwen
    
    Args:
        models: List of model names (full names with provider prefixes)

    Returns:
        Sorted list of model names
    """
    # Define preferred order based on model patterns
    # Order: claude, gemini with reasoning, gemini without reasoning, gpt 5.2, gpt 4o, qwen
    def sort_key(model):
        model_lower = model.lower()
        
        # Claude models
        if "claude" in model_lower:
            return (0, model)
        # Gemini with reasoning (CoT)
        elif "gemini" in model_lower and "(cot)" in model_lower:
            return (1, model)
        # Gemini without reasoning (IO or base)
        elif "gemini" in model_lower:
            return (2, model)
        # GPT-5.2
        elif "gpt-5" in model_lower:
            return (3, model)
        # GPT-4o
        elif "gpt-4" in model_lower:
            return (4, model)
        # Qwen models
        elif "qwen" in model_lower:
            return (5, model)
        # Unknown models go last, sorted alphabetically
        else:
            return (6, model)
    
    return sorted(models, key=sort_key)
