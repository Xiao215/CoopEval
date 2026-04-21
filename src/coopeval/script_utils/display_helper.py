"""Display and naming helpers shared by repository scripts."""

from __future__ import annotations

import re
from typing import Any, Iterable

PLAYER_ID_SUFFIX_RE = re.compile(r"#P(?P<player_id>\d+)$")


# Abbreviations for mechanism kwargs used when building display suffixes.
KWARG_ABBREVS = {
    "discount": r"$\delta$",
    "lookup_depth": "$k$",
    "max_recursion_depth": "$r$",
    "num_rounds": "$n$",
}

KWARG_SORT_ORDER = {
    "lookup_depth": 0,
    "discount": 1,
    "max_recursion_depth": 2,
    "num_rounds": 3,
}


def _fmt_kwarg_val(v: Any) -> str:
    """Format a kwarg value: 3.0 -> '3', 0.7 -> '0.7'."""
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def make_mechanism_suffix(kwargs: dict, varying_keys) -> str:
    """Build a compact 'k1=v1, k2=v2' suffix string for varying kwargs."""
    keys = set(varying_keys)

    if (
        "lookup_depth" in keys
        and "max_recursion_depth" in keys
        and kwargs.get("lookup_depth") == kwargs.get("max_recursion_depth")
    ):
        keys.remove("max_recursion_depth")

    parts = [
        f"{KWARG_ABBREVS.get(k, k)}={_fmt_kwarg_val(kwargs[k])}"
        for k in sorted(
            keys,
            key=lambda key: (KWARG_SORT_ORDER.get(key, 999), key),
        )
    ]
    return ", ".join(parts)


def format_mechanism_name(mechanism: str) -> str:
    """Convert internal mechanism names to plot/table display names."""
    if " (" in mechanism:
        base_type, rest = mechanism.split(" (", 1)
        suffix = " (" + rest
    else:
        base_type, suffix = mechanism, ""
    if base_type == "Reputation":
        return "Reputation+" + suffix
    if base_type == "ReputationFirstOrder":
        return "Reputation-" + suffix
    return mechanism


def strip_player_suffix(name: str) -> str:
    """Remove the player-position suffix, e.g. '#P1', from a player name."""
    return PLAYER_ID_SUFFIX_RE.sub("", name)


def detect_agent_type(player_name: str) -> str:
    """Infer whether the serialized player used CoT or IO prompting."""
    patterns = ["CoT", "IO"]

    for pattern in patterns:
        if f"({pattern})" in player_name:
            return pattern
    raise ValueError(f"Could not detect agent type from: {player_name}")


def extract_player_id(player_name: str) -> int:
    """Return the trailing #P suffix as an integer, e.g. '#P2' -> 2."""
    match = PLAYER_ID_SUFFIX_RE.search(player_name)
    if not match:
        raise ValueError(f"Invalid player name format: {player_name}")
    return int(match.group("player_id"))


def extract_agent_name(player_name: str) -> str:
    """Extract the agent name from a serialized player name."""
    base = PLAYER_ID_SUFFIX_RE.sub("", player_name)
    return base.strip()


def extract_model_name(player_name: str) -> str:
    """Strip agent annotations and seat suffix from serialized player names."""
    agent_name = extract_agent_name(player_name)
    base = agent_name.replace("(CoT)", "").replace("(IO)", "")
    return base.strip()


def normalize_filter(values: list[str] | None) -> set[str]:
    """Normalize optional CLI filter lists to lowercase sets."""
    if values is None:
        return set()
    return {
        value.strip().lower() for value in values if value and value.strip()
    }


def format_model_name(model_name: str) -> str:
    """Format model names to canonical short labels for figures and tables."""
    model_name = strip_player_suffix(model_name)
    model_mappings = {
        "google/gemini-3-flash-preview": {
            "with_cot": "Gemini-R",
            "without_cot": "Gemini-B",
        },
        "openai/gpt-5.2": "GPT-5.2",
        "openai/gpt-4o-2024-05-13": "GPT-4o",
        "anthropic/claude-sonnet-4.5": "Claude",
        "qwen/qwen3-30b-a3b-instruct-2507": "Qwen-30b",
    }

    has_io_suffix = "(IO)" in model_name
    base_model = model_name.replace("(CoT)", "").replace("(IO)", "").strip()

    mapping = model_mappings.get(base_model)
    if isinstance(mapping, dict):
        variant_key = "without_cot" if has_io_suffix else "with_cot"
        return mapping[variant_key]
    if mapping:
        return mapping

    return model_name


def format_action_name(name: str) -> str:
    """Convert an internal action name to a human-readable display name."""
    if "_" in name:
        return name.replace("_", " ").title()
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return spaced.title() if spaced == name else spaced


def clean_action(raw: str) -> str:
    """Normalize serialized action labels to bare action names."""
    return raw.split(".")[-1] if "." in raw else raw


def sort_mechanisms(mechanisms: Iterable[str]) -> list[str]:
    """Sort mechanisms in the preferred display order."""
    preferred_order = [
        "NoMechanism",
        "Repetition",
        "ReputationFirstOrder",
        "Reputation",
        "Mediation",
        "Contracting",
    ]

    order_map = {name.lower(): i for i, name in enumerate(preferred_order)}

    def sort_key(mech):
        """Group known mechanisms before unknown mechanisms."""
        base_mech = mech.split(" (")[0]
        mech_lower = base_mech.lower()
        if mech_lower in order_map:
            return (0, order_map[mech_lower], mech)
        return (1, mech, "")

    return sorted(mechanisms, key=sort_key)


def sort_games(games: Iterable[str]) -> list[str]:
    """Sort games in the preferred display order."""
    preferred_order = [
        "PrisonersDilemma",
        "PublicGoods",
        "TravellersDilemma",
        "TrustGame",
        "StagHunt",
        "MatchingPennies",
    ]

    order_map = {name.lower(): i for i, name in enumerate(preferred_order)}

    def sort_key(game):
        """Group known games before unknown games."""
        game_lower = game.lower()
        if game_lower in order_map:
            return (0, order_map[game_lower])
        return (1, game)

    return sorted(games, key=sort_key)


def sort_agents(agents: Iterable[str]) -> list[str]:
    """Sort agents in the preferred display order."""

    def sort_key(agent):
        """Group known agent/model families in the paper display order."""
        agent_lower = agent.lower()

        if "claude" in agent_lower:
            return (0, agent)
        if "gemini" in agent_lower and "(cot)" in agent_lower:
            return (1, agent)
        if "gemini" in agent_lower:
            return (2, agent)
        if "gpt-5" in agent_lower:
            return (3, agent)
        if "gpt-4" in agent_lower:
            return (4, agent)
        if "qwen" in agent_lower:
            return (5, agent)
        return (6, agent)

    return sorted(agents, key=sort_key)


def to_snake_case(text: str) -> str:
    """Convert PascalCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", text)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def format_identifier_as_title(text: str) -> str:
    """Convert compact identifiers to title text with spaces."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1 \2", text)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1 \2", s1)
    return s2.strip()


def _action_self_payoffs(
    game_name: str, game_config: dict
) -> list[tuple[str, float]]:
    """Return [(action_name, self_payoff)] sorted NE to cooperative."""

    from coopeval.registry.game_registry import GAME_REGISTRY

    game_cls = GAME_REGISTRY[game_name]
    game = game_cls(**game_config["game"]["kwargs"])
    pairs = [
        (a.name, game.get_action_self_payoff(a))
        for a in game.action_class.game_actions()
    ]
    return sorted(pairs, key=lambda x: x[1])


def build_action_display_names(
    game_name: str, game_config: dict
) -> dict[str, str]:
    """Return {internal_action_name: display_name} for game actions."""

    from coopeval.registry.game_registry import GAME_REGISTRY

    game = GAME_REGISTRY[game_name](**game_config["game"]["kwargs"])
    if game_name == "TravellersDilemma":
        return {
            a.name: f"Claim {a.value}" for a in game.action_class.game_actions()
        }
    return {
        a.name: format_action_name(a.name)
        for a in game.action_class.game_actions()
    }


def build_action_color_map(
    game_name: str, game_config: dict
) -> dict[str, tuple]:
    """Return {action_name: rgba_color} for all actions, ordered by payoff."""

    from coopeval.visualization.analysis_utils import NormalizeScore
    from coopeval.script_utils.colors import CMAP_VMAX, CMAP_VMIN, custom_cmap

    pairs = _action_self_payoffs(game_name, game_config)
    normalizer = NormalizeScore(game_name, game_config["game"])
    return {
        name: custom_cmap(
            (normalizer.normalize(payoff) - CMAP_VMIN) / (CMAP_VMAX - CMAP_VMIN)
        )
        for name, payoff in pairs
    }
