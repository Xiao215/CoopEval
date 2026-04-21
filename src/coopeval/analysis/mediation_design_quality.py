"""Mediation-design analysis for CoopEval mediation runs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Iterable

from .mechanism_design import (
    EPSILON,
    DesignAnalyzer,
)

from coopeval.games.base import Action, Game
from coopeval.mechanisms.mediation import Mediation, Mediator
from coopeval.script_utils.display_helper import extract_agent_name


@dataclass
class MediationContext:
    """Per-game context cached for mediator evaluation."""

    game: Game
    mediation: Mediation
    action_space: tuple[Action, ...]
    mediator_action: Action
    full_profiles: tuple[tuple[Action, ...], ...]
    mediator_profile: tuple[Action, ...]


def get_mediator_action(game: Game) -> Action:
    """Return the mediator delegate action that should already exist on the game."""
    for action in game.action_class:
        if action.is_mediator:
            return action
    raise ValueError(f"Game {game.__class__.__name__} lacks a mediator action.")


def build_mediation_context(game: Game) -> MediationContext:
    """Cache mediation-specific structures (action space, profiles, mechanism)."""
    action_space = tuple(game.action_class)
    mediator_action = get_mediator_action(game)
    num_players = game.num_players
    full_profiles = tuple(product(action_space, repeat=num_players))
    mediator_profile = tuple(mediator_action for _ in range(num_players))
    return MediationContext(
        game=game,
        mediation=Mediation(game),
        action_space=action_space,
        mediator_action=mediator_action,
        full_profiles=full_profiles,
        mediator_profile=mediator_profile,
    )


def build_mediator_entries(
    run_dir: Path,
    config: dict[str, Any],
    payload: dict[str, Any],
    game: Game,
) -> Iterable[tuple[str, Mediator]]:
    """Yield (agent, mediator) tuples parsed from the JSON payload."""
    del run_dir, config  # Unused but kept for consistent signature.
    for player_name, mediator_values in payload.items():
        mediator_spec = mediator_values["mediator"]
        mediator = Mediator(
            {
                int(num_delegating): game.action_class.from_str(action_label)
                for num_delegating, action_label in mediator_spec
            }
        )
        agent_name = extract_agent_name(player_name)
        yield agent_name, mediator


def evaluate_mediator(
    mediator: Mediator, context: MediationContext
) -> tuple[bool, bool]:
    """Return whether delegation is weakly dominant and a Nash equilibrium."""

    game = context.game
    num_players = game.num_players
    mediation_mechanism = context.mediation
    action_space = context.action_space
    mediator_action = context.mediator_action
    mediator_profile = context.mediator_profile
    full_profiles = context.full_profiles
    mediator_callable = mediation_mechanism.mediator_mapping(mediator)

    everyone_delegates_action = mediator.get(num_players)
    if (
        everyone_delegates_action is None
        or not everyone_delegates_action.is_cooperate_action()
    ):
        return False, False

    @lru_cache(maxsize=None)
    def resolved_payoffs(profile: tuple[Action, ...]) -> tuple[float, ...]:
        """Return mediation-resolved payoffs for an action profile."""
        actions = {idx: action for idx, action in enumerate(profile)}
        resolved = mediator_callable(actions)
        resolved_actions = tuple(resolved[idx] for idx in range(num_players))
        return tuple(game.get_actions_payoff(resolved_actions))

    weak_dominance = True
    for player_idx in range(num_players):
        player_weak_dominance = True
        for action in action_space:
            if action == mediator_action:
                continue
            never_worse = True
            strictly_better_once = False

            for incumbent_profile in full_profiles:
                profile_template = list(incumbent_profile)
                profile_template[player_idx] = mediator_action
                mediator_payoff = resolved_payoffs(tuple(profile_template))[
                    player_idx
                ]

                profile_template[player_idx] = action
                alt_payoff = resolved_payoffs(tuple(profile_template))[
                    player_idx
                ]

                if alt_payoff - mediator_payoff > EPSILON:
                    never_worse = False
                    break
                if mediator_payoff - alt_payoff > EPSILON:
                    strictly_better_once = True

            if not never_worse or not strictly_better_once:
                player_weak_dominance = False
                break

        if not player_weak_dominance:
            weak_dominance = False
            break

    nash_equilibrium = True
    mediator_payoffs = resolved_payoffs(mediator_profile)
    for player_idx in range(num_players):
        for action in action_space:
            if action == mediator_action:
                continue
            deviated_profile = list(mediator_profile)
            deviated_profile[player_idx] = action
            deviated_points = resolved_payoffs(tuple(deviated_profile))[
                player_idx
            ]
            if deviated_points - mediator_payoffs[player_idx] > EPSILON:
                nash_equilibrium = False
                break
        if not nash_equilibrium:
            break

    return weak_dominance, nash_equilibrium


MEDIATION_ANALYZER = DesignAnalyzer[Mediator, MediationContext](
    name="Mediation",
    design_filename="mediator_design.json",
    item_noun_singular="mediator design",
    item_noun_plural="mediator designs",
    build_designs=build_mediator_entries,
    context_factory=build_mediation_context,
    evaluate_design=evaluate_mediator,
    configure_game=lambda game: game.add_mediator_action(),
    figure_subdir="design_quality",
    figure_stem="mediation_design_quality",
    figure_title_prefix="Mediator Proposals",
)
