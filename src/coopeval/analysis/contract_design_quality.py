"""Contract-design analysis for CoopEval contracting runs."""

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
from coopeval.mechanisms.contracting import Contract, Contracting
from coopeval.script_utils.display_helper import extract_agent_name


@dataclass
class ContractContext:
    """Per-game context cached for contract evaluation."""

    game: Game
    contracting: Contracting
    action_space: tuple[Action, ...]
    full_profiles: tuple[tuple[Action, ...], ...]
    coop_profile: tuple[Action, ...]


def get_action_space(game: Game) -> tuple[Action, ...]:
    """Return the playable action set, excluding mediator placeholders."""

    return tuple(
        action for action in game.action_class if not action.is_mediator
    )


def build_contract_context(game: Game) -> ContractContext:
    """Cache contracting-specific action space, cooperate profile, and mechanism."""
    action_space = get_action_space(game)
    coop_action = next(
        (action for action in action_space if action.is_cooperate_action())
    )
    num_players = game.num_players
    full_profiles = tuple(product(action_space, repeat=num_players))
    coop_profile = tuple(coop_action for _ in range(num_players))
    return ContractContext(
        game=game,
        contracting=Contracting(game),
        action_space=action_space,
        full_profiles=full_profiles,
        coop_profile=coop_profile,
    )


def build_contract_entries(
    run_dir: Path,
    config: dict[str, Any],
    payload: dict[str, Any],
    game: Game,
) -> Iterable[tuple[str, Contract]]:
    """Yield (agent, contract) tuples parsed from the JSON payload."""
    del run_dir, config  # Unused but kept for consistent signature.
    for player_name, contract_values in payload.items():
        contract_detail = contract_values["contract"]
        contract = Contract(
            {
                game.action_class.from_str(action_name): int(payoff)
                for action_name, payoff in contract_detail.items()
            }
        )
        agent_name = extract_agent_name(player_name)
        yield agent_name, contract


def evaluate_contract(
    contract: Contract, context: ContractContext
) -> tuple[bool, bool]:
    """Return whether cooperation is weakly dominant and a Nash equilibrium."""

    game = context.game
    contracting_mechanism = context.contracting
    action_space = context.action_space
    num_players = game.num_players
    full_profiles = context.full_profiles
    coop_profile = context.coop_profile

    @lru_cache(maxsize=None)
    def adjusted_payoffs(profile: tuple[Action, ...]) -> tuple[float, ...]:
        """Return contract-adjusted payoffs for an action profile."""
        base_payoffs = tuple(game.get_actions_payoff(profile))
        adjustments = contracting_mechanism.compute_contract_adjustments(
            profile, contract
        )
        return tuple(
            base + delta for base, delta in zip(base_payoffs, adjustments)
        )

    weak_dominance = True
    for player_idx in range(num_players):
        player_weak_dominance = True
        for action in action_space:
            if action.is_cooperate_action():
                continue
            never_worse = True
            strictly_better_once = False

            for incumbent_profile in full_profiles:
                profile_template = list(incumbent_profile)
                profile_template[player_idx] = coop_profile[player_idx]
                coop_payoff = adjusted_payoffs(tuple(profile_template))[
                    player_idx
                ]

                profile_template[player_idx] = action
                alt_payoff = adjusted_payoffs(tuple(profile_template))[
                    player_idx
                ]

                if alt_payoff - coop_payoff > EPSILON:
                    never_worse = False
                    break
                if coop_payoff - alt_payoff > EPSILON:
                    strictly_better_once = True

            if not never_worse or not strictly_better_once:
                player_weak_dominance = False
                break

        if not player_weak_dominance:
            weak_dominance = False
            break

    nash_equilibrium = True
    coop_payoffs = adjusted_payoffs(coop_profile)
    for player_idx in range(num_players):
        for action in action_space:
            if action.is_cooperate_action():
                continue
            deviated_profile = list(coop_profile)
            deviated_profile[player_idx] = action
            deviated_points = adjusted_payoffs(tuple(deviated_profile))[
                player_idx
            ]
            if deviated_points - coop_payoffs[player_idx] > EPSILON:
                nash_equilibrium = False
                break
        if not nash_equilibrium:
            break

    return weak_dominance, nash_equilibrium


CONTRACT_ANALYZER = DesignAnalyzer[Contract, ContractContext](
    name="Contracting",
    design_filename="contract_design.json",
    item_noun_singular="contract",
    item_noun_plural="contracts",
    build_designs=build_contract_entries,
    context_factory=build_contract_context,
    evaluate_design=evaluate_contract,
    figure_subdir="design_quality",
    figure_stem="contract_design_quality",
    figure_title_prefix="Contract Proposals",
)
