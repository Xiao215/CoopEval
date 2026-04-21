"""Matching Pennies zero-sum benchmark for head-to-head agent play."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence, override

from coopeval.agents.agent_manager import Agent
from coopeval.games.base import Action, GridGame, Move


class MatchingPenniesAction(Action):
    """Possible actions in Matching Pennies"""

    HEADS = "H"
    TAILS = "T"

    def is_cooperate_action(self) -> bool:
        """Return True if this action is treated as cooperation (arbitrary for zero-sum)."""
        return self == type(self).HEADS


class MatchingPennies(GridGame):
    """
    Matching Pennies environment that allows for one round of interaction
    between two LLM agents. This is a zero-sum game where players choose
    Heads or Tails.
    """

    def __init__(
        self,
        payoff_matrix: Mapping[str, Sequence[float]],
    ) -> None:
        super().__init__(
            payoff_matrix=payoff_matrix,
            action_class=MatchingPenniesAction,
            num_players=2,
            is_symmetric=True,
        )

    @override
    async def play(
        self,
        additional_info: list[str] | str,
        players: Sequence[Agent],
        action_map: Callable = lambda x: x,
    ) -> list[Move]:
        if len(players) != self.num_players:
            raise ValueError(
                f"Expected {self.num_players} agents, got {len(players)}."
            )
        player1, player2 = players

        if isinstance(additional_info, str):
            additional_info = [additional_info] * self.num_players

        players_decision = await self._collect_actions(players, additional_info)
        players_decision = self._apply_action_map(players_decision, action_map)
        pts1, pts2 = self.get_actions_payoff(
            (
                players_decision[player1][0],
                players_decision[player2][0],
            )
        )
        return [
            Move(
                player=player1,
                action=players_decision[player1][0],
                points=pts1,
                trace_id=players_decision[player1][1],
                mediated=players_decision[player1][2],
            ),
            Move(
                player=player2,
                action=players_decision[player2][0],
                points=pts2,
                trace_id=players_decision[player2][1],
                mediated=players_decision[player2][2],
            ),
        ]
