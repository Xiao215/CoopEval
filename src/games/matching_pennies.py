from __future__ import annotations

from typing import Callable, Mapping, Sequence, override

from src.agents.agent_manager import Agent
from src.games.base import Action, GridGame, Move


class MatchingPenniesAction(Action):
    """Possible actions in Matching Pennies"""

    HEADS = "H"
    TAILS = "T"


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
    def play(
        self,
        additional_info: list[str] | str,
        players: Sequence[Agent],
        action_map: Callable = lambda x: x,
    ) -> list[Move]:
        assert len(players) == self.num_players
        player1, player2 = players

        if isinstance(additional_info, str):
            additional_info = [additional_info] * self.num_players

        players_decision = self._collect_actions(
            players,
            additional_info,
        )

        players_decision = action_map(players_decision)
        pts1, pts2 = self.payoff_matrix[
            (
                players_decision[player1][0],
                players_decision[player2][0],
            )
        ]
        return [
            Move(
                player=player1,
                action=players_decision[player1][0],
                points=pts1,
                response=players_decision[player1][1],
                trace_id=players_decision[player1][2],
                mediated=players_decision[player1][3],
            ),
            Move(
                player=player2,
                action=players_decision[player2][0],
                points=pts2,
                response=players_decision[player2][1],
                trace_id=players_decision[player2][2],
                mediated=players_decision[player2][3],
            ),
        ]
