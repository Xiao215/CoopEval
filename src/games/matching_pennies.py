from __future__ import annotations

import textwrap
from typing import Callable, Mapping, Sequence

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
        actions_block = "\n".join(
            [f"- {act.to_token()}" for act in self.action_cls]
        )
        self.prompt_template = textwrap.dedent(
            """
        Setup:
        You are playing a decision-making game with another player.
        Your objective is to maximize your points received in the game described in length below.

        Actions available to each player:
        {actions_block}

        Basic game rules:
        1. You and the other player each choose a probability for each action, simultaneously.
        2. After both decisions are locked in, the final action will be drawn from the probability distributions.
        3. Both players receive the points specified in the payoff description below.

        Payoff description:
        {payoff_description}
        """
        )

        super().__init__(
            prompt=self.prompt_template.format(
                actions_block=actions_block,
                payoff_description=self._payoff_description(),
            ),
            payoff_matrix=payoff_matrix,
            num_players=2,
            is_symmetric=True,
        )

    @property
    def action_cls(self):
        return MatchingPenniesAction

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
            action_map,
        )

        uid1 = player1.uid
        uid2 = player2.uid
        pts1, pts2 = self.payoff_matrix[
            (
                players_decision[uid1][0],
                players_decision[uid2][0],
            )
        ]
        return [
            Move(
                player_name=player1.name,
                uid=uid1,
                action=players_decision[uid1][0],
                points=pts1,
                response=players_decision[uid1][1],
                trace_id=players_decision[uid1][2],
            ),
            Move(
                player_name=player2.name,
                uid=uid2,
                action=players_decision[uid2][0],
                points=pts2,
                response=players_decision[uid2][1],
                trace_id=players_decision[uid2][2],
            ),
        ]
