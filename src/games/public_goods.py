from __future__ import annotations

import textwrap
from typing import Callable, Sequence

from src.agents.agent_manager import Agent
from src.games.base import Action, Game, Move


class PublicGoodsAction(Action):
    """Possible actions in the Public Goods game"""

    CONTRIBUTE = "C"
    FREE_RIDE = "F"


class PublicGoods(Game):
    """
    Public Goods environment in which N agents each decide to either
    contribute their full endowment or free-ride. After all choices are made,
    total contributions are multiplied by a factor and split equally.
    """

    def __init__(
        self,
        *,
        num_players: int,
        multiplier: float,
    ) -> None:
        if num_players <= 1:
            raise ValueError("Public Goods must have at least 2 players.")
        if not (1.0 <= multiplier <= num_players):
            raise ValueError("Multiplier should be between 1 and num_players.")

        self.endowment = 1  # value of endowment does not matter
        self.multiplier = multiplier
        self.num_players = num_players

        actions_block = "\n".join(
            [f"- {act.to_token()}" for act in self.action_cls]
        )
        self.prompt_template = textwrap.dedent(
            """
        Setup:
        You are playing a decision-making game with {num_other_players} other players.
        Your objective is to maximize your points received in the game described in length below.

        Actions available to each player:
        {actions_block}

        Basic game rules:
        1. You and the other players each choose a probability for each action, simultaneously.
        2. After all decisions are locked in, the final action will be drawn from the probability distributions.
        3. All players receive the points specified in the payoff description below.

        Payoff description:
        Suppose k out of {num_players} players play {contribute_tok}.
            If player i plays {contribute_tok}, their payoff is: {multiplier} * k / {num_players}.
            If player i plays {free_ride_tok}, their payoff is: 1 + ({multiplier} * k / {num_players}).
        """
        )

        super().__init__(
            prompt=self.prompt_template.format(
                actions_block=actions_block,
                endowment=self.endowment,
                multiplier=self.multiplier,
                num_players=num_players,
                num_other_players=num_players - 1,
                contribute_tok=PublicGoodsAction.CONTRIBUTE.to_token(),
                free_ride_tok=PublicGoodsAction.FREE_RIDE.to_token(),
            ),
            num_players=num_players,
            is_symmetric=True,
        )

    @property
    def action_cls(self):
        return PublicGoodsAction

    def play(
        self,
        additional_info: list[str] | str,
        players: Sequence[Agent],
        action_map: Callable = lambda x: x,
    ) -> list[Move]:
        """
        Runs the Public Goods game: collects all actions, computes payoffs,
        and returns a list of Moves with each agent's action and points earned.
        """
        assert (
            len(players) == self.num_players
        ), f"Expected {self.num_players} agents, got {len(players)}."

        if isinstance(additional_info, str):
            additional_info = [additional_info] * self.num_players

        players_decision = self._collect_actions(
            players,
            additional_info,
            action_map,
        )

        share = self._calculate_share([v[0] for v in players_decision.values()])

        moves = []
        for player in players:
            moves.append(
                Move(
                    player_name=player.name,
                    uid=player.uid,
                    action=players_decision[player.uid][0],
                    points=(
                        share
                        if players_decision[player.uid][0]
                        == PublicGoodsAction.CONTRIBUTE
                        else self.endowment + share
                    ),
                    response=players_decision[player.uid][1],
                    trace_id=players_decision[player.uid][2],
                )
            )
        return moves

    def _calculate_share(self, actions: list[Action]) -> float:
        """
        Calculate the payoff for each agent based on their contributions.
        """

        contribution_count = sum(
            1 for v in actions if v == PublicGoodsAction.CONTRIBUTE
        )

        return (
            contribution_count
            * self.endowment
            * self.multiplier
            / self.num_players
        )
