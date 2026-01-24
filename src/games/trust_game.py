from __future__ import annotations

import textwrap
from typing import Callable, Mapping, Sequence, override

from src.agents.agent_manager import Agent
from src.games.base import Action, Game, Move


class TrustGameAction(Action):
    """Available actions for the trust game."""

    GIVE = "G"
    KEEP = "K"


class TrustGame(Game):
    """Two-player trust game modelled as a simultaneous move game."""
    def __init__(
        self,
        payoff_matrix: Mapping[str, Sequence[float]],
    ) -> None:
        self.action_class = TrustGameAction
        self.raw_payoff_matrix = payoff_matrix
        self.payoff_matrix = self._parse_payoff_matrix(payoff_matrix)
        actions_block = "\n".join(
            [f"- {act.to_token()}" for act in self.action_class]
        )

        self.prompt_template = textwrap.dedent(
            """
        Setup:
        You are playing a decision-making game with another player.
        Your objective is to maximize your total points received in the game described in length below.

        Actions available to each player:
        {actions_block}

        Basic game rules:
        1. You and the other player each choose a probability for each action, simultaneously.
        2. After both decisions are locked in, the final action will be drawn from the probability distributions.
        3. Both players receive the points specified in the payoff description below.
        """
        )

        super().__init__(
            prompt=self.prompt_template.format(
                actions_block=actions_block,
            ),
            action_class=self.action_class,
            num_players=2,
            is_symmetric=False,
        )

        self.number_to_position = {1: "first", 2: "second"}

    def _payoff_description(self) -> tuple[str, str]:
        p1_lines = []
        p2_lines = []
        for (a, b), (pts_a, pts_b) in self.payoff_matrix.items():
            p1_lines.append(
                f"\t- If you choose {a.to_token()} and the other player chooses {b.to_token()}: "
                f"You get {pts_a} points, the other player gets {pts_b} points."
            )
            p2_lines.append(
                f"\t- If you choose {b.to_token()} and the other player chooses {a.to_token()}: "
                f"You get {pts_b} points, the other player gets {pts_a} points."
            )
        return "\n".join(p1_lines), "\n".join(p2_lines)

    def get_player_prompt(self, player_id: int) -> str:
        """Get prompt from specific player's perspective."""
        p1_desc, p2_desc = self._payoff_description()
        player_desc = p1_desc if player_id == 1 else p2_desc
        payoff_section = "\nPayoff description:\n" + player_desc
        return self.prompt + payoff_section + f"\nIn case player identification becomes relevant, you are playing in the position of Player {player_id} in this game.\n"

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

    def _parse_payoff_matrix(
        self,
        raw_payoff: Mapping[str, Sequence[float]],
    ) -> dict[tuple[Action, Action], tuple[float, float]]:
        """Convert a raw payoff matrix with string keys into typed action pairs."""
        payoffs = {}
        for key, (p1, p2) in raw_payoff.items():
            a1 = self.action_class(key[0])
            a2 = self.action_class(key[1])
            payoffs[(a1, a2)] = (p1, p2)
        return payoffs

    @override
    def add_mediator_action(self) -> None:
        super().add_mediator_action()
        self.payoff_matrix = self._parse_payoff_matrix(self.raw_payoff_matrix)
