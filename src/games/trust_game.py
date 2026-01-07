from __future__ import annotations

import textwrap
from typing import Callable, Mapping, Sequence

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
        self.payoff_matrix = self._parse_payoff_matrix(payoff_matrix)
        actions_block = "\n".join(
            [f"- {act.to_token()}" for act in TrustGameAction]
        )

        self.prompt_template = textwrap.dedent(
            """
        Setup:
        You are playing a decision-making game with another player.
        Your objective is to maximize your points received in the game described below.

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
            num_players=2,
            num_actions=len(TrustGameAction),
        )

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
        return self.prompt + payoff_section + f"\nIn case player identification becomes relevant, you are Player {player_id} in this game.\n"

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

        for i, player in enumerate(players):
            player_payoff = self.get_player_prompt(player.player_id)
            player_payoff = player_payoff[len(self.prompt):]
            additional_info[i] = player_payoff + additional_info[i]

        results = self._collect_actions(
            players,
            additional_info,
        )
        action_indices = {uid: action_idx for uid, action_idx, _ in results}
        responses = {uid: resp for uid, _, resp in results}

        mapped_indices = action_map(action_indices)
        final_actions = {
            uid: TrustGameAction.from_index(action)
            for uid, action in mapped_indices.items()
        }

        uid1 = player1.uid
        uid2 = player2.uid
        pts1, pts2 = self.payoff_matrix[
            (final_actions[uid1], final_actions[uid2])
        ]

        return [
            Move(
                player_name=player1.name,
                uid=uid1,
                action=final_actions[uid1],
                points=pts1,
                response=responses[uid1],
            ),
            Move(
                player_name=player2.name,
                uid=uid2,
                action=final_actions[uid2],
                points=pts2,
                response=responses[uid2],
            ),
        ]

    @classmethod
    def _parse_payoff_matrix(
        cls,
        raw_payoff: Mapping[str, Sequence[float]],
    ) -> dict[tuple[TrustGameAction, TrustGameAction], tuple[float, float]]:
        """Convert a raw payoff matrix with string keys into typed action pairs."""
        payoffs = {}
        for key, (p1, p2) in raw_payoff.items():
            a1 = TrustGameAction(key[0])
            a2 = TrustGameAction(key[1])
            payoffs[(a1, a2)] = (p1, p2)
        return payoffs
