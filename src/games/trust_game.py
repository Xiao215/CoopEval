from __future__ import annotations

import textwrap
from typing import Callable, Mapping, Sequence

from src.agents.agent_manager import Agent
from src.games.base import Action, Game, Move


class TrustGameAction(Action):
    """Available actions for the trust game."""

    INVEST = "I"
    KEEP = "K"


class TrustGame(Game):
    """Two-player trust game modelled as a simultaneous move game."""

    def __init__(
        self,
        payoff_matrix: Mapping[str, Sequence[float]],
    ) -> None:
        self.payoff_matrix = self._parse_payoff_matrix(payoff_matrix)
        actions_block = "\n".join(
            [
                f"- {TrustGameAction.INVEST.to_token()} — Invest / trust the other player",
                f"- {TrustGameAction.KEEP.to_token()} — Keep your endowment",
            ]
        )

        self.prompt_template = textwrap.dedent(
            """
        You are a participant in a two-player trust game. Both players decide simultaneously
        whether to invest their endowment (trust) or keep it.

        Actions:
        {actions_block}

        Payoff outcomes:
        {payoff_description}
        """
        )

        super().__init__(
            prompt=self.prompt_template.format(
                actions_block=actions_block,
                payoff_description=self._payoff_description(),
            ),
            num_players=2,
            num_actions=len(TrustGameAction),
        )

    def _payoff_description(self) -> str:
        lines = []
        for (a, b), (pts_a, pts_b) in self.payoff_matrix.items():
            lines.append(
                f"  • If you choose {a.to_token()} and your counterpart chooses {b.to_token()}: "
                f"you get {pts_a} points, they get {pts_b} points."
            )
        return "\n".join(lines)

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

        results = self._collect_actions(
            players,
            additional_info,
        )
        action_indices = {uid: action_idx for uid, action_idx, _ in results}
        responses = {uid: resp for uid, _, resp in results}

        mapped_indices = action_map(action_indices)
        final_actions: dict[int, TrustGameAction] = {
            uid: TrustGameAction.from_index(action)
            for uid, action in mapped_indices.items()
        }

        uid1 = player1.uid
        uid2 = player2.uid
        pts1, pts2 = self.payoff_matrix[(final_actions[uid1], final_actions[uid2])]

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
