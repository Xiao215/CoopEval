from __future__ import annotations

import textwrap
from enum import Enum
from typing import Callable, Iterable, Mapping, Sequence, override

from src.agents.agent_manager import Agent
from src.games.base import Action, Game, Move


def build_travellers_action(claims: Iterable[int]) -> type[Action]:
    """Create an Action enum for the given claim schedule."""
    claims = tuple(claims)
    if not claims:
        raise ValueError("claims must be a non-empty tuple.")
    members = {f"CLAIM_{i}": str(claim) for i, claim in enumerate(claims)}
    return Enum("TravellersDilemmaAction", members, type=Action)  # type: ignore[misc]


class TravellersDilemma(Game):
    """
    Traveler's Dilemma for two players with a configurable action set.

    Actions represent claims (integers). When claims differ, both players
    receive the lower claim, with a bonus added to the lower claimant and a
    penalty subtracted from the higher claimant.
    """

    def __init__(
        self,
        *,
        min_claim: int,
        num_actions: int,
        claim_spacing: int,
        bonus: float,
    ) -> None:
        if num_actions < 2:
            raise ValueError("Travellers Dilemma requires at least 2 actions.")
        if claim_spacing <= 0:
            raise ValueError("claim_spacing must be a positive integer.")

        min_claim = int(min_claim)
        claim_spacing = int(claim_spacing)
        self.claims = tuple(
            min_claim + i * claim_spacing for i in range(num_actions)
        )
        self.bonus = float(bonus)

        self.action_class = build_travellers_action(self.claims)
        actions_block = "\n".join(
            f"- {act.to_token()}: correspond to the number {act.value}"
            for act in self.action_class
        )

        payoff_description = textwrap.dedent(
            f"""
            Suppose you choose number X and the other player chooses number Y.
                - If X = Y: you get X points, the other player gets Y (=X) points.
                - If X < Y: you get X + {self.bonus}, the other player gets X - {self.bonus}.
                - If X > Y: you get Y - {self.bonus}, the other player gets Y + {self.bonus}.
        """
        ).strip()

        prompt_template = textwrap.dedent(
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
            prompt=prompt_template.format(
                actions_block=actions_block,
                payoff_description=payoff_description,
            ),
            action_class=self.action_class,
            num_players=2,
            is_symmetric=True,
        )

        # Override mixed-strategy instruction to reflect multi-action correctly
        self.default_output_instruction = textwrap.dedent(
            """
        Instruction:
        - Choose a probability distribution over ALL actions each round.
        - Output must contain a valid JSON object at the end.
        - Keys must be the action names exactly as given.
        - Values must be integers between 0 and 100.
        - All values must sum to exactly 100.

        Format requirement:
        Return exactly one JSON object, for example:
        {"A0": <INT>, "A1": <INT>, ...}
        """
        )

    @override
    def play(
        self,
        additional_info: list[str] | str,
        players: Sequence[Agent],
        action_map: Callable = lambda x: x,
    ) -> list[Move]:
        assert len(players) == 2
        player1, player2 = players

        if isinstance(additional_info, str):
            additional_info = [additional_info] * 2

        players_decision = self._collect_actions(
            players,
            additional_info,
        )
        players_decision = action_map(players_decision)

        pts1, pts2 = self._calculate_payoffs(
            players_decision[player1][0], players_decision[player2][0]
        )

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

    def _calculate_payoffs(
        self, action_a: Action, action_b: Action
    ) -> tuple[float, float]:
        """Return payoffs for a pair of claims given the Traveller's Dilemma rules."""
        claim_a, claim_b = int(action_a.value), int(action_b.value)
        if claim_a == claim_b:
            value = float(claim_a)
            return value, value

        lower_claim = float(min(claim_a, claim_b))
        lower_payoff = lower_claim + self.bonus
        higher_payoff = lower_claim - self.bonus

        if claim_a < claim_b:
            return lower_payoff, higher_payoff
        return higher_payoff, lower_payoff

    @classmethod
    def parse_raw_payoff_matrix(
        cls,
        raw: Mapping[str, Sequence[float]],
        *,
        num_actions: int,
    ) -> dict[tuple[int, int], tuple[float, float]]:
        """
        Optional helper if a full payoff matrix is provided.

        Accepted key formats (i, j are action indices in [0, num_actions)):
        - "i,j" (e.g., "0,1")
        - "Ai,Aj" (e.g., "A0,A1")

        Values are two-element lists [p1, p2].
        """
        payoffs: dict[tuple[int, int], tuple[float, float]] = {}
        for key, val in raw.items():
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                raise ValueError(
                    f"Invalid payoff for {key!r}: expected [p1, p2]"
                )
            p1, p2 = float(val[0]), float(val[1])

            key = key.strip()
            if "," not in key:
                raise ValueError(
                    f"Invalid key {key!r}; expected 'i,j' or 'Ai,Aj' format."
                )
            a, b = [s.strip() for s in key.split(",", 1)]
            if a.startswith("A"):
                ai = int(a[1:])
                bi = int(b[1:])
            else:
                ai = int(a)
                bi = int(b)
            if not (0 <= ai < num_actions and 0 <= bi < num_actions):
                raise ValueError(
                    f"Action indices out of bounds in key {key!r}: "
                    f"got {(ai, bi)} with num_actions={num_actions}"
                )
            payoffs[(ai, bi)] = (p1, p2)
        return payoffs
