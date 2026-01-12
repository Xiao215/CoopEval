"""Track payoffs for reputation-based mechanisms where matchups are history-dependent."""

from collections import defaultdict
from typing import Any, Sequence, override

import numpy as np

from src.agents.agent_manager import Agent
from src.games.base import Move
from src.ranking_evaluations.payoffs_base import PayoffsBase
from src.registry.agent_registry import create_agent


class ReputationPayoffs(PayoffsBase):
    """Track payoffs for reputation-based mechanisms.

    Unlike matchup-based mechanisms, reputation mechanisms cannot evaluate
    matchups in isolation since outcomes depend on the entire interaction history.
    This class only tracks aggregate statistics per model type.
    """

    def __init__(
        self,
        *,
        discount: float | None = None,
    ) -> None:
        """
        Args:
            discount: Geometric discount factor in (0, 1].
        """
        super().__init__(discount=discount)

        # Storage: All points grouped by Player
        # Structure: {player: list of points over all rounds}
        self._profiles: dict[Agent, list[float]] = defaultdict(list)

    @override
    def reset(self) -> None:
        """Clear all recorded payoff outcomes."""
        self._profiles.clear()

    @override
    def add_profile(self, moves_over_rounds: Sequence[Sequence[Move]]) -> None:
        """
        Record match outcomes without grouping by matchup.

        For reputation mechanisms, we just track all moves by UID.

        Args:
            moves_over_rounds: Sequence of rounds, where each round is a sequence of moves.
        """
        for round_moves in moves_over_rounds:
            for move in round_moves:
                self._profiles[move.player].append(move.points)

    def model_average_payoff(self) -> dict[str, float | None]:
        """
        Compute the average discounted payoff of each model type across all rounds.

        Returns:
            Dictionary mapping model type to average discounted payoff, or None
            for models that were never drawn.
        """
        # Aggregate payoffs by model type
        model_payoffs: dict[str, list[float]] = defaultdict(list)

        for player, points_list in self._profiles.items():
            model_type = player.model_type

            if not points_list:
                continue

            # Extract payoffs
            payoffs_array = np.array(points_list, dtype=float)

            # Reshape to (num_rounds, 1) for discounting
            payoffs_2d = payoffs_array.reshape(-1, 1)

            # Apply discounting
            discounted = self._compute_discounted_average(payoffs_2d)

            # Extract the single value
            model_payoffs[model_type].append(float(discounted[0]))

        # Average across all instances of each model
        # Include all models from _uid_to_model, with None for those never drawn
        all_models = set(player.model_type for player in self._profiles.keys())
        return {
            model_type: float(np.mean(model_payoffs[model_type])) if model_type in model_payoffs else None
            for model_type in all_models
        }

    @override
    def to_json(self) -> dict[str, Any]:
        """Serialize payoff records."""
        serialized_profile = []

        for player, points_list in sorted(self._profiles.items()):
            serialized_profile.append(
                {
                    "player": player,
                    "points": points_list,
                }
            )

        return {"discount": self.discount, "profile": serialized_profile}

    @classmethod
    @override
    def from_json(
        cls,
        json_data: dict[str, Any],
    ) -> "ReputationPayoffs":
        """Reconstruct instance from JSON.

        Args:
            json_data: JSON data from to_json()

        Returns:
            Reconstructed ReputationPayoffs instance.
        """
        instance = cls(
            discount=json_data["discount"],
        )

        for entry in json_data["profile"]:
            player = create_agent(entry["player"])
            for point in entry["points"]:
                instance._profiles[player].append(float(point))
        return instance
