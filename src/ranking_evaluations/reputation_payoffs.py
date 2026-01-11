"""Track payoffs for reputation-based mechanisms where matchups are history-dependent."""

from collections import defaultdict
from typing import Any, Sequence

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
        players: Sequence[Agent],
        *,
        discount: float | None = None,
    ) -> None:
        """
        Args:
            players: List of Agents involved in the tournament.
            discount: Geometric discount factor in (0, 1].
        """
        super().__init__(players, discount=discount)

        # Storage: All moves grouped by UID
        # Structure: {uid: list of (round_idx, points) tuples}
        self._moves_by_uid: dict[int, list[tuple[int, float]]] = defaultdict(list)

    def reset(self) -> None:
        """Clear all recorded move outcomes."""
        self._moves_by_uid.clear()

    def add_profile(self, moves_over_rounds: Sequence[Sequence[Move]]) -> None:
        """
        Record match outcomes without grouping by matchup.

        For reputation mechanisms, we just track all moves by UID.

        Args:
            moves_over_rounds: Sequence of rounds, where each round is a sequence of moves.
        """
        if not moves_over_rounds:
            raise ValueError("Cannot add empty moves list to payoff table")

        for round_idx, round_moves in enumerate(moves_over_rounds, 1):
            for move in round_moves:
                self._moves_by_uid[int(move.uid)].append((round_idx, float(move.points)))

    def model_average_payoff(self) -> dict[str, float | None]:
        """
        Compute the average discounted payoff of each model type across all rounds.

        Returns:
            Dictionary mapping model type to average discounted payoff, or None
            for models that were never drawn.
        """
        if not self._moves_by_uid:
            return {model: None for model in set(self._uid_to_model.values())}

        # Aggregate payoffs by model type
        model_payoffs: dict[str, list[float]] = defaultdict(list)

        for uid, move_list in self._moves_by_uid.items():
            model_type = self._uid_to_model[uid]

            if not move_list:
                continue

            # Moves should already be ordered by round index
            assert all(
                move_list[i][0] < move_list[i + 1][0]
                for i in range(len(move_list) - 1)
            ), "move_list must be sorted by round index. maybe you added multiple mechanism experiments into this object?"

            # Extract payoffs
            payoffs_array = np.array([points for _, points in move_list], dtype=float)

            # Reshape to (num_rounds, 1) for discounting
            payoffs_2d = payoffs_array.reshape(-1, 1)

            # Apply discounting
            discounted = self._compute_discounted_average(payoffs_2d)

            # Extract the single value
            model_payoffs[model_type].append(float(discounted[0]))

        # Average across all instances of each model
        # Include all models from _uid_to_model, with None for those never drawn
        all_models = set(self._uid_to_model.values())
        return {
            model_type: float(np.mean(model_payoffs[model_type])) if model_type in model_payoffs else None
            for model_type in all_models
        }

    def to_json(self) -> dict[str, Any]:
        """Serialize payoff records.

        Returns:
            JSON-serializable dictionary.
        """
        serialized_moves = []

        for uid, move_list in sorted(self._moves_by_uid.items()):
            serialized_moves.append({
                "uid": uid,
                "moves": [{"round": round_idx, "points": points} for round_idx, points in move_list]
            })

        return {
            "discount": self.discount,
            "player_configs": self._player_configs,
            "debug_uids_map": {
                str(k): v for k, v in self._uid_to_model.items()
            },  # for debugging only
            "moves": serialized_moves,
        }

    @classmethod
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
        players = [create_agent(cfg) for cfg in json_data["player_configs"]]
        instance = cls(
            players=players,
            discount=json_data["discount"],
        )

        for entry in json_data.get("moves", []):
            uid = entry["uid"]
            for move_record in entry["moves"]:
                round_idx = move_record["round"]
                points = move_record["points"]
                instance._moves_by_uid[uid].append((round_idx, points))

        return instance
