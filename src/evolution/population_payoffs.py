"""Store and aggregate tournament payoffs for evolutionary dynamics."""

import math
import warnings
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from src.agents.agent_manager import Agent
from src.games.base import Move


class PopulationPayoffs:
    """Manage payoff tables while tracking seat-level outcomes."""

    def __init__(
        self,
        players: Sequence[Agent] | None = None,
        *,
        discount: float | None = None,
        uids_to_model_types: Mapping[int, str] | None = None,
    ) -> None:
        if players is None and uids_to_model_types is None:
            raise ValueError(
                "PopulationPayoffs requires either players or uids_to_model_types."
            )

        normalized_mapping = (
            {
                int(uid): str(model_type)
                for uid, model_type in uids_to_model_types.items()
            }
            if uids_to_model_types is not None
            else None
        )

        if players is not None:
            players_mapping = {int(p.uid): str(p.model_type) for p in players}
            if normalized_mapping is not None and normalized_mapping != players_mapping:
                raise ValueError(
                    "Provided players and uids_to_model_types contain inconsistent entries."
                )
            # A game could have multiple players with the same model type.
            self.uids_to_model_types = normalized_mapping or players_mapping
        else:
            if normalized_mapping is None:
                raise ValueError(
                    "PopulationPayoffs requires uids_to_model_types when players is None."
                )
            # A game could have multiple players with the same model type.
            self.uids_to_model_types = normalized_mapping

        self.discount = discount if discount is not None else 1.0
        if not 0.0 < self.discount <= 1.0:
            warnings.warn(
                f"Discount factor should be in (0, 1], got {self.discount}. "
                "Please make sure this is intended."
            )

        # Maps frozenset of uids to a 3D array of shape (num_profiles, num_rounds, num_players)
        # Num profiles would be >1 for multiple recorded matchups with single round match up such as
        # Mediation, Contracting and Disarmament.
        self._table: dict[
            frozenset[int],
            np.ndarray,
        ] = {}

    def reset(self) -> None:
        """Clear all recorded matchup outcomes."""
        self._table.clear()

    def add_profile(self, moves_over_rounds: list[list[Move]]) -> None:
        """Record a single matchup outcome consisting of the provided ``moves``.

        Args:
            moves: the list of recorded moves for a specific matchup profile.
                The later entries would be more discounted.
        """
        if not moves_over_rounds:
            raise ValueError("Cannot add empty moves list to payoff table")

        # all uids must be consistent across rounds
        k = frozenset(move.uid for move in moves_over_rounds[0])

        # stack points: shape (num_rounds, num_players)
        round_points = []
        for moves_per_round in moves_over_rounds:
            # keep order consistent with `uids`
            uid_to_points = {m.uid: m.points for m in moves_per_round}
            round_points.append([uid_to_points[uid] for uid in k])

        profile = np.array(round_points, dtype=float)  # shape (R, P)

        # store in table: append as new profile (flattened or 2D)
        if k not in self._table:
            self._table[k] = profile[None, ...]  # add profile dimension
        else:
            self._table[k] = np.vstack([self._table[k], profile[None, ...]])

    def _average_payoff(self, payoffs: np.ndarray) -> np.ndarray:
        """Apply geometric discounting over the payoff sequence, and average over all profiles ."""
        num_profiles, _, num_players = payoffs.shape

        discounted_payoffs = np.empty((num_profiles, num_players), dtype=float)
        for i, profile_payoffs in enumerate(payoffs):
            n = len(profile_payoffs)
            if n == 0:
                raise ValueError(
                    "Empty payoffs array, cannot compute discounted average"
                )

            d = self.discount
            if n > 1 and d == 1.0:
                warnings.warn(
                    "Discount factor is currently 1, but your payoff record indicates multiple rounds. "
                    "This means only the last round payoff will be counted. "
                    "Please make sure this is intended."
                )

            weights = np.array([(1 - d) * (d**i) for i in range(n)], dtype=float)
            weights[-1] = d ** (n - 1)
            if sum(weights) != 1.0:
                raise ValueError(
                    f"All discount weights must sum to 1.0, currently the sum is {sum(weights)}"
                )
            discounted_payoff = np.sum(weights[:, None] * profile_payoffs, axis=0)
            discounted_payoffs[i] = discounted_payoff
        return discounted_payoffs.mean(axis=0)

    def model_average_payoff(self) -> dict[str, float]:
        """
        Compute the average payoff of each model type in the population.

        """

        aggregated_payoffs: dict[str, list[float]] = defaultdict(list)
        for uids, payoffs in self._table.items():
            # Distribute the weighted payoff to each model type involved
            for uid, payoff in zip(uids, self._average_payoff(payoffs)):
                model_type = self.uids_to_model_types[uid]
                aggregated_payoffs[model_type].append(payoff)

        average_payoff = {
            model_type: float(np.mean(np.array(payoffs)))
            for model_type, payoffs in aggregated_payoffs.items()
        }
        return average_payoff

    def fitness(self, population: dict[str, float]) -> dict[str, float]:
        """
        Compute the fitness of each model type in the population.

        Args:
            population: dict mapping model type to its probability.
                Note the key here would be the model type,
                not the unique id.
        """
        if not math.isclose(sum(population.values()), 1.0):
            raise ValueError("Population probabilities must sum to 1.0")

        model_average_payoff = self.model_average_payoff()

        # Weight the average payoff by the probability of this model type
        fitness = {
            model_type: model_average_payoff[model_type] * prob
            for model_type, prob in population.items()
        }

        return fitness

    def to_json(self) -> dict[str, Any]:
        """Serialize payoff records into a JSON-friendly dictionary."""

        serialized_table = []
        for uid_set, payoffs in self._table.items():
            uid_order = list(uid_set)
            serialized_table.append(
                {
                    "uids": [int(uid) for uid in uid_order],
                    "payoffs": payoffs.tolist(),
                }
            )

        return {
            "discount": float(self.discount),
            "uids_to_model_types": {
                str(uid): model_type
                for uid, model_type in self.uids_to_model_types.items()
            },
            "matchups": serialized_table,
        }

    @classmethod
    def from_json(
        cls,
        payload: dict[str, Any],
        *,
        players: Sequence[Agent] | None = None,
    ) -> "PopulationPayoffs":
        """Reconstruct a ``PopulationPayoffs`` instance from serialized data."""

        discount = float(payload.get("discount", 1.0))

        raw_mapping = payload.get("uids_to_model_types", {})
        if isinstance(raw_mapping, dict):
            uid_mapping = {
                int(uid): str(model_type) for uid, model_type in raw_mapping.items()
            }
        else:
            uid_mapping = {
                int(entry["uid"]): str(entry["model_type"]) for entry in raw_mapping
            }

        instance = cls(
            players=players,
            discount=discount,
            uids_to_model_types=uid_mapping,
        )

        instance._table.clear()
        for matchup in payload.get("matchups", []):
            stored_order = [int(uid) for uid in matchup.get("uids", [])]
            payoffs = np.array(matchup.get("payoffs", []), dtype=float)

            if payoffs.ndim != 3:
                raise ValueError("Serialized payoff tensor must have three dimensions")
            if payoffs.shape[-1] != len(stored_order):
                raise ValueError(
                    "Last dimension of payoff tensor must match number of uids"
                )

            key = frozenset(stored_order)
            uid_iteration_order = list(key)

            if stored_order != uid_iteration_order:
                reorder_index = {uid: idx for idx, uid in enumerate(stored_order)}
                payoffs = payoffs[
                    ..., [reorder_index[uid] for uid in uid_iteration_order]
                ]

            instance._table[key] = payoffs

        return instance
