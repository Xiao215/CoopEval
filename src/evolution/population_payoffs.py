"""Store and aggregate tournament payoffs for evolutionary dynamics."""

import math
import warnings
from collections import defaultdict
from itertools import permutations
from typing import Any, Sequence, TypeAlias

import numpy as np

from src.agents.agent_manager import Agent
from src.games.base import Move

ProfileKey: TypeAlias = tuple[int, ...]


class PopulationPayoffs:
    """Manage payoff tables while tracking seat-level outcomes.

    Payoffs are stored by unique match profiles (sorted UIDs). Aggregation
    by model type is performed lazily using the provided `players` list
    as the source of truth for agent identities.
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
                This is required to map UIDs to model types.
            discount: Geometric discount factor in (0, 1].
        """
        if not players:
            raise ValueError(
                "PopulationPayoffs requires a non-empty sequence of players."
            )

        self.discount = discount if discount is not None else 1.0
        if not 0.0 < self.discount <= 1.0:
            warnings.warn(
                f"Discount factor should be in (0, 1], got {self.discount}. "
                "Ensure this is intended."
            )

        # Single Source of Truth: Build mapping directly from players
        self._uid_to_model: dict[int, str] = {
            int(p.uid): str(p.model_type) for p in players
        }

        # Storage: Map sorted UIDs to a list of match arrays.
        # Structure: { (1, 2): [ Array(Rounds, Players), ... ] }
        self._profiles: dict[ProfileKey, np.ndarray] = defaultdict(list)

        # Cached payoff tensor (populated by build_payoff_tensor)
        self._payoff_tensor: np.ndarray | None = None
        self._tensor_model_types: list[str] | None = None

    def reset(self) -> None:
        """Clear all recorded matchup outcomes."""
        self._profiles.clear()

    def add_profile(self, moves_over_rounds: Sequence[Sequence[Move]]) -> None:
        """Record a single matchup outcome.

        Args:
            moves_over_rounds: List of Move objects, organized by round.
        """
        if not moves_over_rounds:
            raise ValueError("Cannot add empty moves list to payoff table")

        # 1. Determine the key (Canonical Sorted Order)
        # We assume moves_over_rounds[0] contains all players present in the match.
        uids_in_match = {int(move.uid) for move in moves_over_rounds[0]}
        # Sort to ensure column 0 is always lowest UID, column 1 is next, etc.
        key: ProfileKey = tuple(sorted(uids_in_match))

        # 2. Extract points into a shape (num_rounds, num_players)
        # We must respect the sorted order of 'key' when extracting points.
        round_points = []
        for round_moves in moves_over_rounds:
            # Create a quick lookup for this round
            uid_lookup = {int(m.uid): float(m.points) for m in round_moves}
            try:
                # Extract points in the strict order of 'key'
                ordered_points = [uid_lookup[uid] for uid in key]
            except KeyError as e:
                raise ValueError(
                    f"Inconsistent player UIDs across rounds. Missing: {e}"
                ) from e
            round_points.append(ordered_points)

        # 3. Store
        self._profiles[key].append(np.array(round_points, dtype=float))

    def _compute_discounted_average(self, payoffs: np.ndarray) -> np.ndarray:
        """Apply geometric discounting to a 3D payoff array (Matches, Rounds, Players).

        Returns:
            2D array of shape (Matches, Players)
        """
        _, num_rounds, _ = payoffs.shape
        d = self.discount

        if num_rounds == 0:
            raise ValueError("Empty payoffs array, cannot compute average")

        if num_rounds > 1 and d == 1.0:
            warnings.warn(
                "Discount is 1.0 but multiple rounds detected. "
                "Only the last round will be counted due to weight logic."
            )

        # Vectorized weight calculation
        rounds_idx = np.arange(num_rounds)
        weights = (1 - d) * (d**rounds_idx)
        # Fix the tail probability to ensure sum is exactly 1.0
        weights[-1] = d ** (num_rounds - 1)

        if not math.isclose(np.sum(weights), 1.0):
            raise ValueError(
                f"Discount weights sum to {np.sum(weights)}, expected 1.0"
            )

        # Broadcasting weights: (num_rounds, 1) to multiply against (matches, rounds, players)
        # We sum over axis 1 (rounds)
        weighted_sums = np.sum(payoffs * weights[None, :, None], axis=1)

        return weighted_sums

    def model_average_payoff(self) -> dict[str, float]:
        """
        Compute the average payoff of each model type in the population.
        """

        aggregated_payoffs: dict[str, list[float]] = defaultdict(list)
        for uids, payoffs in self._profiles.items():
            # Distribute the weighted payoff to each model type involved
            for uid, payoff in zip(uids, self._compute_discounted_average(payoffs)):
                model_type = self._uid_to_model[uid]
                aggregated_payoffs[model_type].append(payoff)

        average_payoff = {
            model_type: float(np.mean(np.array(payoffs)))
            for model_type, payoffs in aggregated_payoffs.items()
        }
        return average_payoff

    def build_payoff_tensor(self) -> None:
        """
        Build and cache a payoff tensor of tensor dimension equal to the number of players in the game, and indexed by model types.

        Stores the tensor in self._payoff_tensor and model types in self._tensor_model_types.
        This should be called once after all matchups are recorded and before calling fitness().
        """

        all_model_types = sorted(set(self._uid_to_model.values()))
        model_to_idx = {m: i for i, m in enumerate(all_model_types)}
        k = len(all_model_types)

        # Get number of players from any matchup
        n_players = len(next(iter(self._profiles.keys())))

        # Initialize tensor and counts
        tensor = np.zeros([k] * n_players)
        counts = np.zeros([k] * n_players)

        for uid_set, payoffs_array in self._profiles.items():
            uids = list(uid_set)

            # Get model types for each UID
            models = [self._uid_to_model[uid] for uid in uids]
            # Average payoffs per model type
            avg_per_uid = self._compute_discounted_average(payoffs_array)
            model_payoffs = defaultdict(list)
            for uid, payoff in zip(uids, avg_per_uid):
                model_payoffs[self._uid_to_model[uid]].append(payoff)
            avg_by_model = {m: np.mean(p) for m, p in model_payoffs.items()}

            # Fill all symmetric positions
            for perm in set(permutations(models)):
                focal_model = perm[0]
                indices = tuple(model_to_idx[m] for m in perm)
                tensor[indices] += avg_by_model[focal_model]
                counts[indices] += 1

        # Average if filled multiple times
        mask = counts > 0
        tensor[mask] /= counts[mask]

        # Cache the results
        self._payoff_tensor = tensor
        self._tensor_model_types = all_model_types

    def fitness(self, population: dict[str, float]) -> dict[str, float]:
        """
        Compute expected payoff for each model type against the current population.

        Uses einsum to compute: fitness[i] = Σ_{j,k,...} tensor[i,j,k,...] × pop[j] × pop[k] × ...
        This represents the expected payoff for a model i playing against
        opponents randomly sampled from the population distribution.

        Args:
            population: dict mapping model type to its probability.
        """
        if not math.isclose(sum(population.values()), 1.0):
            raise ValueError("Population probabilities must sum to 1.0")

        # Ensure tensor has been built
        assert self._payoff_tensor is not None and self._tensor_model_types is not None, \
            "Must call build_payoff_tensor() before fitness(). Tensor has not been built yet."

        tensor = self._payoff_tensor
        model_types = self._tensor_model_types
        n_players = tensor.ndim

        # Verify consistency between population and tensor model types
        assert set(model_types) == set(population.keys()), \
            f"Model types mismatch: tensor has {set(model_types)}, population has {set(population.keys())}"

        # Build population vector in the same order as tensor indices
        pop = np.array([population[m] for m in model_types])

        # Create einsum expression: 'ijk...,j,k,...->i'
        indices = ''.join(chr(ord('a') + i) for i in range(n_players))
        expr = indices + ',' + ','.join(indices[1:]) + '->' + indices[0]

        # Compute fitness via einsum
        fitness_vec = np.einsum(expr, tensor, *([pop] * (n_players - 1)))

        return {m: float(fitness_vec[i]) for i, m in enumerate(model_types)}

    def to_json(self) -> dict[str, Any]:
        """Serialize payoff records.

        Note: We store the current uid_to_model mapping in the JSON
        purely for debugging/inspection purposes. It is ignored by from_json.
        """
        serialized_matches = []

        for uids, match_list in sorted(self._profiles.items()):
            payoffs_data = [m.tolist() for m in match_list]
            serialized_matches.append(
                {"uids": list(uids), "payoffs": payoffs_data}
            )

        return {
            "discount": self.discount,
            "debug_uids_map": {
                str(k): v for k, v in self._uid_to_model.items()
            },  # for debugging only
            "matchups": serialized_matches,
        }

    @classmethod
    def from_json(
        cls,
        payload: dict[str, Any],
        players: Sequence[Agent],
    ) -> "PopulationPayoffs":
        """Reconstruct instance from JSON.

        Args:
            payload: The JSON data.
            players: The list of agents. This is REQUIRED to reconstruct
                     the uid-to-model mapping.
        """
        # We ignore 'debug_uids_map' from JSON and strictly use 'players'
        instance = cls(
            players=players,
            discount=payload.get("discount"),
        )

        for entry in payload.get("matchups", []):
            uids = tuple(sorted(entry["uids"]))
            raw_payoffs = entry["payoffs"]
            restored_arrays = [np.array(p, dtype=float) for p in raw_payoffs]
            instance._profiles[uids].extend(restored_arrays)

        return instance
