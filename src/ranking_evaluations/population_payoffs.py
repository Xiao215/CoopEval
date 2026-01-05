"""Store and aggregate tournament payoffs for evolutionary dynamics."""

import math
import warnings
from collections import defaultdict
from itertools import permutations, product
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
        # Structure: { (1, 2): [ ndarray(Point1, Points2, ...), ... ] }
        self._profiles: dict[ProfileKey, list[np.ndarray]] = defaultdict(list)

        # Cached payoff tensor (populated by build_payoff_tensor)
        self._payoff_tensor: np.ndarray | None = None
        self._tensor_model_types: list[str] | None = None

    def reset(self) -> None:
        """Clear all recorded matchup outcomes."""
        self._profiles.clear()
        self._payoff_tensor = None
        self._tensor_model_types = None

    def add_profile(self, moves_over_rounds: Sequence[Sequence[Move]]) -> None:
        """
        Record match outcomes.

        The call to this method should be intended to record all the entire sequence of matches, included repetitive rounds.
        Multiple calls to this method should be intended for averaging over multiple independent batches of matches.
        """
        if not moves_over_rounds:
            raise ValueError("Cannot add empty moves list to payoff table")

        # Since moves_over_rounds could contain different matchup profiles,
        # we need to accumulate them first by profile key.
        match_accumulator: dict[ProfileKey, list[list[float]]] = defaultdict(
            list
        )

        for round_moves in moves_over_rounds:
            round_data = {int(m.uid): float(m.points) for m in round_moves}
            key = tuple(sorted(round_data.keys()))
            ordered_points = [round_data[uid] for uid in key]
            match_accumulator[key].append(ordered_points)

        # Final Commit: Convert accumulated lists to arrays and store
        for key, history_list in match_accumulator.items():
            match_array = np.array(history_list, dtype=float)
            self._profiles[key].append(match_array)

    def _compute_discounted_average(self, payoffs: np.ndarray) -> np.ndarray:
        """Apply geometric discounting to a 2D payoff array (Rounds, Players).

        Returns:
            1D array of shape (Players,)
        """
        num_rounds, _ = payoffs.shape
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
        weighted_sums = np.sum(payoffs * weights[:, None], axis=0)

        return weighted_sums

    def model_average_payoff(self) -> dict[str, float]:
        """
        Compute the average payoff of each model type in the population.
        """

        aggregated_payoffs: dict[str, list[float]] = defaultdict(list)
        for uids, payoff_list in self._profiles.items():
            for rounds_payoff in payoff_list:
                discounted_score = self._compute_discounted_average(
                    rounds_payoff
                )
                for i, uid in enumerate(uids):
                    model_type = self._uid_to_model[uid]
                    aggregated_payoffs[model_type].append(discounted_score[i])
        return {
            model_type: float(np.mean(np.array(scores)))
            for model_type, scores in aggregated_payoffs.items()
        }

    def build_payoff_tensor(self) -> None:
        """
        Aggregate all recorded match histories into a canonical payoff tensor.

        The resulting tensor represents the expected payoff for a 'focal' model
        (index 0) when playing against a specific combination of opponents.
        Symmetry is enforced by filling all permutations of observed profiles.
        """
        if not self._profiles:
            raise ValueError(
                "No matches recorded. Cannot compute payoff tensor."
            )

        model_types = sorted(set(self._uid_to_model.values()))
        model_to_idx = {m: i for i, m in enumerate(model_types)}

        k = len(model_types)
        # Determine N-players from the first available key
        n_players = len(next(iter(self._profiles.keys())))

        # Initialize Accumulators
        # tensor[i, j, ...] stores sum of payoffs for Model_i vs Model_j ...
        payoff_sums = np.zeros([k] * n_players, dtype=float)
        counts = np.zeros([k] * n_players, dtype=int)

        # Process Each Unique Profile (Specific Set of UIDs)
        for profile_key, match_list in self._profiles.items():
            uids = list(profile_key)

            # Calculate Average Payoff per Seat for this Profile
            match_scores = []
            for match_arr in match_list:
                s = self._compute_discounted_average(match_arr)
                match_scores.append(s)

            profile_avg_scores = np.mean(match_scores, axis=0)
            current_models = [self._uid_to_model[uid] for uid in uids]

            model_payoff_map = defaultdict(list)
            for score, model in zip(profile_avg_scores, current_models):
                model_payoff_map[model].append(score)

            avg_by_model = {
                m: np.mean(vals) for m, vals in model_payoff_map.items()
            }

            # Fill all permutations to ensure the tensor is seat-agnostic
            for perm in set(permutations(current_models)):
                indices = tuple(model_to_idx[m] for m in perm)
                focal_model = perm[0]

                payoff_sums[indices] += avg_by_model[focal_model]
                counts[indices] += 1

        with np.errstate(invalid="ignore"):
            tensor = np.divide(payoff_sums, counts, where=counts > 0)

        self._payoff_tensor = tensor
        self._tensor_model_types = model_types

    def build_full_payoff_tensor(self) -> np.ndarray:
        """
        Expand Player 1's payoff tensor to all N players for symmetric games.

        Uses the permutation rule: G_p(a_1, ..., a_p, ..., a_N) = G_1(a_p, a_2, ..., a_{p-1}, a_1, a_{p+1}, ..., a_N)

        Returns:
            np.ndarray of shape (N, S^N) where:
            - N is the number of players
            - S is the number of strategies per player
            - G[p, i] is the payoff for player p at the i-th joint strategy
        """
        if self._payoff_tensor is None:
            self.build_payoff_tensor()
        assert self._payoff_tensor is not None

        tensor = self._payoff_tensor
        n_players = tensor.ndim
        n_strategies = tensor.shape[0]

        # Generate all joint strategies as tuples
        joint_strategies = list(product(range(n_strategies), repeat=n_players))
        n_joint_strategies = len(joint_strategies)

        # Initialize full payoff matrix: G[player, joint_strategy_index]
        G = np.zeros((n_players, n_joint_strategies), dtype=float)

        # Fill payoffs for each player
        for player_idx in range(n_players):
            for joint_strat_idx, joint_strat in enumerate(joint_strategies):
                # Apply permutation rule to get payoff for this player
                # G_p(a_1, ..., a_p, ..., a_N) = G_1(a_p, a_2, ..., a_{p-1}, a_1, a_{p+1}, ..., a_N)

                # Convert joint_strategy tuple to list for manipulation
                js = list(joint_strat)

                # Create permuted indices: swap position 0 (Player 1) with position player_idx
                permuted_strat = js.copy()
                if player_idx != 0:
                    # Swap: position 0 gets player_idx's strategy, position player_idx gets position 0's strategy
                    permuted_strat[0], permuted_strat[player_idx] = (
                        permuted_strat[player_idx],
                        permuted_strat[0],
                    )

                # Look up payoff from Player 1's tensor using permuted indices
                G[player_idx, joint_strat_idx] = tensor[tuple(permuted_strat)]

        return G

    def fitness(self, population: dict[str, float]) -> dict[str, float]:
        """
        Compute expected payoff for each model type against the current population.

        Uses einsum to compute: fitness[i] = Σ_{j,k,...} tensor[i,j,k,...] x pop[j] x pop[k] x ...
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
