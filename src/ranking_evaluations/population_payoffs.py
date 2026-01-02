"""Store and aggregate tournament payoffs for evolutionary dynamics."""

import math
import warnings
from collections import defaultdict
from itertools import permutations, product
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
            if (
                normalized_mapping is not None
                and normalized_mapping != players_mapping
            ):
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

        # Cached payoff tensor (populated by build_payoff_tensor)
        self._payoff_tensor: np.ndarray | None = None
        self._tensor_model_types: list[str] | None = None

    def reset(self) -> None:
        """Clear all recorded matchup outcomes."""
        self._table.clear()

    def add_profile(
        self, moves_over_rounds: Sequence[Sequence[Move]]
    ) -> None:
        """Record a single matchup outcome consisting of the provided ``Move`` objects.

        Args:
            moves: the list of recorded moves for a specific matchup profile.
                The later entries would be more discounted.
        """
        if not moves_over_rounds:
            raise ValueError("Cannot add empty moves list to payoff table")

        # all uids must be consistent across rounds
        k = frozenset(int(move.uid) for move in moves_over_rounds[0])

        # stack points: shape (num_rounds, num_players)
        round_points = []
        for moves_per_round in moves_over_rounds:
            # keep order consistent with `uids`
            uid_to_points = {int(m.uid): float(m.points) for m in moves_per_round}
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

            weights = np.array(
                [(1 - d) * (d**i) for i in range(n)], dtype=float
            )
            weights[-1] = d ** (n - 1)
            if sum(weights) != 1.0:
                raise ValueError(
                    f"All discount weights must sum to 1.0, currently the sum is {sum(weights)}"
                )
            discounted_payoff = np.sum(
                weights[:, None] * profile_payoffs, axis=0
            )
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

    def build_payoff_tensor(self) -> None:
        """
        Build and cache a payoff tensor of tensor dimension equal to the number of players in the game, and indexed by model types.

        Stores the tensor in self._payoff_tensor and model types in self._tensor_model_types.
        This should be called once after all matchups are recorded and before calling fitness().
        """

        all_model_types = sorted(set(self.uids_to_model_types.values()))
        model_to_idx = {m: i for i, m in enumerate(all_model_types)}
        k = len(all_model_types)

        # Get number of players from any matchup
        n_players = len(next(iter(self._table.keys())))

        # Initialize tensor and counts
        tensor = np.zeros([k] * n_players)
        counts = np.zeros([k] * n_players)

        for uid_set, payoffs_array in self._table.items():
            uids = list(uid_set)

            # Get model types for each UID
            models = [self.uids_to_model_types[uid] for uid in uids]

            # Average payoffs per model type
            avg_per_uid = self._average_payoff(payoffs_array)
            model_payoffs = defaultdict(list)
            for uid, payoff in zip(uids, avg_per_uid):
                model_payoffs[self.uids_to_model_types[uid]].append(payoff)
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
                    permuted_strat[0], permuted_strat[player_idx] = permuted_strat[player_idx], permuted_strat[0]

                # Look up payoff from Player 1's tensor using permuted indices
                G[player_idx, joint_strat_idx] = tensor[tuple(permuted_strat)]

        return G

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
                int(uid): str(model_type)
                for uid, model_type in raw_mapping.items()
            }
        else:
            uid_mapping = {
                int(entry["uid"]): str(entry["model_type"])
                for entry in raw_mapping
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
                raise ValueError(
                    "Serialized payoff tensor must have three dimensions"
                )
            if payoffs.shape[-1] != len(stored_order):
                raise ValueError(
                    "Last dimension of payoff tensor must match number of uids"
                )

            key = frozenset(stored_order)
            uid_iteration_order = list(key)

            if stored_order != uid_iteration_order:
                reorder_index = {
                    uid: idx for idx, uid in enumerate(stored_order)
                }
                payoffs = payoffs[
                    ..., [reorder_index[uid] for uid in uid_iteration_order]
                ]

            instance._table[key] = payoffs

        return instance
