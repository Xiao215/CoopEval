"""Store and aggregate matchup-based tournament payoffs for evolutionary dynamics."""

import math
from collections import defaultdict
from itertools import permutations, product
from typing import Any, Sequence, TypeAlias, override

import numpy as np

from src.agents.agent_manager import Agent
from src.games.base import Move
from src.ranking_evaluations.payoffs_base import PayoffsBase
from src.registry.agent_registry import create_agent

ProfileKey: TypeAlias = tuple[Agent, ...]


class MatchupPayoffs(PayoffsBase):
    """Manage payoff tables while tracking seat-level outcomes.

    Payoffs are stored by unique match profiles (sorted UIDs). Aggregation
    by agent type is performed lazily using the provided `players` list
    as the source of truth for agent identities.
    """

    def __init__(
        self,
        *,
        discount: float | None = None,
    ) -> None:
        """
        Args:
            players: List of Agents involved in the tournament.
                This is required to map UIDs to agent types.
            discount: Geometric discount factor in (0, 1].
        """
        super().__init__(discount=discount)

        # Storage: Map player tuples (in seat order) to a list of match arrays.
        # Structure: { (player_seat0, player_seat1): [ ndarray(payoff_seat0, payoff_seat1, ...), ... ] }
        self._profiles: dict[ProfileKey, list[np.ndarray]] = defaultdict(list)

        # Cached payoff tensor (populated by build_payoff_tensor)
        self._payoff_tensor: np.ndarray | None = None
        self._tensor_agent_types: list[str] | None = None

    @override
    def reset(self) -> None:
        """Clear all recorded matchup outcomes."""
        self._profiles.clear()
        self._payoff_tensor = None
        self._tensor_agent_types = None

    @override
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
            round_data = {m.player: float(m.points) for m in round_moves}
            # Preserve seat order - critical for building symmetric payoff tensor
            key = tuple(round_data.keys())
            ordered_points = [round_data[uid] for uid in key]
            match_accumulator[key].append(ordered_points)

        # Final Commit: Convert accumulated lists to arrays and store
        for key, history_list in match_accumulator.items():
            match_array = np.array(history_list, dtype=float)
            self._profiles[key].append(match_array)

    @override
    def agent_average_payoff(self) -> dict[str, float | None]:
        """
        Compute the average payoff of each agent type in the population.

        Returns:
            Dictionary mapping agent type to average payoff. For matchup-based
            mechanisms, all agents typically have observations (returns float).
        """

        aggregated_payoffs: dict[str, list[float]] = defaultdict(list)
        for players, match_list in self._profiles.items():
            for rounds_payoff in match_list:
                discounted_score = self._compute_discounted_average(
                    rounds_payoff
                )
                for i, player in enumerate(players):
                    aggregated_payoffs[player.agent_type].append(
                        discounted_score[i]
                    )
        return {
            agent_type: float(np.mean(np.array(scores)))
            for agent_type, scores in aggregated_payoffs.items()
        }

    def build_payoff_tensor(self) -> None:
        """
        Aggregate all recorded match histories into a canonical payoff tensor.

        The resulting tensor represents the expected payoff for a 'focal' agent
        (index 0) when playing against a specific combination of other players.
        Symmetry is enforced by filling all permutations of observed profiles.
        """
        if not self._profiles:
            raise ValueError(
                "No matches recorded. Cannot compute payoff tensor."
            )

        agent_types = sorted(
            {
                player.agent_type
                for players in self._profiles.keys()
                for player in players
            }
        )
        agent_type_to_idx = {m: i for i, m in enumerate(agent_types)}

        k = len(agent_types)
        # Determine N-players from the first available key
        n_players = len(next(iter(self._profiles.keys())))

        # Initialize Accumulators
        # tensor[i, j, ...] stores sum of payoffs for Agent_i vs Agent_j ...
        payoff_sums = np.zeros([k] * n_players, dtype=float)
        counts = np.zeros([k] * n_players, dtype=int)

        # First pass: Collect ALL observations grouped by (composition, focal_agent)
        composition_observations = defaultdict(list)

        for players, match_list in self._profiles.items():
            current_agent_types = [player.agent_type for player in players]

            # Process each match separately to avoid nested averaging
            for match_arr in match_list:
                discounted_scores = self._compute_discounted_average(match_arr)

                # Record each seat's observation
                for agent_type, score in zip(
                    current_agent_types, discounted_scores
                ):
                    # Use tuple as composition key
                    composition_key = tuple(current_agent_types)
                    composition_observations[
                        (composition_key, agent_type)
                    ].append(score)

        # Second pass: Sum all observations and track counts for averaging
        for (
            comp_key,
            focal_agent_type,
        ), scores in composition_observations.items():
            # Fill all permutations where focal_agent_type is in position 0
            for perm in set(permutations(comp_key)):
                if perm[0] == focal_agent_type:
                    indices = tuple(agent_type_to_idx[m] for m in perm)
                    # Add each observation individually to let final division handle averaging
                    for score in scores:
                        payoff_sums[indices] += score
                        counts[indices] += 1

        assert np.all(counts >= 1), "All tensor entries must have at least one observation."
        tensor = payoff_sums / counts

        self._payoff_tensor = tensor
        self._tensor_agent_types = agent_types

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
        Compute expected payoff for each agent type against the current population.

        Uses einsum to compute: fitness[i] = Σ_{j,k,...} tensor[i,j,k,...] x pop[j] x pop[k] x ...
        This represents the expected payoff for an agent i playing against
        other players randomly sampled from the population distribution.

        Args:
            population: dict mapping agent type to its probability.
        """
        if not math.isclose(sum(population.values()), 1.0):
            raise ValueError(
                f"Population probabilities must sum to 1.0, but got {sum(population.values())}"
            )

        # Ensure tensor has been built
        assert (
            self._payoff_tensor is not None
            and self._tensor_agent_types is not None
        ), "Must call build_payoff_tensor() before fitness(). Tensor has not been built yet."

        tensor = self._payoff_tensor
        agent_types = self._tensor_agent_types
        n_players = tensor.ndim

        # Verify consistency between population and tensor agent types
        assert set(agent_types) == set(
            population.keys()
        ), f"Agent types mismatch: tensor has {set(agent_types)}, population has {set(population.keys())}"

        # Build population vector in the same order as tensor indices
        pop = np.array([population[agent_type] for agent_type in agent_types])

        # Create einsum expression: 'ijk...,j,k,...->i'
        indices = ''.join(chr(ord('a') + i) for i in range(n_players))
        expr = indices + ',' + ','.join(indices[1:]) + '->' + indices[0]

        # Compute fitness via einsum
        fitness_vec = np.einsum(expr, tensor, *([pop] * (n_players - 1)))

        return {
            agent_type: float(fitness_vec[i])
            for i, agent_type in enumerate(agent_types)
        }

    @override
    def to_json(self) -> dict[str, Any]:
        """Serialize payoff records.

        Note: We store the current uid_to_agent mapping and player configs
        in the JSON so from_json can reconstruct when players are not provided.
        """
        serialized_profile = []

        for players, match_list in sorted(self._profiles.items()):
            payoffs_data = [m.tolist() for m in match_list]
            serialized_profile.append(
                {
                    "players": players,
                    "payoffs": payoffs_data,
                }
            )

        return {
            "discount": self.discount,
            "profile": serialized_profile,
        }

    @classmethod
    @override
    def from_json(
        cls,
        json_data: dict[str, Any],
    ) -> "MatchupPayoffs":
        """Reconstruct instance from JSON."""
        instance = cls(
            discount=json_data["discount"],
        )

        for entry in json_data["profile"]:
            players = tuple(create_agent(p_data) for p_data in entry["players"])
            raw_payoffs = entry["payoffs"]
            restored_arrays = [np.array(p, dtype=float) for p in raw_payoffs]
            instance._profiles[players].extend(restored_arrays)

        return instance
