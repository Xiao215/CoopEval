"""Store and aggregate tournament payoffs for evolutionary dynamics."""

import math
import warnings
from collections import defaultdict
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
        self._profiles: dict[ProfileKey, list[np.ndarray]] = defaultdict(list)

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
        """Compute the average payoff of each model type in the population."""
        aggregated_payoffs: dict[str, list[float]] = defaultdict(list)

        for uids, match_list in self._profiles.items():
            # Handle potential ragged arrays (different round counts per match)
            try:
                matches_array = np.stack(match_list)
            except ValueError:
                # Fallback: process individually
                match_payoffs = [
                    self._compute_discounted_average(m[None, ...])[0]
                    for m in match_list
                ]
                matches_array = np.array(match_payoffs)
                discounted_payoffs = matches_array
            else:
                discounted_payoffs = self._compute_discounted_average(
                    matches_array
                )

            # Average across all matches for this specific profile
            mean_payoffs = np.mean(discounted_payoffs, axis=0)

            # Map back to model types using the internal mapping
            for i, uid in enumerate(uids):
                # If a player was not in the initial `players` list, this will raise KeyError
                model_type = self._uid_to_model[uid]
                aggregated_payoffs[model_type].append(mean_payoffs[i])

        return {
            m_type: float(np.mean(vals))
            for m_type, vals in aggregated_payoffs.items()
        }

    def fitness(self, population: dict[str, float]) -> dict[str, float]:
        """Compute fitness weighted by population probability."""
        if not math.isclose(sum(population.values()), 1.0):
            raise ValueError("Population probabilities must sum to 1.0")

        avg_payoffs = self.model_average_payoff()

        return {m_type: avg_payoffs[m_type] for m_type in population.keys()}

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
