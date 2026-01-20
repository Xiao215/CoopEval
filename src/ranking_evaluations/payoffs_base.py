"""Base class for tracking tournament payoffs across different mechanism types."""

import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from src.games.base import Move


class PayoffsBase(ABC):
    """Abstract base class for tracking tournament payoffs.

    Different mechanisms require different payoff tracking strategies:
    - Matchup-based mechanisms (Mediation, Disarmament, Repetition) can evaluate
      matchups in isolation and build payoff tensors.
    - History-dependent mechanisms (Reputation) cannot evaluate matchups in
      isolation and only track aggregate statistics.
    """

    def __init__(
        self,
        *,
        discount: float | None = None,
    ) -> None:
        """
        Args:
            players: List of Agents involved in the tournament.
                This is required to map UIDs to model types.
            discount: Geometric discount factor in (0, 1].
        """
        self.discount = discount if discount is not None else 1.0
        if not 0.0 < self.discount <= 1.0:
            warnings.warn(
                f"Discount factor should be in (0, 1], got {self.discount}. "
                "Ensure this is intended."
            )

    @abstractmethod
    def reset(self) -> None:
        """Clear all recorded matchup outcomes."""
        raise NotImplementedError

    @abstractmethod
    def add_profile(self, moves_over_rounds: Sequence[Sequence[Move]]) -> None:
        """
        Record match outcomes.
        """
        raise NotImplementedError

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
        # Update the tail probability. This is the same as weights[-1] += d ** num_rounds.
        # It assumes that whatever the last round's payoff is that the players achieved,
        # will be the estimate of the payoffs they can expect for the rest of the iterations.
        weights[-1] = d ** (num_rounds - 1)

        # In particular, the sum to 1.0
        if not math.isclose(np.sum(weights), 1.0):
            raise ValueError(
                f"Discount weights sum to {np.sum(weights)}, expected 1.0"
            )

        # Broadcasting weights: (num_rounds, 1) to multiply against (matches, rounds, players)
        # We sum over axis 0 (rounds)
        weighted_sums = np.sum(payoffs * weights[:, None], axis=0)

        return weighted_sums

    @abstractmethod
    def agent_average_payoff(self) -> dict[str, float | None]:
        """
        Compute the average payoff of each agent type in the population.
        """
        raise NotImplementedError

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        """Serialize payoff records."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_json(cls, json_data: dict[str, Any]) -> "PayoffsBase":
        """Reconstruct instance from JSON."""
        raise NotImplementedError
