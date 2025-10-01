"""Store and aggregate tournament payoffs for evolutionary dynamics."""

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from src.agents.agent_manager import Agent
from src.games.base import Move


class PopulationPayoffs:
    """Manage payoff tables while tracking seat-level outcomes."""

    def __init__(
        self,
        *,
        agents: Sequence[Agent],
        discount: float | None = None,
    ) -> None:
        self.agent_names = [agent.name for agent in agents]
        self.k = len(self.agent_names)
        self._name_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, name in enumerate(self.agent_names):
            self._name_to_indices[name].append(i)
        self.discount = discount if discount is not None else 1.0

        # lineup_key -> entry dict
        self._table: dict[
            tuple[str, ...],
            dict[str, Any],
        ] = {}

    def _discounted_average(self, payoffs: np.ndarray) -> np.ndarray:
        """Apply geometric discounting over the payoff sequence."""
        n = len(payoffs)
        if n == 0:
            return np.zeros(payoffs.shape[1], dtype=float)

        cumsum_ave_payoffs = np.cumsum(payoffs, axis=0) / np.arange(1, n + 1)[:, None]

        x = self.discount
        if x < 1.0:
            weights = np.array([(1 - x) * (x**i) for i in range(n)], dtype=float)
            weights[-1] += x**n  # fold in the tail mass so weights sum to one
            weights /= weights.sum()
        else:
            weights = np.ones(n, dtype=float) / n
        return np.sum(weights[:, None] * cumsum_ave_payoffs, axis=0)

    def _aggregate_rounds(
        self, rounds: np.ndarray, base_indices: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate seat-level payoffs into averages per base agent."""
        if rounds.size == 0:
            return np.zeros((0, self.k), dtype=float), np.zeros(self.k, dtype=int)

        counts = np.zeros(self.k, dtype=int)
        per_agent = np.zeros((rounds.shape[0], self.k), dtype=float)
        for seat_idx, base_idx in enumerate(base_indices):
            per_agent[:, base_idx] += rounds[:, seat_idx]
            counts[base_idx] += 1

        nonzero = counts > 0
        if np.any(nonzero):
            per_agent[:, nonzero] /= counts[nonzero]
        return per_agent, counts

    def reset(self) -> None:
        self._table.clear()

    def merge_from(self, other: "PopulationPayoffs") -> None:
        """Merge another payoff table by stacking round data for matching lineups."""
        if self.agent_names != other.agent_names:
            raise ValueError("Cannot merge payoffs with different agent sets/order")
        if not math.isclose(self.discount, other.discount):
            raise ValueError("Cannot merge payoffs with different discounts")

        for key, entry in other._table.items():
            if key not in self._table:
                self._table[key] = {
                    "labels": entry["labels"],
                    "base_names": entry["base_names"],
                    "base_indices": entry["base_indices"],
                    "rounds": entry["rounds"].copy(),
                }
            else:
                self._table[key]["rounds"] = np.vstack(
                    [self._table[key]["rounds"], entry["rounds"]]
                )

    def add_profile(self, moves: list[Move]) -> None:
        """Record a single matchup outcome consisting of the provided ``moves``."""
        if not moves:
            return

        labels = tuple(move.label for move in moves)
        base_names = tuple(move.name for move in moves)

        usage: defaultdict[str, int] = defaultdict(int)
        base_indices_list: list[int] = []
        for name in base_names:
            usage[name] += 1
            idx_list = self._name_to_indices.get(name)
            if not idx_list:
                raise ValueError(f"Unknown agent name {name!r}")
            # Allow duplicate seat appearances of the same base agent by reusing
            # the first known index when more seats are observed than base entries.
            index_pos = min(usage[name] - 1, len(idx_list) - 1)
            base_indices_list.append(idx_list[index_pos])
        base_indices = tuple(base_indices_list)
        row = np.array([[float(move.points) for move in moves]], dtype=float)

        if labels not in self._table:
            self._table[labels] = {
                "labels": labels,
                "base_names": base_names,
                "base_indices": base_indices,
                "rounds": row,
            }
        else:
            self._table[labels]["rounds"] = np.vstack(
                [self._table[labels]["rounds"], row]
            )

    def fitness(self, population: np.ndarray) -> np.ndarray:
        """Compute expected payoff vector for a population distribution."""
        if population.shape != (self.k,):
            raise ValueError(
                f"population must be shape ({self.k},), got {population.shape}"
            )
        s = population.sum()
        if not np.isclose(s, 1.0):
            raise ValueError(f"Total population must sum to 1.0, got {s}")

        fitness = np.zeros(self.k, dtype=float)
        for entry in self._table.values():
            rounds = entry["rounds"]
            base_indices = entry["base_indices"]
            per_agent_rounds, counts = self._aggregate_rounds(rounds, base_indices)

            if np.any((population == 0) & (counts > 0)):
                continue

            logprod = np.sum(
                counts * np.log(np.where(population > 0, population, 1.0))
            )
            comb_factor = math.factorial(int(np.sum(counts))) / np.prod(
                [math.factorial(int(c)) for c in counts if c > 0]
            )
            weight = comb_factor * math.exp(logprod)

            if weight == 0.0:
                continue

            avg_profile = self._discounted_average(per_agent_rounds)
            fitness += weight * avg_profile

        return fitness

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of accumulated matchup data."""
        profiles: list[dict[str, Any]] = []
        for entry in self._table.values():
            labels = entry["labels"]
            base_names = entry["base_names"]
            base_indices = entry["base_indices"]
            rounds = entry["rounds"]

            seat_discounted = self._discounted_average(rounds)
            per_agent_rounds, _ = self._aggregate_rounds(rounds, base_indices)
            agent_discounted = self._discounted_average(per_agent_rounds)

            discounted_average: dict[str, float] = {}
            for idx in dict.fromkeys(base_indices):
                discounted_average[self.agent_names[idx]] = float(agent_discounted[idx])

            instance_discounted = {
                label: float(seat_discounted[i]) for i, label in enumerate(labels)
            }

            profiles.append(
                {
                    "players": list(base_names),
                    "player_labels": list(labels),
                    "rounds": rounds.tolist(),
                    "discounted_average": discounted_average,
                    "instance_discounted_average": instance_discounted,
                }
            )

        expected_payoff_vec = self.fitness(np.ones(self.k, dtype=float) / self.k)
        expected_payoff = {
            name: float(expected_payoff_vec[i])
            for i, name in enumerate(self.agent_names)
        }

        return {
            "discount": float(self.discount),
            "profiles": profiles,
            "expected_payoff": expected_payoff,
        }
