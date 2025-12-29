from typing import Mapping, Sequence, Any

import numpy as np

class PopulationPayoffs:

    def __init__(self, agent_names: Sequence[str], *, discount: float | None = None) -> None:
        self.agent_names = list(agent_names)
        self.k = len(self.agent_names)
        self._idx = {t: i for i, t in enumerate(self.agent_names)}
        self.discount = discount if discount is not None else 1.0

        # Store list of payoff vectors
        self._table: dict[tuple[str, ...], np.ndarray] = {}

    def reset(self) -> None:
        """Reset the payoff table to be empty."""
        self._table.clear()

    def add_profile(self, payoff_map: Mapping[str, float]) -> None:
        """Add a new payoff profile to the table.
        If the game is repetitive, only the new match payoff is added rather than the cumulative payoff.

        Eg, if the match payoff is [1, 2, 3] over 3 rounds, [1, 2, 3] should be added sequentially.
        """
        key = tuple(name for name in self.agent_names if name in payoff_map)

        vec = np.zeros(self.k, dtype=float)
        for t, p in payoff_map.items():
            vec[self._idx[t]] = float(p)

        if key not in self._table:
            # start a (1, k) array
            self._table[key] = vec[None, :]
        else:
            # append to shape (n+1, k)
            self._table[key] = np.vstack([self._table[key], vec[None, :]])

    def _discounted_average(self, payoffs: np.ndarray) -> np.ndarray:
        """
        Apply geometric discounting over the payoff sequence.
        Return the discounted average payoff vector for all players.
        """
        n = len(payoffs)
        cumsum_ave_payoffs = np.cumsum(payoffs, axis=0) / np.arange(1, n + 1)[:, None]

        x = self.discount
        if x < 1.0:
            weights = np.array([(1 - x) * (x**i) for i in range(n)])
            weights[-1] = x ** (n)  # last one special
        else:
            weights = np.ones(n) / n
        # weights already sum to 1, so no need to normalize
        return np.sum(weights[:, None] * cumsum_ave_payoffs, axis=0)

    def fitness(self, population: np.ndarray) -> np.ndarray:
        """Calculate the fitness of the population based on the payoff profiles."""
        fitness = np.zeros(self.k, float)
        for key, payoff_list in self._table.items():
            avg_profile = self._discounted_average(payoff_list)
            prob = 1.0
            for t in key:
                prob *= population[self._idx[t]]
            fitness += prob * avg_profile

        return fitness

    def to_record(self) -> dict[str, Any]:
        """
        JSON-serializable snapshot.

        - expected_payoff: dict[name -> payoff] computed as fitness(population=all ones).
        """
        profiles = []
        for key, arr in self._table.items():
            idxs = [self._idx[name] for name in key]
            local_rounds = arr[:, idxs]  # shape: (rounds, len(key))
            local_avg = self._discounted_average(local_rounds)
            entry = {
                "players": list(key),
                "rounds": local_rounds.tolist(),
                "discounted_average": local_avg.tolist(),
            }
            profiles.append(entry)

        expected_payoff = self.fitness(np.ones(self.k) / self.k)

        payoff_record = {
            "discount": float(self.discount),
            "profiles": profiles,
            "expected_payoff": {
                name: v * self.k
                for name, v in zip(self.agent_names, expected_payoff.tolist())
            },
        }
        print(payoff_record)
        return payoff_record
