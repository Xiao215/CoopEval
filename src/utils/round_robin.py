from typing import Sequence
import random
from itertools import combinations

from src.agents.agent_manager import Agent


class RoundRobin:
    """Round Robin scheduler for K-player matches."""
    def __init__(self, players: Sequence[Agent], group_size: int = 2):
        self.players = players if isinstance(players, list) else list(players)
        self.k = group_size
        self.n = len(self.players)

    def generate_schedule(
        self, randomize_order: bool = True
    ) -> list[list[list[Agent]]]:
        """
        Returns a list of rounds. Each round is a list of matches.
        Each match is a list of player IDs.

        Guarantees:
        1. Every player plays every other player exactly once (for K=2).
        2. No player plays twice in the same time-step (Round).
        """
        if self.k == 2:
            return self._schedule_round_robin_circle(randomize_order)
        else:
            return self._schedule_k_player_approx(randomize_order)

    def _schedule_round_robin_circle(
        self, randomize_order: bool
    ) -> list[list[list[Agent]]]:
        """
        Standard Circle Method for K=2.
        Time Complexity: O(N^2)
        """
        rotation_pool: list[Agent | None] = []
        rotation_pool.extend(self.players)
        if self.n % 2 != 0:
            # If odd number of players, add None as a placeholder for the extra spot
            rotation_pool.append(None)

        if randomize_order:
            random.shuffle(rotation_pool)

        num_players = len(rotation_pool)
        num_rounds = num_players - 1
        half = num_players // 2

        schedule = []

        player_indices = list(range(num_players))
        for _ in range(num_rounds):
            round_matches = []

            # Pair top vs bottom indices
            # Top: [0, 1, ..., half-1]
            # Bottom: [end, end-1, ..., half]
            for i in range(half):
                p1 = rotation_pool[player_indices[i]]
                p2 = rotation_pool[player_indices[num_players - 1 - i]]

                if p1 is not None and p2 is not None:
                    round_matches.append([p1, p2])

            if randomize_order:
                random.shuffle(round_matches)

            schedule.append(round_matches)

            # Rotate indices: Keep index 0 fixed, rotate the rest
            # 0, [1, 2, ... N] -> 0, [N, 1, 2 ... N-1]
            player_indices = (
                [player_indices[0]]
                + [player_indices[-1]]
                + player_indices[1:-1]
            )

        if randomize_order:
            random.shuffle(schedule)

        return schedule

    def _schedule_k_player_approx(self, randomize_order: bool) -> list[list[list[Agent]]]:
        """
        For K > 2, we use 'Random Grouping without Replacement'
        until all unique combinations are exhausted or a limit is hit.
        """
        # Get all unique combinations of size K
        all_combos = list(combinations(self.players, self.k))
        if randomize_order:
            random.shuffle(all_combos)

        # Group them into rounds where no player appears twice
        schedule = []
        while all_combos:
            current_round = []
            seen_in_round = set()

            # Greedy packing of the round
            remaining_combos = []
            for combo in all_combos:
                if not any(p in seen_in_round for p in combo):
                    current_round.append(list(combo))
                    seen_in_round.update(combo)
                else:
                    remaining_combos.append(combo)

            schedule.append(current_round)
            all_combos = remaining_combos

        return schedule
