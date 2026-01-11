import random
from itertools import combinations

from src.agents.agent_manager import Agent


class RandomMatcher:
    """Random matcher that creates uniform random matchups with proper player ID seating.

    Each iteration generates a new random matching by independently shuffling each
    player group and matching players at the same index across seats.

    Args:
        players_by_id: List of player groups, where players_by_id[i] contains all
                      agents that should play in seat i (player_id=i+1)
    """
    def __init__(self, players_by_id: list[list[Agent]]):
        self.players_by_id = players_by_id
        self.k = len(players_by_id)  # Number of seats/players per match

        # Verify all groups have the same size
        group_sizes = [len(group) for group in players_by_id]
        if len(set(group_sizes)) > 1:
            raise ValueError(
                f"All player groups must have the same size for random matching. "
                f"Got sizes: {group_sizes}"
            )
        self.n = group_sizes[0]

    def __iter__(self):
        """Reset iterator (no-op for random matcher as it generates infinite sequence)."""
        return self

    def __next__(self) -> list[list[Agent]]:
        """Generate next random matching.

        Returns:
            List of matchups, where each matchup is a list of agents (one from each seat).
        """

        # Shuffle each player group independently
        shuffled_groups = [random.sample(group, len(group)) for group in self.players_by_id]

        # Create matchups by pairing players at the same index
        matchups = [
            [shuffled_groups[seat_idx][i] for seat_idx in range(self.k)]
            for i in range(self.n)
        ]

        return matchups


class RoundRobin:
    """Round Robin scheduler for K-player matches with proper player ID seating.

    Args:
        players_by_id: List of player groups, where players_by_id[i] contains all agents that should play in seat i (player_id=i+1)
    """
    def __init__(self, players_by_id: list[list[Agent]]):
        self.players_by_id = players_by_id
        self.k = len(players_by_id)  # Number of seats/players per match

        #TODO: Update round robin for seated players
        # Generate all properly-seated matchups using Cartesian product
        # Each matchup is a tuple where position i has a player from players_by_id[i]
        # self.all_matchups = list(product(*players_by_id))
        self.players = [player for group in players_by_id for player in group]
        self.n = len(self.players)

        # Generate complete schedule once for iterator support
        self._complete_schedule = self._generate_complete_schedule()
        self._current_index = 0

    def _generate_complete_schedule(
        self, randomize_order: bool = True
    ) -> list[list[list[Agent]]]:
        """
        Generate the complete round-robin schedule.

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

    def generate_schedule(
        self, randomize_order: bool = True
    ) -> list[list[list[Agent]]]:
        """
        Returns a list of rounds. Each round is a list of matches.
        Each match is a list of player IDs.

        Guarantees:
        1. Every player plays every other player exactly once (for K=2).
        2. No player plays twice in the same time-step (Round).

        Note: For iterator usage, prefer using the RoundRobin object directly in a for loop.
        """
        return self._complete_schedule

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

    def __iter__(self):
        """Reset iterator to start of schedule."""
        self._current_index = 0
        return self

    def __next__(self) -> list[list[Agent]]:
        """Yield next matches_group, cycling through schedule."""
        if len(self._complete_schedule) == 0:
            raise StopIteration

        # Cycle through schedule using modulo
        matches_group = self._complete_schedule[self._current_index % len(self._complete_schedule)]
        self._current_index += 1
        return matches_group

    def get_schedule_length(self) -> int:
        """Return number of rounds in complete schedule."""
        return len(self._complete_schedule)
