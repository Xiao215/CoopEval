import random

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
