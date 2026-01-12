import random
from collections import defaultdict
from typing import Sequence

from src.agents.agent_manager import Agent


class RandomMatcher:
    """
    Random matcher that automatically groups agents by their player_id
    and creates uniform random matchups.

    Args:
        agents: A flat list of all Agents participating in the simulation.
                The matcher will group them based on agent.player_id.
    """

    def __init__(self, agents: Sequence[Agent]):
        groups_map = defaultdict(list)
        for agent in agents:
            groups_map[agent.player_id].append(agent)

        sorted_keys = sorted(groups_map.keys())
        self.player_groups = [groups_map[pid] for pid in sorted_keys]

        self.k = len(self.player_groups)

        # Ensures group created are non-empty and of equal size
        if self.k == 0:
            raise ValueError("No agents provided to RandomMatcher.")
        group_sizes = [len(group) for group in self.player_groups]
        if len(set(group_sizes)) > 1:
            raise ValueError(
                f"Uneven player counts detected. All roles must have the same number of agents. "
                f"Counts by player_id {sorted_keys}: {group_sizes}"
            )

        self.n = group_sizes[0]

    def __iter__(self):
        return self

    def __next__(self) -> list[list[Agent]]:
        """Generate next random matching."""

        # Shuffle each group independently (e.g., shuffle all Player 1s, shuffle all Player 2s)
        shuffled_groups = [
            random.sample(group, len(group)) for group in self.player_groups
        ]

        # Zip them together: The 1st Player 1 plays the 1st Player 2, etc.
        matchups = [
            [shuffled_groups[seat_idx][i] for seat_idx in range(self.k)]
            for i in range(self.n)
        ]

        return matchups
