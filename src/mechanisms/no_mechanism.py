from typing import Sequence

from src.agents.agent_manager import Agent
from src.games.base import Move
from src.mechanisms.base import Mechanism


class NoMechanism(Mechanism):
    """A mechanism that does nothing."""

    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """Run the base game without any modifications."""
        moves = self.base_game.play(additional_info="None.", players=players)
        return [moves]
