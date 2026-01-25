from typing import Sequence, override

from src.agents.agent_manager import Agent
from src.games.base import Move
from src.mechanisms.base import Mechanism
from src.logger_manager import LOGGER


class NoMechanism(Mechanism):
    """A mechanism that does nothing."""

    @override
    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """Run the base game without any modifications."""
        moves = self.base_game.play(additional_info="", players=players)
        LOGGER.log_record(
            record=moves,
            file_name=self.record_file,
        )
        return [moves]
