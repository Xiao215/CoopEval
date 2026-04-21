"""Baseline mechanism that simply runs the underlying game without interventions."""

from typing import Sequence, override

from coopeval.agents.agent_manager import Agent
from coopeval.games.base import Move
from coopeval.logger_manager import LOGGER
from coopeval.mechanisms.base import Mechanism


class NoMechanism(Mechanism):
    """A mechanism that does nothing."""

    @override
    async def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """Run the base game without any modifications."""
        moves = await self.base_game.play(additional_info="", players=players)
        LOGGER.log_record(
            record=moves,
            file_name=self.record_file,
        )
        return [moves]
