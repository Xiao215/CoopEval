from typing import Sequence

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.mechanisms.base import Mechanism


class NoMechanism(Mechanism):
    """A mechanism that does nothing."""

    def _play_matchup(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> None:
        """Run the base game without any modifications."""
        moves = self.base_game.play(additional_info="None.", players=players)
        payoffs.add_profile([[move.to_dict() for move in moves]])
