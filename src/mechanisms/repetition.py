import random
from typing import Sequence

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.mechanisms.base import RepetitiveMechanism
from src.logger_manager import log_record

random.seed(42)


class Repetition(RepetitiveMechanism):
    """
    Repetition mechanism that allows for multiple rounds of the same game.
    """

    def _parse_history(self, history: list[tuple[dict]]) -> str:
        """Parse the history of past actions as the mechanism information."""
        if not history:
            return "History: None of the players have played yet, so there is no history."

        history_str = "History:\n"
        for i, (players_moves) in enumerate(history):
            history_str = f"  Round {i + 1}: "
            for move in players_moves:
                history_str += f"{move['name']}: {move['action']}, "
            history_str = history_str[:-2]
            history_str += "\n"

        return (
            history_str.strip()
            + "\n\nNote: This game is repetitive so your chosen action "
            "will be visible to your opponents in future rounds."
        )

    def _play_matchup(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> None:
        """Repeat the base game for a specified number of repetitions.

        Returns:
            final_score (dict[str, float]): A dictionary mapping player names to
            their final scores after all rounds.
        """

        history = []
        for _ in tqdm(
            range(self.num_rounds),
            desc=f"Running {self.__class__.__name__} repetitive rounds",
        ):
            repetition_information = self._parse_history(history)
            players_moves = self.base_game.play(
                additional_info=repetition_information, players=players
            )

            history.append([move.to_dict() for move in players_moves])
            payoffs.add_profile({move.name: move.points for move in players_moves})

        log_record(record=history, file_name=self.record_file)
