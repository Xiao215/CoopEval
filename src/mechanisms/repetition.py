from typing import Sequence

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.logger_manager import LOGGER
from src.mechanisms.base import RepetitiveMechanism


class Repetition(RepetitiveMechanism):
    """
    Repetition mechanism that allows for multiple rounds of the same game.
    """

    def _build_history_prompts(
        self,
        players: Sequence[Agent],
        history: list[list[dict]],
    ) -> list[str]:
        """Build perspective-specific history prompts for each player."""

        note = (
            "Note: This game is repetitive so your chosen action will be "
            "visible to the same opponent(s) in future rounds."
        )

        if not history:
            base = "History: No rounds have been played yet, so there is no history."
            return [f"{base}\n\n{note}"] * len(players)

        prompts: list[str] = []
        for focus in players:
            opponent_labels: dict[str, str] = {}
            opp_idx = 1
            for other in players:
                if other.label == focus.label:
                    continue
                opponent_labels[other.label] = f"Opponent {opp_idx}"
                opp_idx += 1

            lines = ["History:"]
            for round_idx, round_moves in enumerate(history, start=1):
                move_map = {move["label"]: move for move in round_moves}
                parts = []
                for other in players:
                    actor = "You"
                    if other.label != focus.label:
                        actor = opponent_labels[other.label]
                    action_taken = move_map[other.label]["action"]
                    parts.append(f"{actor}: {action_taken}")
                lines.append(f"  Round {round_idx}: " + ", ".join(parts))

            prompt = "\n".join(lines) + f"\n\n{note}"
            prompts.append(prompt)

        return prompts

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
            repetition_information = self._build_history_prompts(players, history)
            moves = self.base_game.play(
                additional_info=repetition_information,
                players=players,
            )

            history.append([move.to_dict() for move in moves])
            payoffs.add_profile(moves)
        LOGGER.log_record(record=history, file_name=self.record_file)
