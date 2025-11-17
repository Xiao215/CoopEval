from typing import Sequence

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.logger_manager import LOGGER
from src.mechanisms.base import RepetitiveMechanism
from src.mechanisms.prompts import (
    REPETITION_HISTORY_HEADER,
    REPETITION_NO_HISTORY_DESCRIPTION,
    REPETITION_NOTE,
    REPETITION_OPPONENT_LABEL,
    REPETITION_ROUND_DESCRIPTION,
    REPETITION_ROUND_LINE,
    REPETITION_SELF_LABEL,
)


class Repetition(RepetitiveMechanism):
    """
    Repetition mechanism that allows for multiple rounds of the same game.
    """

    def _build_history_prompts(
        self,
        players: Sequence[Agent],
        round_idx: int,
    ) -> list[str]:
        """Build perspective-specific history prompts for each player."""

        note = REPETITION_NOTE
        round_description = REPETITION_ROUND_DESCRIPTION.format(
            round_idx=round_idx
        )

        if not self.history:
            base = (
                round_description
                + REPETITION_NO_HISTORY_DESCRIPTION
            )
            return [f"{base}\n\n{note}"] * len(players)

        # Hide players with generic names
        global_names = {
            p.uid: f"Player#{i}" for i, p in enumerate(players, start=1)
        }

        prompts: list[str] = []
        for focus in players:
            lines = [REPETITION_HISTORY_HEADER]
            for past_round_index, round_moves in enumerate(
                self.history, start=1
            ):
                move_map = {m["uid"]: m["action"] for m in round_moves}
                # Always put the focused player first
                actions = [
                    REPETITION_SELF_LABEL.format(
                        action=move_map[focus.uid],
                    )
                ]
                # Then all opponents in stable numeric order
                for other in players:
                    if other.uid == focus.uid:
                        continue
                    actions.append(
                        REPETITION_OPPONENT_LABEL.format(
                            opponent=global_names[other.uid],
                            action=move_map[other.uid],
                        )
                    )
                lines.append(
                    REPETITION_ROUND_LINE.format(
                        round_idx=past_round_index,
                        actions="\n".join(actions),
                    )
                )

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

        for round_idx in tqdm(
            range(1, self.num_rounds + 1),
            desc=f"Running {self.__class__.__name__} repetitive rounds",
        ):
            repetition_information = self._build_history_prompts(
                players, round_idx
            )
            moves = self.base_game.play(
                additional_info=repetition_information,
                players=players,
            )
            self.history.append(match_result=moves)
        LOGGER.log_record(
            record=[
                [m.to_dict() for m in round_moves]
                for round_moves in self.history
            ],
            file_name=self.record_file,
        )
        payoffs.add_profile(self.history.get_records())
