from typing import Sequence

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.games.base import Move
from src.logger_manager import LOGGER
from src.mechanisms.base import RepetitiveMechanism


class Repetition(RepetitiveMechanism):
    """
    Repetition mechanism that allows for multiple rounds of the same game.
    """

    def _build_history_prompts(
        self,
        players: Sequence[Agent],
        moves_over_rounds: list[list[Move]],
        *,
        current_round_number: int,
    ) -> list[str]:
        """Build perspective-specific history prompts for each player."""

        note = (
            "Note: This game is repetitive so your chosen action will be "
            "visible to the same opponent(s) in future rounds."
        )

        round_description = f"This is round {current_round_number} of the game.\n"

        if not moves_over_rounds:
            base = (
                round_description
                + "History: No rounds have been played yet, so there is no history."
            )
            return [f"{base}\n\n{note}"] * len(players)

        prompts = [round_description] * len(players)

        # Hide players with generic names
        global_names = {p.uid: f"Player#{i}" for i, p in enumerate(players, start=1)}

        prompts: list[str] = []
        for focus in players:
            lines = ["History:"]
            for round_idx, round_moves in enumerate(moves_over_rounds, start=1):
                move_map = {m.uid: m.action for m in round_moves}

                # Always put the focused player first
                actions = [f"\tYou: {move_map[focus.uid]}"]

                # Then all opponents in stable numeric order
                for other in players:
                    if other.uid == focus.uid:
                        continue
                    actions.append(
                        f"\t{global_names[other.uid]}: {move_map[other.uid]}"
                    )

                lines.append(f"Round {round_idx}:\n" + "\n".join(actions))

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

        moves_over_rounds = []
        for _ in tqdm(
            range(self.num_rounds),
            desc=f"Running {self.__class__.__name__} repetitive rounds",
        ):
            repetition_information = self._build_history_prompts(
                players,
                moves_over_rounds,
                current_round_number=len(moves_over_rounds) + 1,
            )
            moves = self.base_game.play(
                additional_info=repetition_information,
                players=players,
            )
            moves_over_rounds.append(moves)
        LOGGER.log_record(
            record=[
                [m.to_dict() for m in round_moves] for round_moves in moves_over_rounds
            ],
            file_name=self.record_file,
        )
        payoffs.add_profile(moves_over_rounds)
