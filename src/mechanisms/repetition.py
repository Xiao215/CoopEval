from typing import Sequence

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.games.base import Move
from src.logger_manager import LOGGER
from src.mechanisms.base import RepetitiveMechanism
from src.mechanisms.prompts import (
    REPETITION_MECHANISM_PROMPT,
    REPETITION_NO_HISTORY_DESCRIPTION,
    REPETITION_OPPONENT_LABEL,
    REPETITION_RECENT_HISTORY_PROMPT,
    REPETITION_RECENT_ROUND_LINE,
    REPETITION_ROUND_LINE,
    REPETITION_SELF_LABEL,
)


class Repetition(RepetitiveMechanism):
    """
    Repetition mechanism that allows for multiple rounds of the same game.
    """

    def __init__(
        self,
        base_game,
        num_rounds: int,
        discount: float,
    ) -> None:
        super().__init__(base_game, num_rounds, discount)
        self.formatter = self._format_full_history

    def _build_history_prompts(
        self,
        players: Sequence[Agent],
        round_idx: int,
    ) -> list[str]:
        """Build perspective-specific history prompts for each player."""

        if len(self.history) == 0:
            history_context = REPETITION_NO_HISTORY_DESCRIPTION
            base_prompt = REPETITION_MECHANISM_PROMPT.format(
                round_idx=round_idx, history_context=history_context
            ).strip()
            return [base_prompt] * len(players)

        base_prompt_template = REPETITION_MECHANISM_PROMPT.strip()
        return [
            base_prompt_template.format(
                round_idx=round_idx,
                history_context=self.formatter(players, focus),
            )
            for focus in players
        ]

    def _format_full_history(
        self,
        players: Sequence[Agent],
        focus: Agent,
    ) -> str:
        """Format prompt including every recorded round."""
        global_names = {
            p.uid: f"Player#{i}" for i, p in enumerate(players, start=1)
        }
        lines: list[str] = []
        for past_round_index, round_moves in enumerate(self.history, start=1):
            move_map = {m.uid: str(m.action) for m in round_moves}
            actions = [
                REPETITION_SELF_LABEL.format(action=move_map[focus.uid]),
            ]
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
                    round_idx=past_round_index, actions="\n".join(actions)
                )
            )
        return "\n".join(lines)

    def _format_recent_history(
        self,
        players: Sequence[Agent],
        focus: Agent,
        lookup_depth: int = 5,
    ) -> str:
        """Format prompt with a limited window of history and opponents' action distribution."""
        if lookup_depth <= 0:
            raise ValueError("lookup_depth must be positive")
        global_names = {
            p.uid: f"Player#{i}" for i, p in enumerate(players, start=1)
        }
        player_name = focus.name
        recent_rounds = self.history.get_prior_rounds(
            player_name, lookback_rounds=0, lookup_depth=lookup_depth
        )
        history_size = len(recent_rounds)

        recent_history: list[str] = []
        for relative_idx, round_moves in enumerate(recent_rounds, start=1):
            move_map = {m.uid: str(m.action) for m in round_moves}
            actions = [
                REPETITION_SELF_LABEL.format(action=move_map[focus.uid]),
            ]
            for other in players:
                if other.uid == focus.uid:
                    continue
                actions.append(
                    REPETITION_OPPONENT_LABEL.format(
                        opponent=global_names[other.uid],
                        action=move_map[other.uid],
                    )
                )
            round_summary = REPETITION_RECENT_ROUND_LINE.format(
                relative_idx=relative_idx, actions="\n".join(actions)
            )
            recent_history.append(round_summary)

        for other in players:
            if other.uid == focus.uid:
                continue
            prior_dist = self.history.get_prior_action_distribution(
                other.name, lookback_rounds=history_size
            )
            if prior_dist:
                recent_history.append(
                    f"{global_names[other.uid]}'s action counts from before these {history_size} recent round(s):"
                )
                for action, count in sorted(
                    prior_dist.items(), key=lambda kv: str(kv[0])
                ):
                    recent_history.append(f"\t{action}: {count}")

        return REPETITION_RECENT_HISTORY_PROMPT.format(
            window_count=history_size,
            recent_history="\n".join(recent_history),
        )

    @staticmethod
    def _serialize_records(
        records: Sequence[Sequence[Move]],
    ) -> list[list[dict]]:
        payload: list[list[dict]] = []
        for round_moves in records:
            round_payload: list[dict] = []
            for move in round_moves:
                round_payload.append(
                    {
                        "uid": move.uid,
                        "player_name": move.player_name,
                        "action": str(move.action),
                        "points": move.points,
                        "response": move.response,
                    }
                )
            payload.append(round_payload)
        return payload

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
            self.history.append(moves)
        records: list[list[Move]] = [list(r) for r in self.history.records]
        LOGGER.log_record(
            record=self._serialize_records(records), file_name=self.record_file
        )
        payoffs.add_profile(records)
