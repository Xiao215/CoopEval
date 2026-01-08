from typing import Sequence

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.games.base import Move
from src.logger_manager import LOGGER
from src.mechanisms.base import RepetitiveMechanism
from src.mechanisms.prompts import (
    REPETITION_MECHANISM_PROMPT,
    REPETITION_NO_HISTORY_DESCRIPTION,
    REPETITION_OTHERPLAYER_LABEL,
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
        lookup_depth: int,
        include_prior_distributions: bool,
    ) -> None:
        super().__init__(base_game, num_rounds, discount)
        self.lookup_depth = lookup_depth
        self.include_prior_distributions = include_prior_distributions
        self.formatter = self._format_history

    def _build_history_prompts(
        self,
        players: Sequence[Agent],
        round_idx: int,
    ) -> list[str]:
        """Build perspective-specific history prompts for each player."""

        if len(self.history) == 0:
            history_context = REPETITION_NO_HISTORY_DESCRIPTION
            base_prompt = REPETITION_MECHANISM_PROMPT.format(
                round_idx=round_idx,
                discount=int(self.discount * 100),
                history_context=history_context
            )
            return [base_prompt] * len(players)

        base_prompt_template = REPETITION_MECHANISM_PROMPT
        return [
            base_prompt_template.format(
                round_idx=round_idx,
                discount=int(self.discount * 100),
                history_context=self.formatter(players, focus),
            )
            for focus in players
        ]

    def _format_history(
        self,
        players: Sequence[Agent],
        focus: Agent,
    ) -> str:
        """Format prompt with a limited window of history and other players' action distribution."""
        lookup_depth = self.lookup_depth
        if lookup_depth < 0:
            raise ValueError("lookup_depth must be non-negative")
        global_names = {
            p.uid: f"PlayerID {p.player_id}" for p in players
        }
        
        total_rounds = len(self.history.records)
        start_idx = max(0, total_rounds - lookup_depth)
        history_size = total_rounds - start_idx

        recent_history: list[str] = []
        for idx in range(total_rounds - 1, start_idx - 1, -1):
            round_moves = self.history.records[idx]
            round_index = idx + 1
            move_map = {m.uid: m.action.to_token() for m in round_moves}
            actions = [
                REPETITION_SELF_LABEL.format(action=move_map[focus.uid]),
            ]
            for other in players:
                if other.uid == focus.uid:
                    continue
                actions.append(
                    REPETITION_OTHERPLAYER_LABEL.format(
                        other_player=global_names[other.uid],
                        action=move_map[other.uid],
                    )
                )
            round_summary = REPETITION_ROUND_LINE.format(
                round_idx=round_index, actions="\n".join(actions)
            )
            recent_history.append(round_summary)

        if self.include_prior_distributions:
            prior_dist = self.history.get_prior_action_distribution(
                focus.name, lookback_rounds=history_size
            )
            if prior_dist:
                prior_dist_exists = True
                if lookup_depth == 0:
                    recent_history.append(
                        f"The aggregate counts of how often each player has chosen each action in the past are:"
                    )
                else:
                    last_rounds_text = "last round" if history_size == 1 else f"{history_size} last rounds"
                    recent_history.append(
                        f"Before and up until the {last_rounds_text}, we had the following aggregate counts of how often each player has chosen each action."
                    )
                recent_history.append(
                    f"You:"
                )
                for action, count in sorted(
                    prior_dist.items(), key=lambda kv: str(kv[0])
                ):
                    recent_history.append(f"\t{action.to_token()}: played {count} time{'s' if count != 1 else ''}")
            else:
                prior_dist_exists = False
            
            for player in players:
                if player.uid == focus.uid:
                    continue
                prior_dist = self.history.get_prior_action_distribution(
                    player.name, lookback_rounds=history_size
                )
                if prior_dist:
                    assert prior_dist_exists, (
                        "If other player's prior distribution exists, then the focus player's prior distribution must also exist."
                    )
                    recent_history.append(
                        f"{global_names[player.uid]}:"
                    )
                    for action, count in sorted(
                        prior_dist.items(), key=lambda kv: str(kv[0])
                    ):
                        recent_history.append(f"\t{action.to_token()}: played {count} time{'s' if count != 1 else ''}")
                else:
                    assert not prior_dist_exists, (
                        "If one player's prior distribution does not exist, then the focus player's prior distribution should not exist either."
                    )
        return "\n".join(recent_history)

    @staticmethod
    def _serialize_records(
        records: list[list[Move]],
    ) -> list[list[dict]]:
        payload: list[list[dict]] = []
        for round_moves in records:
            round_payload: list[dict] = []
            for move in round_moves:
                round_payload.append(
                    {
                        "uid": move.uid,
                        "player_name": move.player_name,
                        "action": (
                            move.action.value
                            if hasattr(move.action, "value")
                            else str(move.action)
                        ),
                        "points": move.points,
                        "response": move.response,
                    }
                )
            payload.append(round_payload)
        return payload

    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """Repeat the base game for a specified number of repetitions.

        Returns:
            records: A list of lists, where each inner list contains the Moves
            for that specific round.
        """
        self.history.clear()
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

        # Convert history records to a standard list of lists
        records: list[list[Move]] = [list(r) for r in self.history.records]

        # Log the interaction to file
        LOGGER.log_record(
            record=self._serialize_records(records), file_name=self.record_file
        )

        return records




    def _LEGACY_format_history(
        self,
        players: Sequence[Agent],
        focus: Agent,
    ) -> str:
        """Format prompt including every recorded round."""
        global_names = {
            p.uid: f"PlayerID {p.player_id}" for p in players
        }
        lines: list[str] = []
        for round_index, round_moves in enumerate(self.history, start=1):
            move_map = {m.uid: m.action.to_token() for m in round_moves}
            actions = [
                REPETITION_SELF_LABEL.format(action=move_map[focus.uid]),
            ]
            for other in players:
                if other.uid == focus.uid:
                    continue
                actions.append(
                    REPETITION_OTHERPLAYER_LABEL.format(
                        other_player=global_names[other.uid],
                        action=move_map[other.uid],
                    )
                )
            lines.append(
                REPETITION_ROUND_LINE.format(
                    round_idx=round_index, actions="\n".join(actions)
                )
            )
        return "\n".join(lines)