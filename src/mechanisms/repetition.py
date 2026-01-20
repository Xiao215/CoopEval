from typing import Sequence, override

from src.agents.agent_manager import Agent
from src.games.base import Move
from src.logger_manager import LOGGER
from src.mechanisms.base import RepetitiveMechanism
from src.mechanisms.prompts import (REPETITION_MECHANISM_PROMPT,
                                    REPETITION_NO_HISTORY_DESCRIPTION,
                                    REPETITION_OTHERPLAYER_LABEL,
                                    REPETITION_ROUND_LINE,
                                    REPETITION_SELF_LABEL)


class Repetition(RepetitiveMechanism):
    """
    Repetition mechanism that allows for multiple rounds of the same game.
    """

    def __init__(
        self,
        base_game,
        *,
        num_rounds: int,
        discount: float,
        lookup_depth: int = 5,
        include_prior_distributions: bool = True,
        tournament_workers: int = 1,
    ) -> None:
        super().__init__(
            base_game, num_rounds, discount, tournament_workers=tournament_workers
        )
        self.lookup_depth = lookup_depth
        self.include_prior_distributions = include_prior_distributions

    def _build_history_prompts(
        self,
        players: Sequence[Agent],
        round_idx: int,
        history: RepetitiveMechanism.History,
    ) -> list[str]:
        """Build perspective-specific history prompts for each player."""
        rounds_played = round_idx - 1  # Number of completed rounds
        if not history:
            return [
                REPETITION_MECHANISM_PROMPT.format(
                    round_idx=rounds_played,
                    discount=int(self.discount * 100),
                    history_context=REPETITION_NO_HISTORY_DESCRIPTION,
                )
                for _ in players
            ]
        return [
            REPETITION_MECHANISM_PROMPT.format(
                round_idx=rounds_played,
                discount=int(self.discount * 100),
                history_context=self._format_history(players, focus, history),
            )
            for focus in players
        ]

    def _format_single_round(self, moves: list[Move], focus: Agent) -> str:
        """Formats the string representation of a single round of moves."""
        # Find the focus player's move
        focus_move = next(m for m in moves if m.player == focus)

        actions = [
            REPETITION_SELF_LABEL.format(
                action_token=focus_move.action.to_token()
            )
        ]

        # Add other players' moves
        for move in moves:
            if move.player == focus:
                continue
            actions.append(
                REPETITION_OTHERPLAYER_LABEL.format(
                    other_player=f"Player {move.player.player_id}",
                    action_token=move.action.to_token(),
                )
            )

        return "\n".join(actions)

    def _get_prior_summary(
        self,
        players: Sequence[Agent],
        focus: Agent,
        history_size: int,
        start_idx: int,
        history: RepetitiveMechanism.History,
    ) -> list[str]:
        """Generates the summary string for prior action distributions."""
        if not self.include_prior_distributions:
            return []

        summaries = []
        prefix = (
            "Up until this round"
            if start_idx == 0
            else f"Up until round {start_idx}"
        )

        # Helper to format the distribution string
        def format_dist(dist) -> str:
            entries = sorted(dist.items(), key=lambda kv: str(kv[0]))
            return (
                ", ".join(
                    f"{count} time{'s' if count != 1 else ''} {action.to_token()}"
                    for action, count in entries
                )
                + "."
            )

        # 1. Process Focus Player
        focus_dist = history.get_prior_action_distribution(
            focus, lookback_rounds=history_size
        )
        if focus_dist:
            summaries.append(
                f"{prefix}, You have played actions as often as follows: {format_dist(focus_dist)}"
            )

        # 2. Process Other Players
        for player in players:
            if player == focus:
                continue

            other_dist = history.get_prior_action_distribution(
                player, lookback_rounds=history_size
            )

            # Enforce consistency logic from original code
            if other_dist and not focus_dist:
                raise AssertionError(
                    "If other player's prior distribution exists, focus player's must also exist."
                )
            if not other_dist and focus_dist:
                raise AssertionError(
                    "If focus player's prior distribution exists, other player's should also exist."
                )

            if other_dist:
                summaries.append(
                    f"{prefix}, Player {player.player_id} has played actions as often as follows: {format_dist(other_dist)}"
                )

        return summaries

    def _format_history(self, players: Sequence[Agent], focus: Agent, history: RepetitiveMechanism.History) -> str:
        """Format prompt with a limited window of history and action distributions."""
        if self.lookup_depth < 0:
            raise ValueError("lookup_depth must be non-negative")

        total_rounds = len(history.records)
        start_idx = max(0, total_rounds - self.lookup_depth)

        # Build Recent History (Reverse Order)
        recent_history: list[str] = []
        for idx in range(total_rounds - 1, start_idx - 1, -1):
            round_moves = history.records[idx]
            actions_str = self._format_single_round(round_moves, focus)

            recent_history.append(
                REPETITION_ROUND_LINE.format(
                    round_idx=idx + 1, actions=actions_str
                )
            )

        # Add Prior Distributions (if applicable)
        priors = self._get_prior_summary(
            players,
            focus,
            history_size=total_rounds - start_idx,
            start_idx=start_idx,
            history=history,
        )
        recent_history.extend(priors)

        return "\n".join(recent_history)
    @override
    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """Repeat the base game for a specified number of repetitions.

        Returns:
            records: A list of lists, where each inner list contains the Moves
            for that specific round.
        """
        # Create per-matchup history to avoid shared state across parallel matchups
        matchup_history = self.History(self.base_game.action_class)

        for round_idx in range(1, self.num_rounds + 1):
            repetition_information = self._build_history_prompts(
                players, round_idx, matchup_history
            )
            moves = self.base_game.play(
                additional_info=repetition_information,
                players=players,
            )
            matchup_history.append(moves)

        LOGGER.log_record(
            record=matchup_history.records,
            file_name=self.record_file,
        )

        return matchup_history.records
