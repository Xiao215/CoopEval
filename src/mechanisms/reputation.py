"""Mechanisms that expose behavioural reputation across repeated rounds."""

from abc import ABC
from typing import Sequence, override

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.games.base import Game, Move
from src.logger_manager import LOGGER
from src.mechanisms.base import RepetitiveMechanism
from src.mechanisms.prompts import (
    REPUTATION_ACTION_DISTRIBUTION, REPUTATION_MECHANISM_PROMPT,
    REPUTATION_NO_ACTION_DISTRIBUTION_DESCRIPTION,
    REPUTATION_NO_HISTORY_DESCRIPTION, REPUTATION_PLAYERS_HEADER)
from src.ranking_evaluations.payoffs_base import PayoffsBase
from src.ranking_evaluations.reputation_payoffs import ReputationPayoffs
from src.registry.agent_registry import create_players_with_player_id
from src.utils.match_scheduler_reputation import RandomMatcher


class Reputation(RepetitiveMechanism, ABC):
    """
    Reputation mechanism that makes each players' reputation visible to all players.
    """

    def __init__(
        self,
        base_game: Game,
        *,
        num_rounds: int,
        discount: float,
        lookup_depth: int = 5,
        max_recursion_depth: int | None = None,
        include_prior_distributions: bool = True,
    ) -> None:
        super().__init__(base_game, num_rounds, discount)
        self.reputation_depth = lookup_depth
        self.include_prior_distributions = include_prior_distributions
        self.player_to_uid = {}
        if not max_recursion_depth:
            self.max_recursion_depth = lookup_depth + 1
        else:
            self.max_recursion_depth = max_recursion_depth

    def _get_display_name(self, target_player: Agent, observer: Agent) -> str:
        """
        Returns 'You' if the target is the observer, otherwise 'Agent {uid}'.
        This removes the need for stateful dictionary mutation.
        """
        if target_player == observer:
            return "You"
        return f"Agent {self.player_to_uid[target_player]}"

    def _get_verb(self, display_name: str) -> str:
        """Helper for grammar: 'You have' vs 'Agent X has'."""
        return "have" if display_name == "You" else "has"

    @override
    def _build_payoffs(self) -> ReputationPayoffs:
        """Override to use ReputationPayoffs instead of MatchupPayoffs."""
        return ReputationPayoffs(discount=self.discount)

    def _build_history_prompts(self, players: Sequence[Agent]) -> list[str]:
        """
        Constructs prompts by iterating players and delegating content generation
        entirely to _format_reputation.
        """
        prompts = []

        for observer in players:
            round_idx = self.history.get_rounds_played_count(observer) + 1
            direct_opponents = [p for p in players if p != observer]
            reputation_text = self._format_reputation(
                observer=observer,
                direct_opponents=direct_opponents,
                current_round=round_idx
            )

            base_prompt = REPUTATION_MECHANISM_PROMPT.format(
                round_idx=round_idx,
                discount=int(self.discount * 100),
                history_context=reputation_text,
            )
            prompts.append(base_prompt)

        return prompts

    def _format_reputation(
        self,
        observer: Agent,
        direct_opponents: Sequence[Agent],
        current_round: int,
    ) -> str:
        """Format the n-order reputation of opponents and self into a tree structure."""

        window_start = max(1, current_round - self.reputation_depth)
        window_end = current_round - 1

        # Header: Generate IDs relative to the focus player's perspective
        opponent_ids = [
            self._get_display_name(opp, observer) for opp in direct_opponents
        ]

        lines = [
            REPUTATION_PLAYERS_HEADER.format(
                num_opponents=len(direct_opponents),
                opponent_ids=", ".join(opponent_ids),
            ),
            "",
        ]

        # Process each player (focus player + opponents)
        # We iterate over a list containing the focus player and all opponents
        for player in [observer] + list(direct_opponents):

            player_label = self._get_display_name(player, observer)

            # Get player's history within window
            recent_rounds = self.history.get_prior_rounds(
                player, lookback_rounds=0, lookup_depth=self.reputation_depth
            )
            filtered_rounds = [
                (idx, moves)
                for (idx, moves) in recent_rounds
                if window_start <= idx <= window_end
            ]

            if not filtered_rounds:
                verb = self._get_verb(player_label)
                lines.append(
                    REPUTATION_NO_HISTORY_DESCRIPTION.format(
                        name_plus_have=f"{player_label} {verb}"
                    )
                )
                continue

            # Header for this player
            history_header = (
                f"History of play of {player_label}:"
                if player_label != "You"
                else "Your history of play:"
            )
            lines.append(history_header)

            # Recursively format player's history
            player_lines = self._format_player_history_recursive(
                target_player=player,
                observer=observer,
                encounter_round=current_round,
                indent="",
                current_round=current_round,
                recursion_depth=0,
            )
            lines.extend(player_lines)

            lines.append("")  # Blank line between players

        return "\n".join(lines).strip()

    def _format_action_distribution(
        self,
        target_player: Agent,
        observer: Agent,
        current_round: int,
        indent: str = "",
    ) -> list[str]:
        """
        Format action distribution for a player for rounds 1 to window_start-1.
        """
        if not self.include_prior_distributions:
            return []

        lines = []
        window_start = max(1, current_round - self.reputation_depth)

        if window_start > 1:
            lookback_for_distribution = current_round - window_start
            stats = self.history.get_prior_action_distribution(
                target_player, lookback_rounds=lookback_for_distribution
            )

            player_label = self._get_display_name(target_player, observer)

            # Add explicit Player ID context if the game isn't symmetric
            potential_player_id = (
                ""
                if self.base_game.is_symmetric
                else f" (as Player {target_player.player_id})"
            )

            verb = self._get_verb(player_label)
            name_plus_have = f"{player_label}{potential_player_id} {verb}"

            if stats:
                stats_str = (
                    ", ".join(
                        f"{count} time{'s' if count != 1 else ''} {action.to_token()}"
                        for action, count in sorted(
                            stats.items(), key=lambda kv: str(kv[0])
                        )
                    )
                    + "."
                )
                lines.append(
                    REPUTATION_ACTION_DISTRIBUTION.format(
                        indent=indent,
                        name_plus_have=name_plus_have,
                        window_start_minus_one=window_start - 1,
                        stats_str=stats_str,
                    )
                )
            else:
                lines.append(
                    REPUTATION_NO_ACTION_DISTRIBUTION_DESCRIPTION.format(
                        indent=indent,
                        window_start_minus_one=window_start - 1,
                        name_plus_have=name_plus_have,
                    )
                )

        return lines

    def _format_player_history_recursive(
        self,
        target_player: Agent,
        observer: Agent,
        encounter_round: int,
        indent: str,
        current_round: int,
        recursion_depth: int,
    ) -> list[str]:
        """
        Recursively format a player's match history.

        Args:
            target_player: The player whose history we are formatting.
            observer: The player who will receive this prompt (determines 'You' vs 'Agent X').
        """
        lines = []

        # Calculate time window
        window_start = max(1, current_round - self.reputation_depth)
        window_end = current_round - 1

        # Get player's history before encounter_round
        lookback = current_round - encounter_round
        recent_rounds = self.history.get_prior_rounds(
            target_player, lookback, self.reputation_depth
        )

        # Filter to time window and before encounter
        filtered_rounds = [
            (idx, moves)
            for (idx, moves) in recent_rounds
            if window_start <= idx <= window_end and idx < encounter_round
        ]

        if not filtered_rounds:
            return lines

        player_label = self._get_display_name(target_player, observer)
        should_expand_opponents = recursion_depth < self.max_recursion_depth

        # Format each match and expand opponents immediately after
        for i, (round_idx, round_moves) in enumerate(reversed(filtered_rounds)):

            # Find the move made by the target player
            player_move = next(m for m in round_moves if m.player == target_player)
            other_moves = [m for m in round_moves if m.player != target_player]

            # Determine tree branch visual
            is_last = i == len(filtered_rounds) - 1
            branch = "└─" if is_last else "├─"
            child_indent = indent + ("   " if is_last else "│  ")

            # Format match description
            potential_pid = (
                ""
                if self.base_game.is_symmetric
                else f" as Player {target_player.player_id}"
            )

            match_desc = f"{player_label} (played {player_move.action.to_token()}{potential_pid}, received {player_move.points}pts)"

            for other_move in other_moves:
                other_player = other_move.player
                other_label = self._get_display_name(other_player, observer)
                other_pid = (
                    ""
                    if self.base_game.is_symmetric
                    else f" as Player {other_player.player_id}"
                )

                match_desc += f" vs {other_label} (played {other_move.action.to_token()}{other_pid}, received {other_move.points}pts)"

            lines.append(f"{indent}{branch} [Round {round_idx}] {match_desc}")

            # Recursively expand opponents immediately after this match
            if should_expand_opponents:
                num_opponents = len(other_moves)
                for opp_idx, other_move in enumerate(other_moves):
                    opponent = other_move.player
                    opponent_label = self._get_display_name(opponent, observer)

                    is_last_opponent = opp_idx == num_opponents - 1
                    opponent_indent = child_indent + ("   " if is_last_opponent else "│  ")

                    will_expand = recursion_depth + 1 < self.max_recursion_depth

                    if will_expand:
                        opponent_lines = self._format_player_history_recursive(
                            target_player=opponent,
                            observer=observer,
                            encounter_round=round_idx,
                            indent=opponent_indent,
                            current_round=current_round,
                            recursion_depth=recursion_depth + 1,
                        )
                        if opponent_lines:
                            lines.append(
                                f"{child_indent}└─ History of {opponent_label} before this match:"
                            )
                            lines.extend(opponent_lines)
                        else:
                            # No history, but check for distribution stats
                            dist_lines = self._format_action_distribution(
                                target_player=opponent,
                                observer=observer,
                                current_round=current_round,
                                indent=child_indent,
                            )
                            lines.extend(dist_lines)
                    else:
                        # Max depth reached: just show action distribution
                        dist_lines = self._format_action_distribution(
                            target_player=opponent,
                            observer=observer,
                            current_round=current_round,
                            indent=child_indent,
                        )
                        lines.extend(dist_lines)

        # Show action distribution for the target player at the current indentation level
        player_dist_lines = self._format_action_distribution(
            target_player=target_player,
            observer=observer,
            current_round=current_round,
            indent=indent,
        )
        lines.extend(player_dist_lines)

        return lines

    @override
    def run_tournament(self, agent_cfgs: list[dict]) -> PayoffsBase:
        """Run reputation tournament with proper player ID seating."""
        self.history.clear()
        players = create_players_with_player_id(
            agent_cfgs, self.base_game.num_players
        )
        self.player_to_uid = {
            player: idx + 1 for idx, player in enumerate(players)
        }
        payoffs = self._build_payoffs()

        all_tournament_moves = self._play_matchup(players=players)
        payoffs.add_profile(all_tournament_moves)
        LOGGER.log_record(
            record=self.history.records,
            file_name=self.record_file,
        )
        return payoffs

    @override
    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """
        Play all rounds/batches for the reputation mechanism,
        """
        all_tournament_moves = []
        matcher = RandomMatcher(players)
        with tqdm(
            total=self.num_rounds,
            desc="Reputation rounds",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            for round_idx, matches_group in enumerate(matcher, 1):
                if round_idx > self.num_rounds:
                    break
                for match_up in matches_group:
                    reputation_information = self._build_history_prompts(
                        match_up
                    )
                    moves = self.base_game.play(
                        additional_info=reputation_information,
                        players=match_up,
                    )
                    self.history.append(moves, round_number=round_idx)
                    all_tournament_moves.append(moves)
                pbar.update(1)
        return all_tournament_moves
