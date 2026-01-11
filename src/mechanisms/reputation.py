"""Mechanisms that expose behavioural reputation across repeated rounds."""

import itertools
import random
from abc import ABC
from typing import Sequence

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.ranking_evaluations.payoffs_base import PayoffsBase
from src.ranking_evaluations.reputation_payoffs import ReputationPayoffs
from src.mechanisms.base import RepetitiveMechanism
from src.mechanisms.prompts import (
    REPUTATION_ACTION_DISTRIBUTION, REPUTATION_MECHANISM_PROMPT,
    REPUTATION_NO_ACTION_DISTRIBUTION_DESCRIPTION,
    REPUTATION_NO_HISTORY_DESCRIPTION, REPUTATION_PLAYERS_HEADER)
from src.utils.match_scheduler_reputation import RandomMatcher
from src.games.base import Game, Move


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
        if not max_recursion_depth:
            self.max_recursion_depth = lookup_depth + 1
        else:
            self.max_recursion_depth = max_recursion_depth

    def _build_payoffs(self, players: list[Agent]) -> PayoffsBase:
        """Override to use ReputationPayoffs instead of MatchupPayoffs."""
        return ReputationPayoffs(players=players, discount=self.discount)

    def _build_history_prompts(self, players: Sequence[Agent]) -> list[str]:
        """
        Constructs prompts by iterating players and delegating content generation
        entirely to _format_reputation.
        """
        prompts = []

        for focus_player in players:
            round_idx = (
                self.history.get_rounds_played_count(focus_player.name) + 1
            )
            direct_opponents = [p for p in players if p != focus_player]

            # Temporarily update player_names to mark focus player as "You"
            original_name = self.player_names[focus_player.uid]
            self.player_names[focus_player.uid] = "You"

            reputation_text = self._format_reputation(
                focus_player=focus_player,
                direct_opponents=direct_opponents,
                current_round=round_idx
            )

            # Restore original name
            self.player_names[focus_player.uid] = original_name

            base_prompt = REPUTATION_MECHANISM_PROMPT.format(
                round_idx=round_idx,
                discount=int(self.discount * 100),
                history_context=reputation_text,
            )
            prompts.append(base_prompt)

        return prompts

    def _format_reputation(
        self,
        focus_player: Agent,
        direct_opponents: Sequence[Agent],
        current_round: int
    ) -> str:
        """Format the n-order reputation of opponents and self into a tree structure."""

        window_start = max(1, current_round - self.reputation_depth)
        window_end = current_round - 1

        # Header
        opponent_ids = [f"Player #{opp.uid}" for opp in direct_opponents]
        lines = [
            REPUTATION_PLAYERS_HEADER.format(
                num_opponents=len(direct_opponents),
                opponent_ids=", ".join(opponent_ids)
            ),
            ""
        ]

        # Process each player (focus player + opponents)
        for player in [focus_player] + list(direct_opponents):
            player_label = self.player_names[player.uid]

            # Get player's history within window
            recent_rounds = self.history.get_prior_rounds(
                player.name,
                lookback_rounds=0,
                lookup_depth=self.reputation_depth
            )
            filtered_rounds = [
                (idx, moves) for (idx, moves) in recent_rounds
                if window_start <= idx <= window_end
            ]

            if not filtered_rounds:
                player_name_plus_have = f"{player_label} has" if player_label != "You" else f"{player_label} have"
                lines.append(REPUTATION_NO_HISTORY_DESCRIPTION.format(name_plus_have=player_name_plus_have))
                continue

            # Header for this player
            history_header = (
                f"History of play of {player_label}:"
                if player_label != "You"
                else f"Your history of play:"
            )
            lines.append(history_header)

            # Recursively format player's history (includes action distribution)
            player_lines = self._format_player_history_recursive(
                player=player,
                encounter_round=current_round,
                indent="",
                current_round=current_round,
                recursion_depth=0
            )
            lines.extend(player_lines)

            lines.append("")  # Blank line between players

        return "\n".join(lines).strip()

    def _format_action_distribution(
        self,
        player: Agent,
        player_label: str,
        current_round: int,
        indent: str = "",
    ) -> list[str]:
        """
        Format action distribution for a player for rounds 1 to window_start-1.

        Args:
            player: The player whose distribution to format
            player_label: Display label for the player (e.g., "You", "Player #5")
            current_round: The upcoming round number
            indent: Indentation prefix for the output line

        Returns:
            List of formatted lines (typically one line or empty if disabled)
        """
        if not self.include_prior_distributions:
            return []

        lines = []
        window_start = max(1, current_round - self.reputation_depth)

        if window_start > 1:
            lookback_for_distribution = current_round - window_start
            stats = self.history.get_prior_action_distribution(
                player.name,
                lookback_rounds=lookback_for_distribution
            )

            potential_player_id = "" if self.base_game.is_symmetric else f" (as Player {player.player_id})"
            player_name_plus_have = f"{player_label}{potential_player_id} has" if player_label != "You" else f"{player_label}{potential_player_id} have"

            if stats:
                stats_str = ", ".join(f"{count} time{'s' if count != 1 else ''} {action.to_token()}" for action, count in sorted(
                        stats.items(), key=lambda kv: str(kv[0]))) + "."
                lines.append(
                    REPUTATION_ACTION_DISTRIBUTION.format(
                        indent=indent,
                        name_plus_have=player_name_plus_have,
                        window_start_minus_one=window_start - 1,
                        stats_str=stats_str
                    )
                )
            else:
                lines.append(
                    REPUTATION_NO_ACTION_DISTRIBUTION_DESCRIPTION.format(
                        indent=indent,
                        window_start_minus_one=window_start - 1,name_plus_have=player_name_plus_have)
                )

        return lines

    def _format_player_history_recursive(
        self,
        player: Agent,
        encounter_round: int,
        indent: str,
        current_round: int,
        recursion_depth: int,
    ) -> list[str]:
        """
        Recursively format a player's match history within the global time window.

        Args:
            player: The player whose history to format
            encounter_round: The round when this player was encountered (exclusive upper bound)
            indent: Current indentation string for tree formatting
            current_round: The upcoming round number (for time window calculation)
            recursion_depth: Current depth of recursion (0 for top-level players)

        Returns:
            List of formatted lines showing the player's history
        """
        lines = []

        # Calculate time window
        window_start = max(1, current_round - self.reputation_depth)
        window_end = current_round - 1

        # Get player's history before encounter_round
        lookback = current_round - encounter_round
        recent_rounds = self.history.get_prior_rounds(player.name, lookback, self.reputation_depth)

        # Filter to time window and before encounter
        filtered_rounds = [
            (idx, moves) for (idx, moves) in recent_rounds
            if window_start <= idx <= window_end and idx < encounter_round
        ]

        if not filtered_rounds:
            return lines

        player_label = self.player_names[player.uid]
        should_expand_opponents = (recursion_depth < self.max_recursion_depth)

        # Format each match and expand opponents immediately after
        for i, (round_idx, round_moves) in enumerate(reversed(filtered_rounds)):
            # Get this player's move and all other moves
            player_move = next(m for m in round_moves if m.player_name == player.name)
            other_moves = [m for m in round_moves if m.player_name != player.name]

            # Determine tree branch
            is_last = (i == len(filtered_rounds) - 1)
            branch = "└─" if is_last else "├─"
            child_indent = indent + ("   " if is_last else "│  ")

            # Format match description
            potential_player_id = "" if self.base_game.is_symmetric else f" as Player {player.player_id}"
            match_desc = f"{player_label} (played {player_move.action.to_token()}{potential_player_id}, received {player_move.points}pts)"
            for other_move in other_moves:
                other_player = self.players_by_uid[other_move.uid]
                match_desc += f" vs {self.player_names[other_move.uid]} (played {other_move.action.to_token()} as Player {other_player.player_id}, received {other_move.points}pts)"

            lines.append(f"{indent}{branch} [Round {round_idx}] {match_desc}")

            # Recursively expand opponents immediately after this match (only if not at max depth)
            if should_expand_opponents:
                num_opponents = len(other_moves)
                for opp_idx, other_move in enumerate(other_moves):
                    # Find the Agent object for this opponent using global lookup by UID
                    opponent = self.players_by_uid[other_move.uid]
                    opponent_label = self.player_names[opponent.uid]

                    # Determine if this is the last opponent (for proper tree formatting)
                    is_last_opponent = (opp_idx == num_opponents - 1)
                    opponent_indent = child_indent + ("   " if is_last_opponent else "│  ")

                    # Check if we'll actually expand (not at the last recursion level)
                    will_expand = (recursion_depth + 1 < self.max_recursion_depth)

                    if will_expand:
                        # Get opponent lines first to check if there's anything to show
                        opponent_lines = self._format_player_history_recursive(
                            player=opponent,
                            encounter_round=round_idx,
                            indent=opponent_indent,
                            current_round=current_round,
                            recursion_depth=recursion_depth + 1
                        )
                        # Only show header if there's actual content to display
                        if opponent_lines:
                            lines.append(f"{child_indent}└─ History of {opponent_label} before this match:")
                            lines.extend(opponent_lines)
                        else:
                            # No history to show, but still show action distribution if available
                            opponent_dist_lines = self._format_action_distribution(
                                player=opponent,
                                player_label=opponent_label,
                                current_round=current_round,
                                indent=child_indent
                            )
                            lines.extend(opponent_dist_lines)
                    else:
                        # At max depth: just show action distribution context without header
                        opponent_dist_lines = self._format_action_distribution(
                            player=opponent,
                            player_label=opponent_label,
                            current_round=current_round,
                            indent=child_indent
                        )
                        lines.extend(opponent_dist_lines)

        # Show action distribution for THIS player, if self.include_prior_distributions == True
        # Always show distribution for rounds 1 to window_start-1 (before the visible window)
        player_dist_lines = self._format_action_distribution(
            player=player,
            player_label=player_label,
            current_round=current_round,
            indent=indent
        )
        lines.extend(player_dist_lines)

        return lines

    def run_tournament(self, agent_cfgs: list[dict]) -> PayoffsBase:
        """Run reputation tournament with proper player ID seating."""
        players = self._create_players_from_cfgs(agent_cfgs)
        payoffs = self._build_payoffs(players)

        # Build global player name mapping
        self.player_names = {p.uid: f"Agent #{p.uid}" for p in players}

        # Build global player lookup by UID (for recursive history expansion)
        self.players_by_uid = {p.uid: p for p in players}

        # Group players by their player_id to ensure proper seating in matchups
        k = self.base_game.num_players
        players_by_id = [
            [p for p in players if p.player_id == player_id]
            for player_id in range(1, k + 1)
        ]

        # Initialize matcher
        matcher = RandomMatcher(players_by_id)

        # Clear history at the start of tournament
        self.history.clear()

        all_tournament_moves = []

        # Loop through num_rounds
        with tqdm(
            total=self.num_rounds,
            desc="Reputation rounds",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            for round_idx, matches_group in enumerate(matcher, 1):
                if round_idx > self.num_rounds:
                    break

                # Play this round's matches (returns all moves from this round)
                round_moves = self._play_matchup(matches_group, round_idx)
                all_tournament_moves.append(round_moves)

                pbar.update(1)

        payoffs.add_profile(all_tournament_moves)

        return payoffs

    def _play_matchup(self, matches_group: list[list[Agent]], round_idx: int) -> list[Move]:
        """
        Play a single round of matches for all matchups in matches_group.

        Args:
            matches_group: List of matchups, where each matchup is a list of agents
            round_idx: The current round number

        Returns:
            Flattened list of all moves from this round (across all matches).
        """
        round_moves = []

        print("Match Group round", round_idx)
        print(
            "Match Group:",
            " | ".join(
                ", ".join(p.name for p in match_up)
                for match_up in matches_group
            ),
        )

        for match_up in matches_group:
            reputation_information = self._build_history_prompts(match_up)

            # Play the game
            moves = self.base_game.play(
                additional_info=reputation_information,
                players=match_up,
            )
            self.history.append(moves, round_number=round_idx)
            round_moves.extend(moves)  # Flatten into single list

        return round_moves
