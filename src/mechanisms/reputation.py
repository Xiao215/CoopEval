"""Mechanisms that expose behavioural reputation across repeated rounds."""

import itertools
import random

from abc import ABC
from typing import Sequence

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.ranking_evaluations.population_payoffs import PopulationPayoffs
from src.mechanisms.base import RepetitiveMechanism
from src.games.base import Game, Move
from utils.match_scheduler_reputation import RandomMatcher, RoundRobin
from src.mechanisms.prompts import (
    REPUTATION_MECHANISM_PROMPT,
    REPUTATION_NO_HISTORY_DESCRIPTION,
    REPUTATION_NO_ACTION_DISTRIBUTION_DESCRIPTION,
)


random.seed(42)


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
        num_repeat_experiment: int | None = None,
    ) -> None:
        super().__init__(base_game, num_rounds, discount)
        self.reputation_depth = lookup_depth
        self.num_repeat_experiment = num_repeat_experiment

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
            f"You are playing with {len(direct_opponents)} other players: " + ", ".join(opponent_ids) + ".",
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
            history_header = f"Your history of play:" if player_label == "You" else f"History of play of {player_label}:"
            lines.append(history_header)

            # Recursively format player's history (includes action distribution)
            player_lines = self._format_player_history_recursive(
                player=player,
                encounter_round=current_round,
                indent="",
                current_round=current_round
            )
            lines.extend(player_lines)

            lines.append("")  # Blank line between players

        return "\n".join(lines).strip()

    def _format_player_history_recursive(
        self,
        player: Agent,
        encounter_round: int,
        indent: str,
        current_round: int,
    ) -> list[str]:
        """
        Recursively format a player's match history within the global time window.

        Args:
            player: The player whose history to format
            encounter_round: The round when this player was encountered (exclusive upper bound)
            indent: Current indentation string for tree formatting
            current_round: The upcoming round number (for time window calculation)

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

        # Collect opponents to expand (we'll process them after showing all matches)
        opponents_to_expand = []

        # Format each match
        for i, (round_idx, round_moves) in enumerate(reversed(filtered_rounds)):
            # Get this player's move and all other moves
            player_move = next(m for m in round_moves if m.player_name == player.name)
            other_moves = [m for m in round_moves if m.player_name != player.name]

            # Determine tree branch
            is_last = (i == len(filtered_rounds) - 1)
            branch = "└─" if is_last else "├─"
            child_indent = indent + ("   " if is_last else "│  ")

            # Format match description
            match_desc = f"{player_label} ({player_move.action.to_token()}, {player_move.points}pts)"
            for other_move in other_moves:
                match_desc += f" vs {self.player_names[other_move.uid]} ({other_move.action.to_token()}, {other_move.points}pts)"

            lines.append(f"{indent}{branch} [Round {round_idx}] {match_desc}")

            # Collect opponents for recursive expansion
            for other_move in other_moves:
                # Find the Agent object for this opponent using global lookup
                opponent = self.players_by_name.get(other_move.player_name)
                if opponent:
                    opponents_to_expand.append((opponent, round_idx, child_indent))

        # Recursively expand opponents
        for opponent, encounter_rd, child_ind in opponents_to_expand:
            opponent_label = self.player_names[opponent.uid]
            lines.append(f"{child_ind}└─ History of {opponent_label} before this match:")

            opponent_lines = self._format_player_history_recursive(
                player=opponent,
                encounter_round=encounter_rd,
                indent=child_ind + "   ",
                current_round=current_round
            )
            lines.extend(opponent_lines)

        # Show action distribution for THIS player (for rounds before encounter)
        if encounter_round > window_start:
            # We want distribution for rounds before the time window we just showed
            # If indent is empty, this is a top-level player: show distribution for rounds [1, window_start-1]
            # If indent is not empty, this is a nested player: show distribution for rounds before encounter_round

            if indent == "":
                # Top-level player: distribution for rounds before the window
                lookback_for_distribution = current_round - window_start
            else:
                # Nested player: distribution for rounds before encounter
                lookback_for_distribution = current_round - encounter_round

            stats = self.history.get_prior_action_distribution(
                player.name,
                lookback_rounds=lookback_for_distribution
            )

            if stats:
                stats_parts = [f"{k.to_token()}: {v}" for k, v in stats.items()]

                if indent == "":
                    # Top-level: no prefix indentation
                    stats_str = f"{player_label}'s action distribution for rounds before window: " + ", ".join(stats_parts)
                    lines.append(stats_str)
                else:
                    # Nested: with "→ Context:" prefix
                    stats_str = f"{player_label}'s action distribution: " + ", ".join(stats_parts)
                    lines.append(f"{indent}→ Context: {stats_str}")
            else:
                player_name_plus_have = f"{player_label} has" if player_label != "You" else f"{player_label} have"
                distribution_msg = REPUTATION_NO_ACTION_DISTRIBUTION_DESCRIPTION.format(name_plus_have=player_name_plus_have)

                if indent == "":
                    # Top-level: no prefix indentation
                    lines.append(distribution_msg)
                else:
                    # Nested: with "→ Context:" prefix
                    lines.append(f"{indent}→ Context: {distribution_msg}")

        return lines

    def run_tournament(self, agent_cfgs: list[dict]) -> PopulationPayoffs:
        """Run reputation tournament with proper player ID seating."""
        players = self._create_players_from_cfgs(agent_cfgs)
        payoffs = self._build_payoffs(players)

        # Build global player name mapping
        self.player_names = {p.uid: f"Player #{p.uid}" for p in players}

        # Build global player lookup by name (for recursive history expansion)
        self.players_by_name = {p.name: p for p in players}

        # Group players by their player_id to ensure proper seating in matchups
        k = self.base_game.num_players
        players_by_id = [
            [p for p in players if p.player_id == player_id]
            for player_id in range(1, k + 1)
        ]

        # Compute num_repeat_experiment if not provided
        # Default: (#LLM models)^(num_players - 1)
        if self.num_repeat_experiment is None:
            num_llm_models = len(agent_cfgs)
            num_repeat_experiment = num_llm_models ** (k - 1)
        else:
            num_repeat_experiment = self.num_repeat_experiment

        all_tournament_moves = []
        for _ in tqdm(
            range(num_repeat_experiment),
            desc=f"Running {num_repeat_experiment} reputation iterations",
            leave=True,
        ):
            # Clear history at the start of each iteration
            self.history.clear()
            round_moves = self._play_matchup(players_by_id)
            all_tournament_moves.extend(round_moves)

        payoffs.add_profile(all_tournament_moves)

        return payoffs

    def _play_matchup(self, players_by_id: list[list[Agent]]) -> list[list[Move]]:
        """
        Play num_rounds rounds of matchups using RandomMatcher iterator.

        Args:
            players_by_id: List of player groups, where players_by_id[i] contains
                          all agents that should play in seat i+1 (player_id=i+1)

        Returns:
            List of moves from num_rounds rounds of play.
        """
        batch_moves = []
        matcher = RandomMatcher(players_by_id)

        # Iterate through num_rounds rounds with random matchings
        with tqdm(
            total=self.num_rounds,
            desc="Random matchups",
            leave=False,
            dynamic_ncols=True,
        ) as pbar:
            for round_idx, matches_group in enumerate(matcher, 1):
                if round_idx > self.num_rounds:
                    break
                print("Match Group round", round_idx)
                print(
                    "Match Group:",
                    " | ".join(
                        ", ".join(p.name for p in match_up)
                        for match_up in matches_group
                    ),
                )
                for match_up in matches_group:
                    matchup_label = " vs ".join(p.name for p in match_up)
                    pbar.set_postfix_str(
                        f"[Round {round_idx}/{self.num_rounds}] {matchup_label}",
                        refresh=False,
                    )

                    reputation_information = self._build_history_prompts(match_up)

                    # Play the game
                    moves = self.base_game.play(
                        additional_info=reputation_information,
                        players=match_up,
                    )
                    self.history.append(moves, round_number=round_idx)
                    batch_moves.append(moves)

                pbar.update(1)

        return batch_moves
