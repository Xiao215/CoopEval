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

        # Use class-level player_names mapping (already has "You" for focus player)
        opponent_ids = [f"Player #{opp.uid}" for opp in direct_opponents]
        lines = [
            f"You are playing with {len(direct_opponents)} other players: " + ", ".join(opponent_ids) + ".",
            ""  # Blank line
        ]

        # We'll format focus player's history using the same loop as opponents
        all_players_to_show = [focus_player] + list(direct_opponents)

        # Build each player's history (including focus player)
        for player in all_players_to_show:
            player_label = self.player_names[player.uid]

            recent_rounds = self.history.get_prior_rounds(
                player.name,
                lookback_rounds=0,
                lookup_depth=self.reputation_depth,
            )

            if not recent_rounds:
                player_name_plus_have = f"{player_label} has" if player_label != "You" else f"{player_label} have"
                lines.append(REPUTATION_NO_HISTORY_DESCRIPTION.format(name_plus_have=player_name_plus_have))
                continue

            history_header = f"Your history of play:" if player_label == "You" else f"History of play of {player_label}:"
            lines.append(history_header)

            # === Level 1 history of each current co-player from before the upcoming match ===
            for round_idx, round_moves in reversed(recent_rounds):
                # Get player's move and all others in that match
                player_move = next(m for m in round_moves if m.player_name == player.name)
                other_moves = [m for m in round_moves if m.player_name != player.name]

                # Determine tree branch
                is_last = (round_idx == recent_rounds[0][0])
                main_branch = "└─" if is_last else "├─"
                child_indent = "   " if is_last else "│  "

                match_desc = f"{player_label} ({player_move.action.to_token()}, {player_move.points}pts)"
                for other_move in other_moves:
                    match_desc += f" vs {self.player_names[other_move.uid]} ({other_move.action.to_token()}, {other_move.points}pts)"

                lines.append(f"{main_branch} [Round {round_idx}] {match_desc}")

                # === Level 2 history: history of the past co-players of current co-player ===
                for other_move in other_moves:
                    other_player_name = other_move.player_name
                    other_label = self.player_names[other_move.uid]

                    # Calculate lookback: exclude rounds >= round_idx
                    # Since every player plays every round: lookback = current_round - round_idx
                    lookback = current_round - round_idx

                    # Get other player's history before this round
                    other_history = self.history.get_prior_rounds(
                        other_player_name,
                        lookback_rounds=lookback,
                        lookup_depth=self.reputation_depth,
                    )

                    if other_history:
                        lines.append(f"{child_indent}  └─ History of {other_label} before this match:")

                        for sub_round_idx, sub_round_moves in reversed(other_history):
                            # Get other player's move and all their opponents
                            other_sub_move = next(m for m in sub_round_moves if m.player_name == other_player_name)
                            other_opponents = [m for m in sub_round_moves if m.player_name != other_player_name]

                            # Determine branch
                            is_last_sub = (sub_round_idx == other_history[0][0])
                            sub_branch = "└─" if is_last_sub else "├─"
                            continuation_char = "  " if is_last_sub else "│ "

                            # Format match
                            sub_match_desc = f"{other_label} ({other_sub_move.action.to_token()}, {other_sub_move.points}pts)"
                            for opp_move in other_opponents:
                                sub_match_desc += f" vs {self.player_names[opp_move.uid]} ({opp_move.action.to_token()}, {opp_move.points}pts)"

                            lines.append(f"{child_indent}     {sub_branch} [Round {sub_round_idx}] {sub_match_desc}")

                            # === Level 2.5 history: Action distributions of past co-player to past co-players to upcoming co-player ===
                            # Show action distribution for all opponents
                            for context_opp in other_opponents:
                                context_label = self.player_names[context_opp.uid]

                                # Calculate lookback: exclude rounds >= sub_round_idx
                                # Since every player plays every round: lookback = current_round - sub_round_idx
                                context_lookback = current_round - sub_round_idx

                                # Get action distribution before this sub-round
                                stats = self.history.get_prior_action_distribution(
                                    context_opp.player_name,
                                    lookback_rounds=context_lookback,
                                )

                                if stats:
                                    stats_parts = [f"{k.to_token()}: {v}" for k, v in stats.items()]
                                    stats_str = f"{context_label}'s action distribution: " + ", ".join(stats_parts)
                                else:
                                    context_name_plus_have = f"{context_label} has" if context_label != "You" else f"{context_label} have"
                                    stats_str = REPUTATION_NO_ACTION_DISTRIBUTION_DESCRIPTION.format(name_plus_have=context_name_plus_have)

                                lines.append(f"{child_indent}     {continuation_char}  → Context: {stats_str}")
                    else:
                        other_name_plus_have = f"{other_label} has" if other_label != "You" else f"{other_label} have"
                        lines.append(f"{child_indent}  └─ " + REPUTATION_NO_HISTORY_DESCRIPTION.format(name_plus_have=other_name_plus_have))

            lines.append("")  # Blank line between players

        return "\n".join(lines).strip()

    def run_tournament(self, agent_cfgs: list[dict]) -> PopulationPayoffs:
        """Run reputation tournament with proper player ID seating."""
        players = self._create_players_from_cfgs(agent_cfgs)
        payoffs = self._build_payoffs(players)

        # Build global player name mapping
        self.player_names = {p.uid: f"Player #{p.uid}" for p in players}

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
