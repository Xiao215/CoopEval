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
        num_repeat_experiment: int | None = None,
    ) -> None:
        super().__init__(base_game, num_rounds, discount)
        self.matchup_workers = 1
        self.reputation_depth = 3
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

            reputation_text = self._format_reputation(direct_opponents)
            base_prompt = REPUTATION_MECHANISM_PROMPT.format(
                round_idx=round_idx,
                discount=int(self.discount * 100),
                history_context=reputation_text,
            )
            prompts.append(base_prompt)

        return prompts

    def _format_reputation(self, direct_opponents: Sequence[Agent]) -> str:
        """
        Format the n-order reputation of the *direct_opponents* into a tree structure.
        """
        opponent_ids = [f"Player #{opp.uid}" for opp in direct_opponents]
        lines = [
            f"You are playing with {len(direct_opponents)} other players: {opponent_ids}.",
        ]

        for direct_opponent in direct_opponents:
            direct_opponent_reputation_lines = []

            recent_rounds = self.history.get_prior_rounds(
                direct_opponent.name,
                lookback_rounds=0,
                lookup_depth=self.reputation_depth,
            )

            if not recent_rounds:
                lines.append(
                    f"Player #{direct_opponent.uid}'s history of play: {REPUTATION_NO_HISTORY_DESCRIPTION.format(
                        opponent_name=f'Player #{direct_opponent.uid}')}"
                )
                continue

            direct_opponent_reputation_lines.append(
                f"Player #{direct_opponent.uid}'s history of play:"
            )

            reversed_history = list(enumerate(reversed(recent_rounds), 1))

            for rounds_ago, round_moves in reversed_history:
                main_branch = "└─" if rounds_ago == len(recent_rounds) else "├─"
                child_indent = (
                    "   " if rounds_ago == len(recent_rounds) else "│  "
                )

                # Identify moves in the Direct Opponent's match
                direct_opp_move = next(
                    m
                    for m in round_moves
                    if m.player_name == direct_opponent.name
                )
                first_order_opp_move = next(
                    m
                    for m in round_moves
                    if m.player_name != direct_opponent.name
                )

                first_order_opp_label = f"Player #{first_order_opp_move.uid}"

                # --- Level 1: Direct Opponent vs First Order Opponent ---
                direct_opponent_reputation_lines.append(
                    f"{main_branch} [{rounds_ago} round(s) ago] "
                    f"Player #{direct_opponent.uid} ({direct_opp_move.action.to_token()}, {direct_opp_move.points}pts) vs "
                    f"{first_order_opp_label} ({first_order_opp_move.action.to_token()}, {first_order_opp_move.points}pts)"
                )

                # --- Level 2: First Order Opponent's History ---
                first_order_opp_name = first_order_opp_move.player_name

                first_order_opp_history = self.history.get_prior_rounds(
                    first_order_opp_name,
                    lookback_rounds=rounds_ago,  # TODO: this could be buggy if two player have different number of rounds. Need to fix!!!
                    lookup_depth=self.reputation_depth,
                )

                if first_order_opp_history:
                    direct_opponent_reputation_lines.append(
                        f"{child_indent}  └─ History of {first_order_opp_label} before this match:"
                    )

                    for sub_idx, sub_round in enumerate(
                        reversed(first_order_opp_history), 1
                    ):
                        sub_branch = (
                            "└─"
                            if sub_idx == len(first_order_opp_history)
                            else "├─"
                        )

                        first_order_sub_move = next(
                            m
                            for m in sub_round
                            if m.player_name == first_order_opp_name
                        )
                        second_order_opp_move = next(
                            m
                            for m in sub_round
                            if m.player_name != first_order_opp_name
                        )

                        second_order_opp_name = (
                            second_order_opp_move.player_name
                        )
                        second_order_opp_label = (
                            f"Player #{second_order_opp_move.uid}"
                        )

                        # --- Level 3: Context (Second Order Opponent's Stats) ---
                        total_lookback = rounds_ago + sub_idx
                        stats = self.history.get_prior_action_distribution(
                            second_order_opp_name,
                            lookback_rounds=total_lookback,
                        )

                        if stats:
                            stats_parts = []
                            for k, v in stats.items():
                                key_str = (
                                    k.to_token()
                                    if hasattr(k, "to_token")
                                    else str(k)
                                )
                                stats_parts.append(f"{key_str}: {v}")
                            stats_str = (
                                f"{second_order_opp_label}'s action distribution: "
                                + ", ".join(stats_parts)
                            )
                        else:
                            stats_str = REPUTATION_NO_ACTION_DISTRIBUTION_DESCRIPTION.format(
                                opponent_name=second_order_opp_label
                            )

                        # Determine continuation indent for context line
                        continuation_char = "│ " if sub_branch == "├─" else "  "

                        direct_opponent_reputation_lines.append(
                            f"{child_indent}     {sub_branch} [{sub_idx} round(s) ago] "
                            f"{first_order_opp_label} ({first_order_sub_move.action.to_token()}, {first_order_sub_move.points}pts) vs "
                            f"{second_order_opp_label} ({second_order_opp_move.action.to_token()}, {second_order_opp_move.points}pts)"
                        )
                        direct_opponent_reputation_lines.append(
                            f"{child_indent}     {continuation_char} → Context: {stats_str}"
                        )
                else:
                    direct_opponent_reputation_lines.append(
                        f"{child_indent}  └─ {REPUTATION_NO_HISTORY_DESCRIPTION.format(
                            opponent_name=first_order_opp_label)
                        }"
                    )

            lines.append("\n".join(direct_opponent_reputation_lines).strip())

        return "\n\n".join(lines).strip()

    def run_tournament(self, agent_cfgs: list[dict]) -> PopulationPayoffs:
        """Run reputation tournament with proper player ID seating."""
        players = self._create_players_from_cfgs(agent_cfgs)
        payoffs = self._build_payoffs(players)

        # Group players by their player_id to ensure proper seating in matchups
        k = self.base_game.num_players
        players_by_id = [
            [p for p in players if p.player_id == player_id]
            for player_id in range(1, k + 1)
        ]

        K = 2
        all_tournament_moves = []
        for _ in tqdm(
            range(K),
            desc=f"Running {K} reputation iterations",
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
                    self.history.append(moves)
                    batch_moves.append(moves)

                pbar.update(1)

        return batch_moves
