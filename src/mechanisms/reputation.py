"""Mechanisms that expose behavioural reputation across repeated rounds."""

import random

from abc import ABC
from typing import Sequence

from src.agents.agent_manager import Agent
from src.ranking_evaluations.population_payoffs import PopulationPayoffs
from src.mechanisms.base import RepetitiveMechanism
from src.games.base import Game
from src.utils.round_robin import RoundRobin


random.seed(42)


class Reputation(RepetitiveMechanism, ABC):
    """
    Reputation mechanism that makes each players' reputation visible to all players.
    """

    def __init__(self, base_game: Game, num_rounds: int, discount: float):
        super().__init__(base_game, num_rounds, discount)
        self.matchup_workers = 1
        self.reputation_depth = 3

    def _format_reputation(self, players: Agent) -> str:
        """
        Format the n-order reputation of each player into a tree structure.
        """

        # 1. Get the main player's recent history (Current view)
        # We use lookback=0 because we want the most recent events
        recent_rounds = self.history.get_prior_rounds(
            players.name, lookback_rounds=0, lookup_depth=self.reputation_depth
        )

        if not recent_rounds:
            return f"Reputation for {players.name}: No history available."
        lines = [f"Reputation History for {players.name} (Most recent first):"]

        # 2. Iterate backwards (Most recent -> Oldest)
        reversed_history = list(enumerate(reversed(recent_rounds), 1))

        for rounds_ago, round_moves in reversed_history:
            main_branch = "└─" if rounds_ago == len(recent_rounds) else "├─"
            child_indent = "   " if rounds_ago == len(recent_rounds) else "│  "

            # Identify self and opponent in this specific round
            # Assuming round_moves is a list of 2 Moves
            my_move = next(
                m for m in round_moves if m.player_name == players.name
            )
            opp_move = next(
                m for m in round_moves if m.player_name != players.name
            )

            opp_name = opp_move.player_name

            # --- Level 1: Main Player's History ---
            lines.append(
                f"{main_branch} Past {rounds_ago} round(s): {players.name} vs {opp_name}. "
                f"{players.name} played {my_move.action} ({my_move.points} pts), "
                f"{opp_name} played {opp_move.action} ({opp_move.points} pts)."
            )

            # --- Level 2: Opponent's History (Relative to THAT moment) ---
            # We need to see what the opponent looked like *before* this match occurred.
            # So lookback = rounds_ago.
            opp_history = self.history.get_prior_rounds(
                opp_name,
                lookback_rounds=rounds_ago,
                lookup_depth=self.reputation_depth,
            )

            if opp_history:
                lines.append(
                    f"{child_indent}  └─ History of {opp_name} before this match:"
                )

                # Iterate through the opponent's nested history
                # detailed_rounds_ago tracks how far back from the MAIN match this was
                for sub_idx, sub_round in enumerate(reversed(opp_history), 1):
                    sub_branch = "└─" if sub_idx == len(opp_history) else "├─"
                    sub_opp_move = next(
                        m for m in sub_round if m.player_name != opp_name
                    )
                    sub_target_move = next(
                        m for m in sub_round if m.player_name == opp_name
                    )
                    third_party_name = sub_opp_move.player_name

                    # --- Level 3: The Context (Action Distribution) ---
                    # We want the stats of the 3rd party *before* they met the 2nd party.
                    # Total lookback = (Main Player offset) + (Opponent offset)
                    total_lookback = rounds_ago + sub_idx

                    stats = self.history.get_prior_action_distribution(
                        third_party_name, lookback_rounds=total_lookback
                    )
                    stats_str = (
                        ", ".join([f"{k}: {v}" for k, v in stats.items()])
                        if stats is not None
                        else "No Data"
                    )

                    lines.append(
                        f"{child_indent}     {sub_branch} Past {sub_idx} round(s): {opp_name} vs {third_party_name}. "
                        f"{opp_name}: {sub_target_move.action}, {third_party_name}: {sub_opp_move.action}. "
                        f"[Context: {third_party_name} had distr {stats_str}]"
                    )
            else:
                lines.append(
                    f"{child_indent}  └─ (No prior history for {opp_name} at this time)"
                )

        return "\n".join(lines)

    def run_tournament(self, agent_cfgs: Sequence[dict]) -> PopulationPayoffs:
        players = self._create_players_from_cfgs(agent_cfgs)
        payoffs = self._build_payoffs(players)

        for _ in range(self.num_rounds):
            self._play_matchup(players, payoffs)

        return payoffs

    def _play_matchup(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> None:
        round_robin_scheduler = RoundRobin(players, group_size=2)
        for matches_groups in round_robin_scheduler.generate_schedule():
            for match_up in matches_groups:
                reputation_information = [
                    self._format_reputation(player) for player in match_up
                ]
                moves = self.base_game.play(
                    additional_info=reputation_information,
                    players=match_up,
                )
                self.history.append(moves)
