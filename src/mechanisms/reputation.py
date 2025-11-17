"""Mechanisms that expose behavioural reputation across repeated rounds."""

import itertools
import random
from abc import ABC
from collections import defaultdict
from typing import Sequence

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.games.base import Game, Move
from src.logger_manager import LOGGER
from src.mechanisms.base import RepetitiveMechanism
from src.registry.agent_registry import create_agent

random.seed(42)


class Reputation(RepetitiveMechanism, ABC):
    """
    Reputation mechanism that makes each players' reputation visible to all players.
    """

    def __init__(self, base_game: Game, num_rounds: int, discount: float):
        super().__init__(base_game, num_rounds, discount)
        self.matchup_workers = 1

    def _format_reputation(self, agents: Sequence[Agent]) -> str:
        """Format the n-hop reputational awareness information into a string."""

    # def run_tournament(self, agent_cfgs: Sequence[dict]) -> PopulationPayoffs:
    #     """Run a multi-round tournament while tracking reputation updates."""
    #     players = [
    #         create_agent(cfg)
    #         for cfg in agent_cfgs
    #         for _ in range(self.base_game.num_players)
    #     ]
    #     payoffs = self._build_payoffs(players)

    #     self._play_matchups(players, payoffs)

    #     return payoffs

    # def _play_matchups(
    #     self, players: Sequence[Agent], payoffs: PopulationPayoffs
    # ) -> None:
    #     """Play all matchups while updating reputation after each round."""
    #     num_players = self.base_game.num_players
    #     all_matchups = list(itertools.combinations(players, r=num_players))
    #     random.shuffle(all_matchups)

    #     k = self.base_game.num_players
    #     matchups = [
    #         tuple(lineup) for lineup in itertools.combinations(players, k)
    #     ]
    #     matchup_histories: defaultdict[tuple[int, ...], list[list[Move]]] = (
    #         defaultdict(list)
    #     )

    #     round_iter = tqdm(
    #         range(1, self.num_rounds + 1),
    #         desc=f"Running Reputation Mechanism for {self.base_game.__class__.__name__}",
    #     )

    #     for round_idx in round_iter:
    #         # Shuffle matchups before each round, even though order does not impact outcomes
    #         matchups_this_round = list(matchups)
    #         random.shuffle(matchups_this_round)

    #         round_matches: list[dict[str, object]] = []

    #         for matchup in matchups_this_round:
    #             reputation_information = self._format_reputation(matchup)
    #             moves = self.base_game.play(
    #                 additional_info=reputation_information,
    #                 players=matchup,
    #             )

    #             round_matches.append(
    #                 {
    #                     "players": [move.player_name for move in moves],
    #                     "moves": [move.to_dict() for move in moves],
    #                 }
    #             )

    #             for move in moves:
    #                 self._update_reputation(move.player_name, move.action.value)

    #         LOGGER.log_record(
    #             record={"round": round_idx, "matchups": round_matches},
    #             file_name=self.record_file,
    #         )

    #     for moves_over_rounds in matchup_histories.values():
    #         payoffs.add_profile(moves_over_rounds)
