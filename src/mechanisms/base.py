import itertools
import json
import math
import os
import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Sequence, Any

from tqdm import tqdm

from src.logger_manager import log_dir
from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.games.base import Game

random.seed(42)


class Mechanism(ABC):

    def __init__(self, base_game: Game):
        self.base_game = base_game

        self.record_file = (
            f"{self.__class__.__name__}_{self.base_game.__class__.__name__}.jsonl"
        )

    def _build_payoffs(self, agent_names: list[str]) -> PopulationPayoffs:
        return PopulationPayoffs(agent_names=agent_names)

    def run_tournament(self, agents: Sequence[Agent]) -> PopulationPayoffs:
        """Run the mechanism over the base game across all players."""
        payoffs = self._build_payoffs(agent_names=[agent.name for agent in agents])

        k = self.base_game.num_players
        n = len(agents)
        total_matches = math.comb(n, k)
        combo_iter = list(itertools.combinations(agents, k))
        random.shuffle(combo_iter)  # The order does not matter, kept just in case

        inner_tqdm_bar = tqdm(
            combo_iter,
            desc="Tournaments",
            total=total_matches,
            leave=False,
            position=1,
        )
        for players in inner_tqdm_bar:
            matchup = " vs ".join(agent.name for agent in players)
            inner_tqdm_bar.set_description(f"Match: {matchup}")
            self._play_matchup(players, payoffs)
        return payoffs

    @abstractmethod
    def _play_matchup(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> Any:
        """Play match(es) between the given players."""
        raise NotImplementedError


class RepetitiveMechanism(Mechanism):
    """A mechanism that repeats the game multiple times."""

    def __init__(self, base_game: Game, num_rounds: int, discount: float) -> None:
        super().__init__(base_game)
        self.num_rounds = num_rounds
        self.discount = discount

    def _build_payoffs(self, agent_names: list[str]) -> PopulationPayoffs:
        return PopulationPayoffs(agent_names=agent_names, discount=self.discount)


class NoMechanism(Mechanism):
    """A mechanism that does nothing."""

    def _play_matchup(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> Any:
        """Run the base game without any modifications."""
        players_moves = self.base_game.play(additional_info="None.", players=players)
        payoff_map = {
            player.name: move.points for player, move in zip(players, players_moves)
        }
        payoffs.add_profile(payoff_map)
        return players_moves
