"""Common infrastructure for tournament mechanisms."""

import itertools
import random
import time
from abc import ABC, abstractmethod
from typing import Sequence

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.games.base import Game
from src.registry.agent_registry import create_agent


class Mechanism(ABC):
    """Base class for tournament mechanisms that wrap a single game."""

    def __init__(self, base_game: Game):
        self.base_game = base_game

        self.record_file = (
            f"{self.__class__.__name__}_{self.base_game.__class__.__name__}.jsonl"
        )

    def _build_payoffs(self, players: Sequence[Agent]) -> PopulationPayoffs:
        return PopulationPayoffs(players=players)

    def run_tournament(self, agent_cfgs: Sequence[dict]) -> PopulationPayoffs:
        """Run the mechanism over the base game across all players."""
        players = [
            create_agent(cfg)
            for cfg in agent_cfgs
            for _ in range(self.base_game.num_players)
        ]
        payoffs = self._build_payoffs(players)

        k = self.base_game.num_players
        combo_iter = list(itertools.combinations(players, k))
        random.shuffle(combo_iter)  # The order does not matter, kept just in case

        matchup_labels = [
            " vs ".join(player.name for player in matchup) for matchup in combo_iter
        ]

        first_duration = None
        with tqdm(
            total=len(combo_iter),
            desc="Tournaments",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            for seat_players, matchup_label in zip(
                combo_iter, matchup_labels, strict=True
            ):
                pbar.set_postfix_str(matchup_label, refresh=False)
                t0 = time.perf_counter()
                self._play_matchup(seat_players, payoffs)
                dt = time.perf_counter() - t0
                if first_duration is None:
                    first_duration = dt
                    # Rough ETA: match-count * per-match duration
                    est_total = dt * len(combo_iter)
                    print(
                        f"[ETA] ~{est_total/60:.1f} min for {len(combo_iter)} matchups (sequential)."
                    )
                pbar.update(1)
        return payoffs

    @abstractmethod
    def _play_matchup(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> None:
        """Play match(es) between the given players."""
        raise NotImplementedError


class RepetitiveMechanism(Mechanism):
    """A mechanism that repeats the game multiple times."""

    def __init__(self, base_game: Game, num_rounds: int, discount: float) -> None:
        super().__init__(base_game)
        self.num_rounds = num_rounds
        self.discount = discount

    def _build_payoffs(self, players: Sequence[Agent]) -> PopulationPayoffs:
        return PopulationPayoffs(players=players, discount=self.discount)


class NoMechanism(Mechanism):
    """A mechanism that does nothing."""

    def _play_matchup(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> None:
        """Run the base game without any modifications."""
        moves = self.base_game.play(additional_info="None.", players=players)
        payoffs.add_profile([moves])
