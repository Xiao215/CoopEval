"""Common infrastructure for tournament mechanisms."""

import itertools
import random
import time
from abc import ABC, abstractmethod
from typing import Sequence

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.games.base import Action, Game, Move
from src.registry.agent_registry import create_agent


class Mechanism(ABC):
    """Base class for tournament mechanisms that wrap a single game."""

    def __init__(self, base_game: Game):
        self.base_game = base_game

        self.record_file = f"{self.__class__.__name__}_{self.base_game.__class__.__name__}.jsonl"

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
        random.shuffle(
            combo_iter
        )  # The order does not matter, kept just in case

        matchup_labels = [
            " vs ".join(player.name for player in matchup)
            for matchup in combo_iter
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
                if self.base_game.__class__.__name__ == "TrustGame":
                    # Trust game is asymmetric, so also play the reverse
                    self._play_matchup(seat_players[::-1], payoffs)

                dt = time.perf_counter() - t0
                if first_duration is None:
                    first_duration = dt
                    # Rough ETA: match-count * per-match duration
                    est_total = dt * len(combo_iter)
                    print(
                        f"[ETA] ~{est_total/60:.1f} min for "
                        f"{len(combo_iter)} matchups (sequential)."
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

    class History:
        def __init__(self):
            # core raw storage
            self.records: list[list[Move]] = []

            # DP indices
            self.player_round_indices: dict[str, list[int]] = {}
            self.player_cumulative_actions: dict[
                str, list[dict[Action, int]]
            ] = {}

        def append(self, moves: list[Move]) -> None:
            if not moves:
                raise ValueError("Each round must have at least one move")

            round_idx = len(self.records)
            self.records.append(moves)

            # Update DP structures
            for m in moves:
                p = m.player_name
                a = m.action

                # round indices
                self.player_round_indices.setdefault(p, []).append(round_idx)

                # cumulative action counters
                if p not in self.player_cumulative_actions:
                    # first ever round for this player
                    self.player_cumulative_actions[p] = [{a: 1}]
                else:
                    prev = self.player_cumulative_actions[p][-1]
                    new_counter = prev.copy()
                    new_counter[a] += 1
                    self.player_cumulative_actions[p].append(new_counter)

        def get_prior_rounds(
            self,
            player_name: str,
            lookback_rounds: int,
            lookup_depth: int,
        ) -> list[list[Move]]:
            """
            Return the last `lookup_depth` rounds from the player's
            history EXCLUDING the most recent `lookback_rounds` rounds,
            in reverse order.

            Args:
                player_name: Name of the player.
                lookback_rounds: Number of most recent rounds to exclude.
                lookup_depth: Number of rounds to return before the lookback.

            Returns:
                List of rounds (each a list of Moves) for the player.
            """

            if lookback_rounds < 0 or lookup_depth <= 0:
                raise ValueError(
                    "lookback_rounds must be >= 0 and lookup_depth > 0"
                )

            indices = self.player_round_indices.get(player_name, [])
            if not indices:
                return []

            m = len(indices)

            # Define the allowed window by removing most recent LBR
            if lookback_rounds >= m:
                # Early exit if no more record within allowed window
                return []
            else:
                allowed = indices[: m - lookback_rounds]

            # Return last lookup_depth entries of allowed window
            k = min(lookup_depth, len(allowed))
            selected = allowed[-k:]

            return [self.records[i] for i in selected]

        def get_prior_action_distribution(
            self,
            player_name: str,
            lookback_rounds: int,
        ) -> dict[Action, int]:
            """
            Return the action distribution over ALL rounds that occurred
            BEFORE the player's most recent `lookback_rounds` rounds.

            Args:
                player_name: Name of the player.
                lookback_rounds: Number of most recent rounds to exclude.
                lookup_depth: Number of rounds to consider before the lookback.

            Returns:
                Counter dict of Action to counts, or None if no data.
            """
            if lookback_rounds < 0:
                raise ValueError(
                    "lookback_rounds must be >= 0 and lookup_depth > 0"
                )
            indices = self.player_round_indices.get(player_name, [])
            if not indices:
                return {}

            m = len(indices)
            if m - lookback_rounds - 1 < 0:
                return {}

            cumulative_idx = m - lookback_rounds - 1
            return self.player_cumulative_actions[player_name][cumulative_idx]

    def __init__(
        self, base_game: Game, num_rounds: int, discount: float
    ) -> None:
        super().__init__(base_game)
        self.num_rounds = num_rounds
        self.discount = discount
        self.history = self.History()

    def _build_payoffs(self, players: Sequence[Agent]) -> PopulationPayoffs:
        return PopulationPayoffs(players=players, discount=self.discount)
