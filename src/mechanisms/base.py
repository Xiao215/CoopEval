"""Common infrastructure for tournament mechanisms."""

from collections import Counter, defaultdict
import itertools
import random
import time
from abc import ABC, abstractmethod
from typing import Sequence, Iterator

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

    def _create_players_from_cfgs(
        self, agent_cfgs: Sequence[dict]
    ) -> list[Agent]:
        """Create players from the given agent configurations."""
        players = [
            create_agent(cfg)
            for cfg in agent_cfgs
            for _ in range(self.base_game.num_players)
        ]
        return players

    def run_tournament(self, agent_cfgs: Sequence[dict]) -> PopulationPayoffs:
        """Run the mechanism over the base game across all players."""
        players = self._create_players_from_cfgs(agent_cfgs)
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
        """History of moves across multiple rounds."""
        def __init__(self):
            # List of all rounds information.
            # Note, the indices is arbitrary and only used for lookup.
            self.records: list[list[Move]] = []

            # Maps player_name -> List of global round indices they participated in
            self.player_round_indices: dict[str, list[int]] = defaultdict(list)

            # Maps player_name -> List of cumulative distributions at each step
            # Index i corresponds to the state after the player's i-th game
            self.player_cumulative_actions: dict[
                str, list[dict[Action, int]]
            ] = defaultdict(list)

        def __len__(self) -> int:
            return len(self.records)

        def __iter__(self) -> Iterator[list[Move]]:
            return iter(self.records)

        def append(self, moves: list[Move]) -> None:
            """Append a new round of moves to the history."""
            if not moves:
                raise ValueError("Each round must have at least one move")

            round_idx = len(self.records)
            self.records.append(moves)

            for m in moves:
                p = m.player_name
                a = m.action
                self.player_round_indices[p].append(round_idx)

                player_history = self.player_cumulative_actions[p]
                if player_history:
                    new_counts = player_history[-1].copy()
                else:
                    new_counts = Counter()
                new_counts[a] += 1
                player_history.append(new_counts)

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
            """

            if lookback_rounds < 0 or lookup_depth <= 0:
                raise ValueError(
                    "lookback_rounds must be >= 0 and lookup_depth > 0"
                )

            indices = self.player_round_indices.get(player_name)
            if not indices:
                return []

            m = len(indices)
            if lookback_rounds >= m:
                return []

            end_index = m - lookback_rounds
            start_index = max(0, end_index - lookup_depth)

            selected_indices = indices[start_index:end_index]

            return [self.records[i] for i in selected_indices]

        def get_prior_action_distribution(
            self,
            player_name: str,
            lookback_rounds: int,
        ) -> dict[Action, int] | None:
            """
            Return the action distribution over ALL rounds that occurred
            BEFORE the player's most recent `lookback_rounds` rounds.
            """
            if lookback_rounds < 0:
                raise ValueError(
                    "lookback_rounds must be >= 0 and lookup_depth > 0"
                )
            history = self.player_cumulative_actions.get(player_name)
            if not history:
                return None

            m = len(history)
            target_idx = m - lookback_rounds - 1

            if target_idx < 0:
                return None
            return history[target_idx].copy()

    def __init__(
        self, base_game: Game, num_rounds: int, discount: float
    ) -> None:
        super().__init__(base_game)
        self.num_rounds = num_rounds
        self.discount = discount
        self.history = self.History()

    def _build_payoffs(self, players: Sequence[Agent]) -> PopulationPayoffs:
        return PopulationPayoffs(players=players, discount=self.discount)
