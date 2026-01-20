"""Common infrastructure for tournament mechanisms."""

import itertools
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from datetime import datetime
from typing import Iterator, Sequence, override

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.games.base import Action, Game, Move
from src.ranking_evaluations.matchup_payoffs import MatchupPayoffs
from src.ranking_evaluations.payoffs_base import PayoffsBase


class Mechanism(ABC):
    """Base class for tournament mechanisms that wrap a single game."""

    def __init__(self, base_game: Game, *, tournament_workers: int = 1):
        self.base_game = base_game
        self.record_file = "records.jsonl"
        self.tournament_workers = tournament_workers

    def _build_payoffs(self) -> PayoffsBase:
        return MatchupPayoffs()

    def run_tournament(self, players: list[Agent]) -> PayoffsBase:
        """Run the mechanism over the base game across all players."""
        payoffs = self._build_payoffs()
        k = self.base_game.num_players

        players_by_id = [
            [p for p in players if p.player_id == player_id]
            for player_id in range(1, k + 1)
        ]

        combo_iter = list(itertools.product(*players_by_id))

        matchup_labels = [
            " vs ".join(player.name for player in matchup)
            for matchup in combo_iter
        ]

        # Run matchups with optional parallelization
        results = self._run_matchups(combo_iter, matchup_labels)

        # Add all results to payoffs
        for match_moves in results:
            payoffs.add_profile(match_moves)

        return payoffs

    def _run_matchups(
        self,
        combo_iter: list[tuple[Agent, ...]],
        matchup_labels: list[str],
    ) -> list[list[list[Move]]]:
        """Run matchups sequentially or in parallel based on tournament_workers.

        Args:
            combo_iter: List of player tuples for each matchup
            matchup_labels: Human-readable labels for progress display

        Returns:
            List of match results, where each result is a list of rounds
        """
        is_parallel = self.tournament_workers > 1

        if is_parallel:
            from src.utils.concurrency import run_tasks
            print(
                f"[Parallel] Running {len(combo_iter)} matchups with "
                f"{self.tournament_workers} workers"
            )

        results = []
        first_duration = None
        desc = "Tournaments (parallel)" if is_parallel else "Tournaments"

        with tqdm(
            total=len(combo_iter),
            desc=desc,
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            if is_parallel:
                # Parallel execution: process in batches
                batch_size = max(1, self.tournament_workers * 2)

                for i in range(0, len(combo_iter), batch_size):
                    batch = combo_iter[i : i + batch_size]
                    batch_results = run_tasks(
                        batch, self._play_matchup, max_workers=self.tournament_workers
                    )
                    results.extend(batch_results)
                    current_time = datetime.now().strftime('%H:%M:%S')
                    pbar.set_postfix_str(f"Time: {current_time}", refresh=True)
                    pbar.update(len(batch))
            else:
                # Sequential execution: process one at a time with detailed progress
                for seat_players, matchup_label in zip(
                    combo_iter, matchup_labels, strict=True
                ):
                    current_time = datetime.now().strftime('%H:%M:%S')
                    pbar.set_postfix_str(f"{matchup_label} | Time: {current_time}", refresh=False)
                    t0 = time.perf_counter()

                    match_moves = self._play_matchup(seat_players)
                    results.append(match_moves)

                    dt = time.perf_counter() - t0
                    if first_duration is None:
                        first_duration = dt
                        est_total = dt * len(combo_iter)
                        print(
                            f"[ETA] ~{est_total/60:.1f} min for "
                            f"{len(combo_iter)} matchups (sequential)."
                        )
                    pbar.update(1)

        return results

    @abstractmethod
    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """Play match(es) between the given players."""
        raise NotImplementedError


class RepetitiveMechanism(Mechanism):
    """A mechanism that repeats the game multiple times."""

    class History:
        """History of moves across multiple rounds."""

        def __init__(self, action_class: type[Action]) -> None:
            self.action_class = action_class
            # List of all rounds information.
            # Note, the indices is arbitrary and only used for lookup.
            self.records: list[list[Move]] = []

            # Maps each record index to its tournament round number
            self.round_numbers: list[int] = []

            # Maps player -> List of global round indices they participated in
            self.player_round_indices: dict[Agent, list[int]] = defaultdict(
                list
            )

            # Maps player -> List of cumulative distributions at each step
            # Index i corresponds to the state after the player's i-th game
            self.player_cumulative_actions: dict[
                Agent, list[dict[Action, int]]
            ] = defaultdict(list)

        def __len__(self) -> int:
            return len(self.records)

        def __iter__(self) -> Iterator[list[Move]]:
            return iter(self.records)

        def append(
            self, moves: list[Move], round_number: int | None = None
        ) -> None:
            """Append a new round of moves to the history.

            Args:
                moves: List of moves from this match
                round_number: Tournament round number (if None, uses sequential numbering)
            """
            if not moves:
                raise ValueError("Each round must have at least one move")

            record_idx = len(self.records)
            self.records.append(moves)

            # Track tournament round number
            if round_number is None:
                round_number = record_idx + 1  # Default: 1-indexed sequential
            self.round_numbers.append(round_number)

            for m in moves:
                p = m.player
                a = m.action
                self.player_round_indices[p].append(record_idx)

                player_history = self.player_cumulative_actions[p]
                if player_history:
                    new_counts = player_history[-1].copy()
                else:
                    new_counts = Counter()
                new_counts[a] += 1
                player_history.append(new_counts)

        def get_prior_rounds(
            self,
            player: Agent,
            lookback_rounds: int,
            lookup_depth: int,
        ) -> list[tuple[int, list[Move]]]:
            """
            Return the last `lookup_depth` rounds from the player's
            history EXCLUDING the most recent `lookback_rounds` rounds.

            Returns:
                List of tuples (round_index, moves) where round_index is the
                global round number (1-indexed).
            """

            if lookback_rounds < 0 or lookup_depth <= 0:
                raise ValueError(
                    "lookback_rounds must be >= 0 and lookup_depth > 0"
                )

            indices = self.player_round_indices.get(player, [])
            if not indices:
                return []

            m = len(indices)
            if lookback_rounds >= m:
                return []

            end_index = m - lookback_rounds
            start_index = max(0, end_index - lookup_depth)

            selected_indices = indices[start_index:end_index]

            # Return tuples of (tournament_round_number, moves)
            return [
                (self.round_numbers[idx], self.records[idx])
                for idx in selected_indices
            ]

        def get_prior_action_distribution(
            self,
            player: Agent,
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
            history = self.player_cumulative_actions.get(player, [])
            if not history:
                return None

            m = len(history)
            target_idx = m - lookback_rounds - 1

            if target_idx < 0:
                return None

            result = {action: 0 for action in self.action_class.game_actions()}
            result.update(history[target_idx])
            return result

        def get_rounds_played_count(self, player: Agent) -> int:
            """
            Return the total number of rounds a specific player has participated in.
            """
            return len(self.player_round_indices[player])

        def clear(self) -> None:
            """Clear the history records."""
            self.records.clear()
            self.round_numbers.clear()
            self.player_round_indices.clear()
            self.player_cumulative_actions.clear()

    def __init__(
        self,
        base_game: Game,
        num_rounds: int,
        discount: float,
        *,
        tournament_workers: int = 1,
    ) -> None:
        super().__init__(base_game, tournament_workers=tournament_workers)
        self.num_rounds = num_rounds
        self.discount = discount
        self.history = self.History(self.base_game.action_class)

    @override
    def _build_payoffs(self) -> PayoffsBase:
        return MatchupPayoffs(discount=self.discount)
