"""Common infrastructure for tournament mechanisms."""

import asyncio
import itertools
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Iterator, Sequence, override

from coopeval.agents.agent_manager import Agent
from coopeval.games.base import Action, Game, Move
from coopeval.ranking_evaluations.matchup_payoffs import MatchupPayoffs
from coopeval.ranking_evaluations.payoffs_base import PayoffsBase


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
        return asyncio.run(self._run_tournament_async(players))

    async def _run_tournament_async(self, players: list[Agent]) -> PayoffsBase:
        """Async implementation of the tournament logic."""
        payoffs = self._build_payoffs()
        k = self.base_game.num_players

        players_by_id = [
            [p for p in players if p.player_id == player_id]
            for player_id in range(1, k + 1)
        ]

        combo_iter = list(itertools.product(*players_by_id))

        results = await self._run_matchups(combo_iter)

        for match_moves in results:
            payoffs.add_profile(match_moves)

        return payoffs

    async def _run_matchups(
        self,
        combo_iter: list[tuple[Agent, ...]],
    ) -> list[list[list[Move]]]:
        """Run matchups using native asyncio gathering.

        Args:
            combo_iter: List of player tuples for each matchup

        Returns:
            List of match results, where each result is a list of rounds
        """
        # To respect tournament_workers limit, we can use an asyncio.Semaphore
        semaphore = asyncio.Semaphore(self.tournament_workers)

        async def bounded_play(players: tuple[Agent, ...]) -> list[list[Move]]:
            async with semaphore:
                return await self._play_matchup(players)

        tasks = [bounded_play(players) for players in combo_iter]
        return await asyncio.gather(*tasks)

    @abstractmethod
    async def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """Play match(es) between the given players."""
        raise NotImplementedError


class RepetitiveMechanism(Mechanism):
    """A mechanism that repeats the game multiple times."""

    class History:
        """History of moves across multiple rounds."""

        def __init__(self, action_class: type[Action]) -> None:
            self.action_class = action_class
            # Raw move histories stored in insertion order; indices act as stable IDs.
            self.records: list[list[Move]] = []

            # Some mechanisms reuse external round numbers, so keep both logical and sequential IDs.
            self.round_numbers: list[int] = []

            # Track which record indices each player participated in for quick lookups.
            self.player_round_indices: dict[Agent, list[int]] = defaultdict(
                list
            )

            # For each player, capture the cumulative action histogram after every appearance.
            # The i-th entry reflects the distribution immediately after their i-th game.
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

            if round_number is None:
                round_number = record_idx + 1
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
