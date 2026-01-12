"""Common infrastructure for tournament mechanisms."""

import copy
import itertools
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Iterator, Sequence

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.games.base import Action, Game, Move
from src.ranking_evaluations.payoffs_base import PayoffsBase
from src.ranking_evaluations.matchup_payoffs import MatchupPayoffs
from src.registry.agent_registry import create_agent


class Mechanism(ABC):
    """Base class for tournament mechanisms that wrap a single game."""

    def __init__(self, base_game: Game):
        self.base_game = base_game

        self.record_file = f"{self.__class__.__name__}_{self.base_game.__class__.__name__}.jsonl"

    def _build_payoffs(self, players: list[Agent]) -> PayoffsBase:
        return MatchupPayoffs(players=players)

    def _create_players_from_cfgs(self, agent_cfgs: list[dict]) -> list[Agent]:
        """Create players with fixed player IDs from agent configurations."""
        players = []
        for cfg in agent_cfgs:
            for player_id in range(1, self.base_game.num_players + 1):
                agent = create_agent(copy.deepcopy(cfg), player_id=player_id)
                players.append(agent)
        return players

    def run_tournament(self, agent_cfgs: list[dict]) -> PayoffsBase:
        """Run the mechanism over the base game across all players."""
        players = self._create_players_from_cfgs(agent_cfgs)
        payoffs = self._build_payoffs(players)

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

                match_moves = self._play_matchup(seat_players)
                payoffs.add_profile(match_moves)

                dt = time.perf_counter() - t0
                if first_duration is None:
                    first_duration = dt
                    est_total = dt * len(combo_iter)
                    print(
                        f"[ETA] ~{est_total/60:.1f} min for "
                        f"{len(combo_iter)} matchups (sequential)."
                    )
                pbar.update(1)
        return payoffs

    @abstractmethod
    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """Play match(es) between the given players."""
        raise NotImplementedError


class RepetitiveMechanism(Mechanism):
    """A mechanism that repeats the game multiple times."""

    class History:
        """History of moves across multiple rounds."""
        def __init__(self, action_cls: type[Action]):
            self.action_cls = action_cls
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

        def append(self, moves: list[Move], round_number: int | None = None) -> None:
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
            return [(self.round_numbers[idx], self.records[idx]) for idx in selected_indices]

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

            result = {action: 0 for action in self.action_cls}
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
        self, base_game: Game, num_rounds: int, discount: float
    ) -> None:
        super().__init__(base_game)
        self.num_rounds = num_rounds
        self.discount = discount
        self.history = self.History(base_game.action_cls)

    def _build_payoffs(self, players: list[Agent]) -> PayoffsBase:
        return MatchupPayoffs(players=players, discount=self.discount)
