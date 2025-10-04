"""Mechanisms that expose behavioural reputation across repeated rounds."""

import itertools
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Sequence

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.games.base import Game, Move
from src.games.prisoners_dilemma import (PrisonersDilemma,
                                         PrisonersDilemmaAction)
from src.games.public_goods import PublicGoods, PublicGoodsAction
from src.logger_manager import LOGGER
from src.mechanisms.base import RepetitiveMechanism
from src.registry.agent_registry import create_agent

random.seed(42)


class ReputationStat:
    def __init__(self):
        # reputation[metric] = (positive_count, total_count)
        self.scores: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))

    def record(self, metric: str, success: bool) -> None:
        """Record one trial for `metric`."""
        pos, tot = self.scores[metric]
        tot += 1
        pos += int(success)
        self.scores[metric] = (pos, tot)

    def rate(self, metric: str) -> float | None:
        """Return success rate or None if no trials."""
        pos, tot = self.scores.get(metric, (0, 0))
        return (pos / tot) if tot else None

    def stat(self, metric: str) -> tuple[int, int]:
        """Return the raw counts for the metric."""
        return self.scores[metric]

    def all_rates(self) -> dict[str, float | None]:
        """Return all rates under the agent."""
        return {m: (p / t if t else None) for m, (p, t) in self.scores.items()}


class Reputation(RepetitiveMechanism, ABC):
    """
    Reputation mechanism that makes each players' reputation visible to all players.
    """

    def __init__(self, base_game: Game, num_rounds: int, discount: float):
        super().__init__(base_game, num_rounds, discount)
        self.reputation: dict[str, ReputationStat] = defaultdict(ReputationStat)
        self.matchup_workers = 1

    @abstractmethod
    def _format_reputation(self, agents: Sequence[Agent]) -> str:
        """Format the reputation information into a string."""
        raise NotImplementedError(
            "`_format_reputation` should be implemented in subclasses."
        )

    def run_tournament(self, agent_cfgs: Sequence[dict]) -> PopulationPayoffs:
        """Run a multi-round tournament while tracking reputation updates."""
        players = [
            create_agent(cfg)
            for cfg in agent_cfgs
            for _ in range(self.base_game.num_players)
        ]
        payoffs = self._build_payoffs(players)

        self._play_matchups(players, payoffs)

        return payoffs

    def _play_matchups(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> None:
        """Play all matchups while updating reputation after each round."""
        num_players = self.base_game.num_players
        all_matchups = list(itertools.combinations(players, r=num_players))
        random.shuffle(all_matchups)

        k = self.base_game.num_players
        matchups = [tuple(lineup) for lineup in itertools.combinations(players, k)]
        matchup_histories: defaultdict[tuple[int, ...], list[list[Move]]] = defaultdict(
            list
        )

        round_iter = tqdm(
            range(1, self.num_rounds + 1),
            desc=f"Running Reputation Mechanism for {self.base_game.__class__.__name__}",
        )

        for round_idx in round_iter:
            # Shuffle matchups before each round, even though order does not impact outcomes
            matchups_this_round = list(matchups)
            random.shuffle(matchups_this_round)

            round_matches: list[dict[str, object]] = []

            for matchup in matchups_this_round:
                reputation_information = self._format_reputation(matchup)
                moves = self.base_game.play(
                    additional_info=reputation_information,
                    players=matchup,
                )

                round_matches.append(
                    {
                        "players": [move.player_name for move in moves],
                        "moves": [move.to_dict() for move in moves],
                    }
                )

                for move in moves:
                    self._update_reputation(move.player_name, move.action.value)

            LOGGER.log_record(
                record={"round": round_idx, "matchups": round_matches},
                file_name=self.record_file,
            )

        for moves_over_rounds in matchup_histories.values():
            payoffs.add_profile(moves_over_rounds)

    @abstractmethod
    def _update_reputation(self, name: str, action: str) -> None:
        """Update the reputation of a player based on their action."""
        raise NotImplementedError


class ReputationPrisonersDilemma(Reputation):
    """
    Reputation mechanism for the Prisoner's Dilemma game.
    This mechanism tracks the cooperation rates of players.
    """

    def __init__(self, base_game: PrisonersDilemma, num_rounds: int, discount: float):
        super().__init__(base_game, num_rounds, discount)
        if not isinstance(self.base_game, PrisonersDilemma):
            raise TypeError(
                f"ReputationPrisonersDilemma can only be used with Prisoner's Dilemma games, "
                f"but got {self.base_game.__class__.__name__}."
            )

    def _format_reputation(self, agents: Sequence[Agent]) -> str:
        """Format the reputation information of the given agents into a string."""
        lines = []
        coop_tok = PrisonersDilemmaAction.COOPERATE.to_token()

        for agent in agents:
            name = agent.name
            agent_reputation = self.reputation[agent.name]

            coop_rate = agent_reputation.rate("cooperation_rate")

            # Initial reputation information at game start
            if coop_rate is None:
                lines.append(f"{name}: No reputation data available.")
                continue
            else:
                coop_count, total_count = agent_reputation.stat("cooperation_rate")
                coop_pct = coop_rate
                lines.append(
                    f"{name} played {coop_tok} in {coop_count}/{total_count} rounds ({coop_pct:.2%})"
                )
        lines = [f"\n\t{line}" for line in lines]
        return (
            "\nReputation:"
            + "".join(lines)
            + "\n\tNote: Your chosen action will affect your reputation score."
        )

    def _update_reputation(self, name: str, action: str) -> None:
        self.reputation[name].record(
            "cooperation_rate",
            action == PrisonersDilemmaAction.COOPERATE.value,
        )


class ReputationPublicGoods(Reputation):
    """
    Reputation mechanism for the Public Goods game.
    This mechanism tracks the contribution rates of players.
    """

    def __init__(self, base_game: Game, num_rounds: int, discount: float):
        super().__init__(base_game, num_rounds=num_rounds, discount=discount)
        if not isinstance(self.base_game, PublicGoods):
            raise TypeError(
                f"ReputationPublicGoods can only be used with PublicGoodsGame, "
                f"but got {self.base_game.__class__.__name__}"
            )

    def _format_reputation(self, agents: Sequence[Agent]) -> str:
        """Format the reputation information of the given agents into a string."""
        lines = []
        coop_tok = PrisonersDilemmaAction.COOPERATE.to_token()

        for agent in agents:
            name = agent.name
            agent_reputation = self.reputation[agent.name]

            coop_rate = agent_reputation.rate("contribution_rate")

            # Initial reputation information at game start
            if coop_rate is None:
                lines.append(f"{name}: No reputation data available.")
                continue
            else:
                coop_count, total_count = agent_reputation.stat("contribution_rate")
                coop_pct = coop_rate
                lines.append(
                    f"They chose {coop_tok} in {coop_count}/{total_count} rounds ({coop_pct:.2%})"
                )
        lines = [f"\n\t\t{line}" for line in lines]
        return (
            "\n\tReputation:"
            + "".join(lines)
            + "\n\t\tNote: Your chosen action will affect your reputation score."
        )

    def _update_reputation(self, name: str, action: str) -> None:
        self.reputation[name].record(
            "contribution_rate",
            action == PublicGoodsAction.CONTRIBUTE.value,
        )
