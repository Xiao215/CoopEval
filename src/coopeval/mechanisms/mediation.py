"""Mechanism where agents may delegate their action to a mediator design."""

import asyncio
import random
from typing import Callable, Sequence, override

from coopeval.agents.agent_manager import Agent
from coopeval.games.base import Action, Game, Move
from coopeval.logger_manager import LOGGER
from coopeval.mechanisms.base import Mechanism
from coopeval.mechanisms.prompts import (
    MEDIATION_MECHANISM_PROMPT,
    MEDIATOR_APPROVAL_VOTE_PROMPT,
    MEDIATOR_DESIGN_PROMPT,
)
from coopeval.mechanisms.json_parsing import parse_bool_votes
from coopeval.ranking_evaluations.payoffs_base import PayoffsBase
from coopeval.utils.json_io import extract_json_object_from_end

Mediator = dict[int, Action]


class Mediation(Mechanism):
    """Mechanism that lets agents delegate their action to a mediator."""

    def __init__(
        self,
        base_game: Game,
        *,
        tournament_workers: int = 1,
    ) -> None:
        super().__init__(base_game, tournament_workers=tournament_workers)
        self.mediators: dict[str, Mediator] = {}
        self.mediator_design_prompt = MEDIATOR_DESIGN_PROMPT
        self.mediation_mechanism_prompt = MEDIATION_MECHANISM_PROMPT
        self._cached_agents: list[Agent] | None = None
        self.base_game.add_mediator_action()

    async def _design_mediator(
        self,
        designer: Agent,
    ) -> tuple[str, Mediator]:
        """
        Design the mediator agent by the given LLM player.
        """
        game_prompt = self.base_game.get_player_prompt(designer.player_id)
        base_prompt = (
            game_prompt
            + "\n"
            + self.mediator_design_prompt.format(
                num_players=self.base_game.num_players,
            )
        )

        _, trace_id, mediator = await designer.chat_with_retries(
            base_prompt=base_prompt,
            parse_func=self._parse_mediator,
        )
        return trace_id, mediator

    def _parse_mediator(self, response: str) -> Mediator:
        """
        Parse the mediator design from the response.
        Expecting a Python dictionary in string format.
        """
        json_obj = extract_json_object_from_end(response)
        valid_action_tokens = [
            f"A{a}" for a in range(self.base_game.num_base_actions)
        ]

        mediator = {}
        for k, v in json_obj.items():
            try:
                num_delegating = int(k)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid player count key {k!r}; expected an integer string."
                ) from exc

            if (
                num_delegating < 1
                or num_delegating > self.base_game.num_players
            ):
                raise ValueError(
                    f"Invalid player number {num_delegating} for the pair {k}: {v}, "
                    f"must be between 1 and {self.base_game.num_players}."
                )
            if not isinstance(v, str) or not v.startswith("A"):
                raise ValueError(
                    f"Invalid action {v!r} for the pair {k}: {v}, "
                    f"must be one of {valid_action_tokens}."
                )
            try:
                action_idx = int(v[1:])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid action {v!r} for the pair {k}: {v}, "
                    f"must be one of {valid_action_tokens}."
                ) from exc
            if not 0 <= action_idx < self.base_game.num_base_actions:
                raise ValueError(
                    f"Invalid action {v} for the pair {k}: {v}, "
                    f"must be one of {valid_action_tokens}."
                )
            mediator[num_delegating] = self.base_game.action_class.from_index(
                action_idx
            )
        if len(mediator) != self.base_game.num_players:
            raise ValueError(
                "There are missing cases in the mediator design, "
                f"you need to have cases for all number of players "
                f"from 1 to {self.base_game.num_players}."
            )
        return mediator

    def _mediator_description(self, mediator: Mediator) -> str:
        """Format the prompt for the mediator agent."""
        lines = []
        for num_delegating, action in mediator.items():
            lines.append(
                f"\t• If {num_delegating} player(s) delegate to the mediator, "
                f"it will play action {action.to_token()}."
            )
        return "\n".join(lines)

    def _all_mediators_description(self, players: Sequence[Agent]) -> str:
        """Format all mediators for the voting prompt."""
        lines = []
        for player in players:
            mediator = self.mediators[player.name]
            lines.append(f"Mediator proposed by Player {player.player_id}:")
            lines.append(self._mediator_description(mediator))
            lines.append("")
        return "\n".join(lines)

    async def _collect_vote(
        self, voter: Agent, players: Sequence[Agent]
    ) -> tuple[str, dict[int, bool]]:
        """
        Ask an agent to vote on which mediators they approve.
        """
        game_prompt = self.base_game.get_player_prompt(voter.player_id)
        all_mediators = self._all_mediators_description(players)
        vote_prompt = (
            game_prompt
            + "\n"
            + MEDIATOR_APPROVAL_VOTE_PROMPT.format(
                all_mediators_description=all_mediators
            )
        )

        def parse_votes(response: str) -> dict[int, bool]:
            return parse_bool_votes(
                response,
                prefix="M",
                count=len(players),
            )

        _, trace_id, votes = await voter.chat_with_retries(
            base_prompt=vote_prompt,
            parse_func=parse_votes,
        )
        return trace_id, votes

    def _select_mediator(
        self, players: Sequence[Agent], all_votes: dict[Agent, dict[int, bool]]
    ) -> tuple[int, Agent]:
        """
        Select winning mediator based on approval votes.

        Args:
            players: Sequence of players in the matchup
            all_votes: {voter_uid: {mediator_index: approval}}

        Returns:
            (winning_index, winning_agent): Index (1-based) and Agent who designed winner
        """

        approval_counts = {i: 0 for i in range(1, len(players) + 1)}
        for _voter, votes in all_votes.items():
            for mediator_idx, approved in votes.items():
                if approved:
                    approval_counts[mediator_idx] += 1

        max_approvals = max(approval_counts.values())

        winners = [
            idx
            for idx, count in approval_counts.items()
            if count == max_approvals
        ]

        winning_idx = random.choice(winners)
        winning_agent = players[winning_idx - 1]

        return winning_idx, winning_agent

    @override
    async def _run_tournament_async(self, players: list[Agent]) -> PayoffsBase:
        self.mediators.clear()
        self._cached_agents = players

        async def design_fn(player: Agent) -> tuple[Agent, str, Mediator]:
            trace_id, mediator = await self._design_mediator(player)
            return player, trace_id, mediator

        tasks = [design_fn(player) for player in players]
        results = await asyncio.gather(*tasks)

        mediator_design = {}
        for player, trace_id, mediator in results:
            self.mediators[player.name] = mediator
            mediator_design[player.name] = {
                "trace_id": trace_id,
                "mediator": [
                    (num_delegating, str(action))
                    for num_delegating, action in mediator.items()
                ],
            }
        LOGGER.log_record(
            record=mediator_design, file_name="mediator_design.json"
        )

        result = await super()._run_tournament_async(players)

        self._cached_agents = None

        return result

    @override
    async def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """
        Have players vote on mediators, select winner, and play once.

        Returns:
            A list containing a single move sequence (one game result).
        """

        async def collect_vote_fn(
            player: Agent,
        ) -> tuple[Agent, str, dict[int, bool]]:
            trace_id, votes = await self._collect_vote(player, players)
            return player, trace_id, votes

        tasks = [collect_vote_fn(player) for player in players]
        vote_results = await asyncio.gather(*tasks)

        all_votes = {}
        vote_records = []
        for player, trace_id, votes in vote_results:
            all_votes[player] = votes
            vote_records.append(
                {
                    "voter_name": player.name,
                    "votes": votes,
                    "trace_id": trace_id,
                }
            )

        winning_idx, winning_agent = self._select_mediator(players, all_votes)
        winning_mediator = self.mediators[winning_agent.name]

        mediator_description = self._mediator_description(winning_mediator)
        mediator_mechanism = self.mediation_mechanism_prompt.format(
            mediator_description=mediator_description,
            additional_action_id=self.base_game.num_base_actions,
            designer_player_id=winning_agent.player_id,
        )

        moves = await self.base_game.play(
            players=players,
            additional_info=mediator_mechanism,
            action_map=self.mediator_mapping(winning_mediator),
        )

        record = {
            "votes": vote_records,
            "selected_mediator_index": winning_idx,
            "selected_mediator_designer_name": winning_agent.name,
            "moves": moves,
        }
        LOGGER.log_record(record=[record], file_name=self.record_file)

        return [moves]

    def mediator_mapping(self, mediator: dict[int, Action]) -> Callable:
        """
        Given the original actions and the mediator design, return the final actions
        after applying the mediator's recommendations.
        """

        def apply_mediation(
            player_actions: dict[Agent, Action],
        ) -> dict[Agent, Action]:
            num_delegating = sum(
                action.is_mediator for action in player_actions.values()
            )

            if num_delegating == 0:
                return dict(player_actions)

            recommended_action = mediator[num_delegating]

            return {
                player: (recommended_action if action.is_mediator else action)
                for player, action in player_actions.items()
            }

        return apply_mediation
