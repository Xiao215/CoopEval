"""Mechanism where agents may delegate their action to a mediator design."""

import json
import random
import re
from typing import Callable, Sequence, override

from src.agents.agent_manager import Agent
from src.games.base import Action, Game, Move
from src.logger_manager import LOGGER
from src.mechanisms.base import Mechanism
from src.mechanisms.prompts import (MEDIATION_MECHANISM_PROMPT,
                                    MEDIATOR_APPROVAL_VOTE_PROMPT,
                                    MEDIATOR_DESIGN_PROMPT)
from src.ranking_evaluations.payoffs_base import PayoffsBase
from src.utils.concurrency import run_tasks


class Mediation(Mechanism):
    """Mechanism that lets agents delegate their action to a mediator."""

    def __init__(
        self,
        base_game: Game,
        *,
        tournament_workers: int = 1,
    ) -> None:
        super().__init__(base_game, tournament_workers=tournament_workers)
        self.mediators: dict[str, dict[int, Action]] = {}
        self.mediator_design_prompt = MEDIATOR_DESIGN_PROMPT
        self.mediation_mechanism_prompt = MEDIATION_MECHANISM_PROMPT
        self._cached_agents: list[Agent] | None = None
        self.base_game.add_mediator_action()

    def _design_mediator(
        self,
        designer: Agent,
    ) -> tuple[str, dict[int, Action]]:
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
        _, trace_id, mediator = designer.chat_with_retries(
            base_prompt=base_prompt,
            parse_func=self._parse_mediator,
        )
        return trace_id, mediator

    def _parse_mediator(self, response: str) -> dict[int, Action]:
        """
        Parse the mediator design from the response.
        Expecting a Python dictionary in string format.
        """
        matches = re.findall(r"\{.*?\}", response, re.DOTALL)
        if not matches:
            raise ValueError(
                f"No JSON object found in the response {response!r}"
            )
        json_str = matches[-1]
        json_obj = json.loads(json_str)

        mediator = {}
        for k, v in json_obj.items():
            k = int(k)
            if k < 1 or k > self.base_game.num_players:
                raise ValueError(
                    f"Invalid player number {k} for the pair {k}: {v}, "
                    f"must be between 1 and {self.base_game.num_players}."
                )
            if not 0 <= int(v[1:]) < self.base_game.num_actions:
                raise ValueError(
                    f"Invalid action {v} for the pair {k}: {v}, "
                    f"must be one of {[f'A{a}' for a in range(self.base_game.num_actions)]}."
                )
            mediator[k] = self.base_game.action_class.from_index(int(v[1:]))
        if len(mediator) != self.base_game.num_players:
            raise ValueError(
                "There are missing cases in the mediator design, "
                f"you need to have cases for all number of players "
                f"from 1 to {self.base_game.num_players}."
            )
        return mediator

    def _mediator_description(self, mediator: dict[int, Action]) -> str:
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

    def _collect_vote(
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
            matches = re.findall(r"\{.*?\}", response, re.DOTALL)
            if not matches:
                raise ValueError(
                    f"No JSON object found in response {response!r}"
                )

            json_str = matches[-1]
            json_obj = json.loads(json_str)

            # Convert M1, M2, ... to integer indices
            votes = {}
            for i in range(1, len(players) + 1):
                key = f"M{i}"
                if key not in json_obj:
                    raise ValueError(f"Missing vote for {key}")
                if not isinstance(json_obj[key], bool):
                    raise ValueError(
                        f"Vote for {key} must be boolean, got {json_obj[key]!r}"
                    )
                votes[i] = json_obj[key]

            return votes

        _, trace_id, votes = voter.chat_with_retries(
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

        # Count approvals per mediator
        approval_counts = {i: 0 for i in range(1, len(players) + 1)}
        for _voter, votes in all_votes.items():
            for mediator_idx, approved in votes.items():
                if approved:
                    approval_counts[mediator_idx] += 1

        # Find max approvals
        max_approvals = max(approval_counts.values())

        # Get all mediators with max approvals (for tie-breaking)
        winners = [
            idx
            for idx, count in approval_counts.items()
            if count == max_approvals
        ]

        # Break ties uniformly at random
        winning_idx = random.choice(winners)
        winning_agent = players[winning_idx - 1]  # Convert 1-based to 0-based

        return winning_idx, winning_agent

    @override
    def run_tournament(self, players: list[Agent]) -> PayoffsBase:
        self.mediators.clear()
        # Cache agents so base class reuses them
        self._cached_agents = players

        def design_fn(player: Agent) -> tuple[Agent, str, dict[int, Action]]:
            trace_id, mediator = self._design_mediator(player)
            return player, trace_id, mediator

        results = run_tasks(players, design_fn)

        mediator_design = {}
        for player, trace_id, mediator in results:
            self.mediators[player.name] = mediator
            mediator_design[player.name] = {
                "trace_id": trace_id,
                "mediator": [
                    (num_delegating, action.to_token())
                    for num_delegating, action in mediator.items()
                ],
            }
        LOGGER.log_record(
            record=mediator_design, file_name="mediator_design.json"
        )

        # Now call base class - it will use our cached agents
        result = super().run_tournament(players)

        # Clear cache for next run
        self._cached_agents = None

        return result

    @override
    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """
        Have players vote on mediators, select winner, and play once.

        Returns:
            A list containing a single move sequence (one game result).
        """

        # Step 1: Collect votes from all players
        def collect_vote_fn(
            player: Agent,
        ) -> tuple[Agent, str, dict[int, bool]]:
            trace_id, votes = self._collect_vote(player, players)
            return player, trace_id, votes

        vote_results = run_tasks(players, collect_vote_fn)

        # Step 2: Process voting results
        all_votes = {}  # {voter: {mediator_idx: approval}}
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

        # Step 3: Select winning mediator
        winning_idx, winning_agent = self._select_mediator(players, all_votes)
        winning_mediator = self.mediators[winning_agent.name]

        # Step 4: Play game once under selected mediator
        mediator_description = self._mediator_description(winning_mediator)
        mediator_mechanism = self.mediation_mechanism_prompt.format(
            mediator_description=mediator_description,
            additional_action_id=self.base_game.num_actions,
            designer_player_id=winning_agent.player_id,
        )

        moves = self.base_game.play(
            players=players,
            additional_info=mediator_mechanism,
            action_map=self.mediator_mapping(winning_mediator),
        )

        # Step 6: Log voting and game results
        record = {
            "votes": vote_records,
            "selected_mediator_index": winning_idx,
            "selected_mediator_designer_name": winning_agent.name,
            "moves": moves,
        }
        LOGGER.log_record(record=[record], file_name=self.record_file)

        # Return list with single game result (base class will call payoffs.add_profile)
        return [moves]

    def mediator_mapping(self, mediator: dict[int, Action]) -> Callable:
        """
        Given the original actions and the mediator design, return the final actions
        after applying the mediator's recommendations.
        """

        def apply_mediation(
            players_decision: dict[Agent, tuple[Action, str, str, bool]],
        ) -> dict[Agent, tuple[Action, str, str, bool]]:
            post_mapping_decision = players_decision.copy()

            num_delegating = sum(
                a[0].is_mediator for a in players_decision.values()
            )

            if num_delegating == 0:
                return post_mapping_decision

            recommended_action = mediator[num_delegating]

            for player, (
                action,
                response,
                trace_id,
                _,
            ) in players_decision.items():
                if action.is_mediator:
                    post_mapping_decision[player] = (
                        recommended_action,
                        response,
                        trace_id,
                        True,
                    )
            return post_mapping_decision

        return apply_mediation
