"""Mechanism where agents may delegate their action to a mediator design."""

import json
import re
from typing import Callable, Sequence

from src.agents.agent_manager import Agent
from src.ranking_evaluations.population_payoffs import PopulationPayoffs
from src.games.base import Game, Move
from src.logger_manager import LOGGER
from src.mechanisms.base import Mechanism
from src.mechanisms.prompts import (
    MEDIATION_MECHANISM_PROMPT,
    MEDIATOR_DESIGN_PROMPT,
    MEDIATOR_APPROVAL_VOTE_PROMPT,
)
from src.utils.concurrency import run_tasks


class Mediation(Mechanism):
    """Mechanism that lets agents delegate their action to a mediator."""

    def __init__(
        self,
        base_game: Game,
    ) -> None:
        super().__init__(base_game)
        # keyed by (model_type, player_id)
        self.mediators: dict[tuple[str, int], dict[int, int]] = {}
        self.mediator_design_prompt = MEDIATOR_DESIGN_PROMPT
        self.mediation_mechanism_prompt = MEDIATION_MECHANISM_PROMPT
        self._cached_agents: list[Agent] | None = None

    def _design_mediator(
        self,
        designer: Agent,
    ) -> tuple[str, dict[int, int]]:
        """
        Design the mediator agent by the given LLM agent.

        Returns:
            response (str): The raw response from the designer.
            mediator (dict[int, int]): A dictionary mapping number of delegating players to recommended action.
        """
        game_prompt = self.base_game.get_player_prompt(designer.player_id)
        base_prompt = (
            game_prompt
            + "\n"
            + self.mediator_design_prompt.format(
                num_players=self.base_game.num_players,
            )
        )
        response, mediator = designer.chat_with_retries(
            base_prompt=base_prompt,
            parse_func=self._parse_mediator,
        )
        return response, mediator

    def _parse_mediator(self, response: str) -> dict[int, int]:
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

        try:
            json_obj = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e

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
            mediator[k] = int(v[1:])
        if len(mediator) != self.base_game.num_players:
            raise ValueError(
                "There are missing cases in the mediator design, "
                f"you need to have cases for all number of players "
                f"from 1 to {self.base_game.num_players}."
            )
        return mediator

    def _mediator_description(self, mediator: dict[int, int]) -> str:
        """Format the prompt for the mediator agent."""
        lines = []
        for num_delegating, action in mediator.items():
            lines.append(
                f"\t• If {num_delegating} player(s) delegate to the mediator, "
                f"it will play action A{action}."
            )
        return "\n".join(lines)

    def _all_mediators_description(self, players: Sequence[Agent]) -> str:
        """Format all mediators for the voting prompt."""
        lines = []
        for player in players:
            key = (player.model_type, player.player_id)
            mediator = self.mediators[key]
            lines.append(f"Mediator proposed by Player {player.player_id}:")
            lines.append(self._mediator_description(mediator))
            lines.append("")
        return "\n".join(lines)

    def _collect_vote(
        self,
        voter: Agent,
        players: Sequence[Agent]
    ) -> tuple[str, dict[int, bool]]:
        """
        Ask an agent to vote on which mediators they approve.

        Returns:
            response (str): The raw response from the voter
            votes (dict[int, bool]): Mapping from mediator index to approval (True/False)
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
                raise ValueError(f"No JSON object found in response {response!r}")

            json_str = matches[-1]
            try:
                json_obj = json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e.msg}") from e

            # Convert M1, M2, ... to integer indices
            votes = {}
            for i in range(1, len(players) + 1):
                key = f"M{i}"
                if key not in json_obj:
                    raise ValueError(f"Missing vote for {key}")
                if not isinstance(json_obj[key], bool):
                    raise ValueError(f"Vote for {key} must be boolean, got {json_obj[key]!r}")
                votes[i] = json_obj[key]

            return votes

        response, votes = voter.chat_with_retries(
            base_prompt=vote_prompt,
            parse_func=parse_votes,
        )
        return response, votes

    def _select_mediator(
        self,
        players: Sequence[Agent],
        all_votes: dict[int, dict[int, bool]]
    ) -> tuple[int, Agent]:
        """
        Select winning mediator based on approval votes.

        Args:
            players: Sequence of players in the matchup
            all_votes: {voter_uid: {mediator_index: approval}}

        Returns:
            (winning_index, winning_agent): Index (1-based) and Agent who designed winner
        """
        import random

        # Count approvals per mediator
        approval_counts = {i: 0 for i in range(1, len(players) + 1)}
        for _voter_uid, votes in all_votes.items():
            for mediator_idx, approved in votes.items():
                if approved:
                    approval_counts[mediator_idx] += 1

        # Find max approvals
        max_approvals = max(approval_counts.values())

        # Get all mediators with max approvals (for tie-breaking)
        winners = [idx for idx, count in approval_counts.items() if count == max_approvals]

        # Break ties uniformly at random
        winning_idx = random.choice(winners)
        winning_agent = players[winning_idx - 1]  # Convert 1-based to 0-based

        return winning_idx, winning_agent

    def _create_players_from_cfgs(self, agent_cfgs: list[dict]) -> list[Agent]:
        """Return cached agents if available, otherwise create new ones."""
        if self._cached_agents is not None:
            return self._cached_agents
        return super()._create_players_from_cfgs(agent_cfgs)

    def run_tournament(self, agent_cfgs: list[dict]) -> PopulationPayoffs:
        # Create num_players agents per config using base class method
        # This ensures each agent gets unique UID and designs their own mediator
        agents = super()._create_players_from_cfgs(agent_cfgs)

        # Cache agents so base class reuses them
        self._cached_agents = agents

        def design_fn(agent: Agent) -> tuple[Agent, str, dict[int, int]]:
            response, mediator = self._design_mediator(agent)
            return agent, response, mediator

        results = run_tasks(agents, design_fn)

        self.mediators.clear()
        mediator_design = {}
        for agent, response, mediator in results:
            key = (agent.model_type, agent.player_id)
            self.mediators[key] = mediator
            mediator_design[agent.name] = {
                "response": response,
                "mediator": mediator,
            }
        LOGGER.log_record(
            record=mediator_design, file_name="mediator_design.json"
        )

        # Now call base class - it will use our cached agents
        result = super().run_tournament(agent_cfgs)

        # Clear cache for next run
        self._cached_agents = None

        return result

    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """
        Have players vote on mediators, select winner, and play once.

        Returns:
            A list containing a single move sequence (one game result).
        """
        # Step 1: Collect votes from all players
        def collect_vote_fn(player: Agent) -> tuple[Agent, str, dict[int, bool]]:
            response, votes = self._collect_vote(player, players)
            return player, response, votes

        vote_results = run_tasks(players, collect_vote_fn)

        # Step 2: Process voting results
        all_votes = {}  # {voter_uid: {mediator_idx: approval}}
        vote_records = []
        for player, response, votes in vote_results:
            all_votes[player.uid] = votes
            vote_records.append({
                "voter_uid": player.uid,
                "voter_name": player.name,
                "votes": votes,
                "response": response,
            })

        # Step 3: Select winning mediator
        winning_idx, winning_agent = self._select_mediator(players, all_votes)
        key = (winning_agent.model_type, winning_agent.player_id)
        winning_mediator = self.mediators[key]

        # Step 4: Play game once under selected mediator
        mediator_description = self._mediator_description(winning_mediator)
        mediator_mechanism = self.mediation_mechanism_prompt.format(
            mediator_description=mediator_description,
            additional_action_id=self.base_game.num_actions,
        )

        moves = self.base_game.play(
            players=players,
            additional_info=mediator_mechanism,
            action_map=self.mediator_mapping(winning_mediator),
        )

        # Step 5: Serialize game results
        serialized_moves = [
            {
                "uid": move.uid,
                "player_name": move.player_name,
                "action": (
                    move.action.value
                    if hasattr(move.action, "value")
                    else str(move.action)
                ),
                # TODO: add the mix strategy
                # TODO: add whether they delegated, maybe change the action
                "points": move.points,
                "response": move.response,
            }
            for move in moves
        ]

        # Step 6: Log voting and game results
        record = {
            "votes": vote_records,
            "selected_mediator_index": winning_idx,
            "selected_mediator_designer_uid": winning_agent.uid,
            "selected_mediator_designer_name": winning_agent.name,
            "moves": serialized_moves,
        }
        LOGGER.log_record(record=[record], file_name=self.record_file)

        # Return list with single game result (base class will call payoffs.add_profile)
        return [moves]

    def mediator_mapping(self, mediator: dict[int, int]) -> Callable:
        """
        Given the original actions and the mediator design, return the final actions
        after applying the mediator's recommendations.
        """

        def apply_mediation(
            player_action_map: dict[str, int],
        ) -> dict[str, int]:
            actions: dict[str, int] = {}
            num_delegating = sum(
                a == self.base_game.num_actions
                for a in player_action_map.values()
            )
            if num_delegating == 0:
                return player_action_map
            recommended_action = mediator[num_delegating]
            for player_id, action_idx in player_action_map.items():
                if action_idx == self.base_game.num_actions:
                    actions[player_id] = recommended_action
                else:
                    actions[player_id] = action_idx
            return actions

        return apply_mediation
