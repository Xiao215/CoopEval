"""Mechanism where agents may delegate their action to a mediator design."""

import json
import re
from typing import Callable, Sequence

from src.agents.agent_manager import Agent
from src.ranking_evaluations.population_payoffs import PopulationPayoffs
from src.games.base import Game
from src.logger_manager import LOGGER
from src.mechanisms.base import Mechanism
from src.mechanisms.prompts import (
    MEDIATION_MECHANISM_PROMPT,
    MEDIATOR_DESIGN_PROMPT,
)
from src.registry.agent_registry import create_agent
from src.utils.concurrency import run_tasks


class Mediation(Mechanism):
    """Mechanism that lets agents delegate their action to a mediator."""

    def __init__(
        self,
        base_game: Game,
    ) -> None:
        super().__init__(base_game)
        self.mediators: dict[int, dict[int, int]] = {}
        self.mediator_design_prompt = MEDIATOR_DESIGN_PROMPT
        self.mediation_mechanism_prompt = MEDIATION_MECHANISM_PROMPT

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
        base_prompt = (
            self.base_game.prompt
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

    def run_tournament(self, agent_cfgs: Sequence[dict]) -> PopulationPayoffs:
        agents = [create_agent(cfg) for cfg in agent_cfgs]

        def design_fn(agent: Agent) -> tuple[Agent, str, dict[int, int]]:
            response, mediator = self._design_mediator(agent)
            return agent, response, mediator

        results = run_tasks(agents, design_fn)

        self.mediators.clear()
        mediator_design = {}
        for agent, response, mediator in results:
            self.mediators[agent.model_type] = mediator
            mediator_design[agent.model_type] = {
                "response": response,
                "mediator": mediator,
            }
        LOGGER.log_record(
            record=mediator_design, file_name="mediator_design.json"
        )
        return super().run_tournament(agent_cfgs)

    def _play_matchup(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> None:
        history = []

        def play_for_mediator(player: Agent) -> tuple[str, list[dict]]:
            mediator = self.mediators[player.model_type]
            mediator_description = self._mediator_description(mediator)
            mediator_mechanism = self.mediation_mechanism_prompt.format(
                mediator_description=mediator_description,
                additional_action_id=self.base_game.num_actions,
            )
            moves = self.base_game.play(
                players=players,
                additional_info=mediator_mechanism,
                action_map=self.mediator_mapping(mediator),
            )
            payoffs.add_profile([moves])
            serialized = [
                {
                    "uid": move.uid,
                    "player_name": move.player_name,
                    "action": move.action.value
                    if hasattr(move.action, "value")
                    else str(move.action),
                    "points": move.points,
                    "response": move.response,
                }
                for move in moves
            ]
            return player.model_type, serialized

        results = run_tasks(
            players,
            play_for_mediator,
        )
        for mediator_model_type, move_dicts in results:
            history.append(
                {"mediator": mediator_model_type, "moves": move_dicts}
            )
        LOGGER.log_record(record=history, file_name=self.record_file)

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
