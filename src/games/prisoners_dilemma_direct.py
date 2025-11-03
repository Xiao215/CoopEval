from __future__ import annotations

import json
import re
import textwrap
from typing import Mapping, Sequence

from src.agents.agent_manager import Agent
from src.games.base import Game, Move
from src.games.prisoners_dilemma import PrisonersDilemma, PrisonersDilemmaAction
from src.logger_manager import LOGGER

ACTION_JSON_REGEX = re.compile(r"\{.*?\}", re.DOTALL)


class PrisonersDilemmaDirect(Game):
    """
    Prisoner's Dilemma variant where agents commit to a single action directly.
    """

    def __init__(self, payoff_matrix: Mapping[str, Sequence[float]]) -> None:
        self.payoff_matrix = PrisonersDilemma._parse_payoff_matrix(payoff_matrix)

        actions_block = "\n".join(
            [
                f"- {act.name.title()} ({act.value}) via token {act.to_token()}"
                for act in PrisonersDilemmaAction
            ]
        )

        prompt = textwrap.dedent(
            f"""
        You are playing a decision-making game against another player.
        Your objective is to maximise your points received in the game described below.

        Available actions:
        {actions_block}

        Game rules:
        1. You and the opponent simultaneously commit to one action each round.
        2. After both decisions are locked in, those actions are revealed.
        3. Both players receive the points specified in the payoff description below.

        Payoff description:
        {self._payoff_description()}
        """
        )

        super().__init__(
            prompt=prompt,
            num_players=2,
            num_actions=len(PrisonersDilemmaAction),
        )

        self.default_output_instruction = textwrap.dedent(
            """
        Instruction:
        - Choose exactly one action token for this round.
        - Respond with a valid JSON object: {"action": "<token>"}.
        - You may supply the action as A0/A1 or as C/D (case-insensitive).
        - No additional keys or text may appear after the JSON object.
        """
        )

    def _payoff_description(self) -> str:
        lines = []
        for (a, b), (pts_a, pts_b) in self.payoff_matrix.items():
            lines.append(
                f"\t- If you choose {a.value} and opponent chooses {b.value}: "
                f"you get {pts_a} points, opponent gets {pts_b} points."
            )
        return "\n".join(lines)

    def _prompt_player_action(
        self,
        player: Agent,
        *,
        extra_info: str | None = None,
    ) -> tuple[str, PrisonersDilemmaAction]:
        prompt = self.prompt
        if extra_info:
            prompt += extra_info
        prompt += "\n" + self.default_output_instruction

        LOGGER.write_to_txt(prompt, "game_prompt.txt")
        response, action = player.chat_with_retries(prompt, self._parse_action_choice)
        return response, action

    @staticmethod
    def _parse_action_choice(response: str) -> PrisonersDilemmaAction:
        matches = ACTION_JSON_REGEX.findall(response)
        if not matches:
            raise ValueError("No JSON object found in the response.")

        try:
            payload = json.loads(matches[-1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc.msg}") from exc

        action_raw = payload.get("action")
        if not isinstance(action_raw, str):
            raise ValueError("JSON field 'action' must be a string.")

        token = action_raw.strip().upper()
        if token in {"C", "COOPERATE"}:
            return PrisonersDilemmaAction.COOPERATE
        if token in {"D", "DEFECT"}:
            return PrisonersDilemmaAction.DEFECT
        if token.startswith("A"):
            return PrisonersDilemmaAction.from_token(token)

        raise ValueError(
            f"Unsupported action value {action_raw!r}; expected C/D or A0/A1 tokens."
        )

    def play(
        self,
        additional_info: list[str] | str,
        players: Sequence[Agent],
        action_map=lambda x: x,
    ) -> list[Move]:
        assert len(players) == self.num_players
        player1, player2 = players

        if isinstance(additional_info, str):
            additional_info = [additional_info] * self.num_players

        responses: dict[int, str] = {}
        action_lookup: dict[int, PrisonersDilemmaAction] = {}
        action_indices: dict[int, int] = {}

        for player, info in zip(players, additional_info):
            response, action = self._prompt_player_action(player, extra_info=info)
            uid = player.uid
            responses[uid] = response
            action_lookup[uid] = action
            action_indices[uid] = list(PrisonersDilemmaAction).index(action)

        mapped_indices = action_map(action_indices)

        final_actions: dict[int, PrisonersDilemmaAction] = {}
        for uid, mapped in mapped_indices.items():
            if isinstance(mapped, PrisonersDilemmaAction):
                final_actions[uid] = mapped
            elif isinstance(mapped, str):
                token = mapped.strip().upper()
                if token.startswith("A"):
                    final_actions[uid] = PrisonersDilemmaAction.from_token(token)
                else:
                    final_actions[uid] = PrisonersDilemmaAction(token)
            else:
                final_actions[uid] = PrisonersDilemmaAction.from_index(int(mapped))

        uid1 = player1.uid
        uid2 = player2.uid
        pts1, pts2 = self.payoff_matrix[(final_actions[uid1], final_actions[uid2])]

        return [
            Move(
                player_name=player1.name,
                uid=uid1,
                action=final_actions[uid1],
                points=pts1,
                response=responses[uid1],
            ),
            Move(
                player_name=player2.name,
                uid=uid2,
                action=final_actions[uid2],
                points=pts2,
                response=responses[uid2],
            ),
        ]
