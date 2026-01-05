"""Mechanism that lets players propose and sign payoff-altering contracts."""

import json
import re
from typing import Sequence

from src.agents.agent_manager import Agent
from src.ranking_evaluations.population_payoffs import PopulationPayoffs
from src.games.base import Game, Move
from src.logger_manager import LOGGER
from src.mechanisms.base import Mechanism
from src.mechanisms.prompts import (
    CONTRACT_CONFIRMATION_PROMPT,
    CONTRACT_DESIGN_PROMPT,
    CONTRACT_MECHANISM_PROMPT,
)
from src.registry.agent_registry import create_agent
from src.utils.concurrency import run_tasks

# Adjust just like mediation

class Contracting(Mechanism):
    """Mechanism where players negotiate and optionally sign payoff contracts."""

    def __init__(
        self,
        base_game: Game,
    ) -> None:
        super().__init__(base_game)
        self.contracts: dict[str, list[int]] = {}
        self.contracts_design_prompt = CONTRACT_DESIGN_PROMPT
        self.contract_confirmation_prompt = CONTRACT_CONFIRMATION_PROMPT
        self.contract_mechanism_prompt = CONTRACT_MECHANISM_PROMPT

    def _design_contract(self, designer: Agent) -> tuple[str, list[int]]:
        """
        Design a contract from the given LLM agent.

        Returns:
            response (str): The raw response from the designer.
            contract (dict[int]): The contract with index representing the action
                and value representing the payoff adjustment.
        """
        base_prompt = (
            self.base_game.prompt + "\n" + self.contracts_design_prompt.format()
        )
        response, contract = designer.chat_with_retries(
            base_prompt=base_prompt,
            parse_func=self._parse_contract,
        )
        return response, contract

    def _agree_to_contract(
        self, *, player: Agent, designer: Agent
    ) -> tuple[str, bool]:
        """
        Ask the LLM to confirm agreement to the contract with automatic retries.
        """
        base_prompt = (
            self.base_game.prompt
            + "\n"
            + self.contract_confirmation_prompt.format(
                contract_description=self._contract_description(
                    self.contracts[designer.model_type]
                )
            )
        )
        response, agreement = player.chat_with_retries(
            base_prompt=base_prompt,
            parse_func=self._parse_agreement,
        )
        return response, agreement

    def _parse_contract(self, response: str) -> list[int]:
        """
        Parse the contract design from the response.
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

        n = self.base_game.num_actions
        got_keys = set(json_obj.keys())
        missing = set(f"A{i}" for i in range(n)) - got_keys
        extra = got_keys - set(f"A{i}" for i in range(n))
        if extra:
            raise ValueError(f"Action key mismatch. Extra: {sorted(extra)}")
        if missing:
            raise ValueError(f"Action key mismatch. Missing: {sorted(missing)}")

        contract = [0] * n
        for k, v in json_obj.items():
            if not isinstance(v, int):
                raise ValueError(f"Value for {k} must be an integer, got {v!r}")
            idx = int(k[1:])  # strip the leading 'A'
            contract[idx] = v
        return contract

    def _parse_agreement(self, response: str) -> bool:
        """
        Parse the agreement to the contract from the response.
        Expecting a JSON object in string format.
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

        if "sign" not in json_obj:
            raise ValueError(f"Missing 'sign' key in the response {response!r}")
        sign = json_obj["sign"]
        if not isinstance(sign, bool):
            raise ValueError(f"'sign' value must be a boolean, got {sign!r}")
        return sign

    def _contract_description(self, contract: list[int]) -> str:
        """Format the prompt for the contract agent.

        Args:
            contract (dict[int]): The contract with index representing the action
                and value representing the payoff adjustment.
        """
        lines = []
        for idx, payoff in enumerate(contract):
            if payoff > 0:
                lines.append(
                    f"- If you choose A{idx}, you receive a total of {payoff} from each other player."
                )
            elif payoff < 0:
                lines.append(
                    f"- If you choose A{idx}, you pay a total of {-payoff} to each other player."
                )
            else:
                lines.append(
                    f"- If you choose A{idx}, there is no extra payoff."
                )
        return "\n".join(lines)

    def run_tournament(self, agent_cfgs: list[dict]) -> PopulationPayoffs:
        agents = [create_agent(cfg) for cfg in agent_cfgs]

        def design_fn(agent: Agent) -> tuple[Agent, str, list[int]]:
            response, contract = self._design_contract(agent)
            return agent, response, contract

        design_results = run_tasks(agents, design_fn)

        self.contracts.clear()
        contract_design = {}
        for agent, response, contract in design_results:
            self.contracts[agent.model_type] = contract
            contract_design[agent.model_type] = {
                "response": response,
                "contract": contract,
            }
        LOGGER.log_record(
            record=contract_design, file_name="contract_design.json"
        )
        return super().run_tournament(agent_cfgs)

    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """Have each designer propose a contract and play the base game.

        Returns:
            A list of move sequences (one sequence per designer's contract round).
        """
        records = []
        matchup_moves = []

        for designer in players:
            record = {
                "designer": designer.name,
                "agreements": {},
            }

            agreement_results = run_tasks(
                players,
                lambda p, d=designer: self._agree_to_contract(
                    player=p, designer=d
                ),
            )

            all_agree = True
            for player, (response, agree) in zip(players, agreement_results):
                record["agreements"][player.name] = {
                    "response": response,
                    "agree": agree,
                }
                if not agree:
                    all_agree = False
            record["all_agree"] = all_agree

            if all_agree:
                contract_prompt = self.contract_mechanism_prompt.format(
                    contract_description=self._contract_description(
                        self.contracts[designer.model_type]
                    )
                )
                additional_info = [contract_prompt] * len(players)
            else:
                additional_info = ["None."] * len(players)

            moves = self.base_game.play(
                additional_info=additional_info,
                players=players,
            )

            # Record keeping for the JSON log
            record["moves"] = [
                {
                    "uid": move.uid,
                    "player_name": move.player_name,
                    "action": (
                        move.action.value
                        if hasattr(move.action, "value")
                        else str(move.action)
                    ),
                    "points": move.points,
                    "response": move.response,
                }
                for move in moves
            ]

            matchup_moves.append(moves)
            records.append(record)

        LOGGER.log_record(record=records, file_name=self.record_file)

        return matchup_moves
