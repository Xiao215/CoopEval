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
    CONTRACT_APPROVAL_VOTE_PROMPT,
    CONTRACT_CONFIRMATION_PROMPT,
    CONTRACT_DESIGN_PROMPT,
    CONTRACT_MECHANISM_PROMPT,
    CONTRACT_REJECTION_PROMPT,
)
from src.utils.concurrency import run_tasks

# Adjust just like mediation

class Contracting(Mechanism):
    """Mechanism where players negotiate and optionally sign payoff contracts."""

    def __init__(
        self,
        base_game: Game,
    ) -> None:
        super().__init__(base_game)
        self.contracts: dict[int, list[int]] = {}
        self.contracts_design_prompt = CONTRACT_DESIGN_PROMPT
        self.contract_confirmation_prompt = CONTRACT_CONFIRMATION_PROMPT
        self.contract_mechanism_prompt = CONTRACT_MECHANISM_PROMPT
        self._cached_agents: list[Agent] | None = None

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
                    self.contracts[designer.uid]
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
                    f"- If a player chooses A{idx}, they receive an additional payment of {payoff} point(s), drawn equally from the other players."
                )
            elif payoff < 0:
                lines.append(
                    f"- If a player chooses A{idx}, they pay an additional payment of {-payoff} point(s), distributed equally among the other players."
                )
            else:
                lines.append(
                    f"- If a player chooses A{idx}, there is no additional payments in either direction."
                )
        return "\n".join(lines)

    def _all_contracts_description(self, players: Sequence[Agent]) -> str:
        """Format all contracts for the voting prompt."""
        lines = []
        for i, player in enumerate(players, start=1):
            contract = self.contracts[player.uid]
            lines.append(f"Contract {i}:")
            lines.append(self._contract_description(contract))
            lines.append("")  # Blank line between contracts
        return "\n".join(lines)

    def _collect_vote(
        self,
        voter: Agent,
        players: Sequence[Agent]
    ) -> tuple[str, dict[int, bool]]:
        """
        Ask an agent to vote on which contracts they approve.

        Returns:
            response (str): The raw response from the voter
            votes (dict[int, bool]): Mapping from contract index to approval (True/False)
        """
        all_contracts = self._all_contracts_description(players)
        vote_prompt = (
            self.base_game.prompt
            + "\n"
            + CONTRACT_APPROVAL_VOTE_PROMPT.format(
                all_contracts_description=all_contracts
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

            # Convert C1, C2, ... to integer indices
            votes = {}
            for i in range(1, len(players) + 1):
                key = f"C{i}"
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

    def _select_contract(
        self,
        players: Sequence[Agent],
        all_votes: dict[int, dict[int, bool]]
    ) -> tuple[int, Agent]:
        """
        Select winning contract based on approval votes.

        Args:
            players: Sequence of players in the matchup
            all_votes: {voter_uid: {contract_index: approval}}

        Returns:
            (winning_index, winning_agent): Index (1-based) and Agent who designed winner
        """
        import random

        # Count approvals per contract
        approval_counts = {i: 0 for i in range(1, len(players) + 1)}
        for _voter_uid, votes in all_votes.items():
            for contract_idx, approved in votes.items():
                if approved:
                    approval_counts[contract_idx] += 1

        # Find max approvals
        max_approvals = max(approval_counts.values())

        # Get all contracts with max approvals (for tie-breaking)
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
        # This ensures each agent gets unique UID and designs their own contract
        agents = super()._create_players_from_cfgs(agent_cfgs)

        # Cache agents so base class reuses them
        self._cached_agents = agents

        def design_fn(agent: Agent) -> tuple[Agent, str, list[int]]:
            response, contract = self._design_contract(agent)
            return agent, response, contract

        design_results = run_tasks(agents, design_fn)

        self.contracts.clear()
        contract_design = {}
        for agent, response, contract in design_results:
            self.contracts[agent.uid] = contract
            contract_design[agent.uid] = {
                "agent_name": agent.name,
                "model_type": agent.model_type,
                "response": response,
                "contract": contract,
            }
        LOGGER.log_record(
            record=contract_design, file_name="contract_design.json"
        )

        # Now call base class - it will use our cached agents
        result = super().run_tournament(agent_cfgs)

        # Clear cache for next run
        self._cached_agents = None

        return result

    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """
        Have players vote on contracts, select winner, get signatures, and play once.

        Returns:
            A list containing a single move sequence (one game result).
        """
        # Step 1: Collect votes from all players
        def collect_vote_fn(player: Agent) -> tuple[Agent, str, dict[int, bool]]:
            response, votes = self._collect_vote(player, players)
            return player, response, votes

        vote_results = run_tasks(players, collect_vote_fn)

        # Step 2: Process voting results
        all_votes = {}  # {voter_uid: {contract_idx: approval}}
        vote_records = []
        for player, response, votes in vote_results:
            all_votes[player.uid] = votes
            vote_records.append({
                "voter_uid": player.uid,
                "voter_name": player.name,
                "votes": votes,
                "response": response,
            })

        # Step 3: Select winning contract
        winning_idx, winning_agent = self._select_contract(players, all_votes)
        winning_contract = self.contracts[winning_agent.uid]

        # Step 4: Collect signatures for the winning contract
        def sign_contract_fn(player: Agent) -> tuple[Agent, str, bool]:
            response, agreement = self._agree_to_contract(
                player=player, designer=winning_agent
            )
            return player, response, agreement

        sign_results = run_tasks(players, sign_contract_fn)

        # Step 5: Process signature results
        # Create player ID mapping (Player 1, Player 2, etc.)
        player_ids = {player: f"Player {i}" for i, player in enumerate(players, start=1)}

        all_agree = True
        rejector_ids = []
        signature_records = {}
        for player, response, agree in sign_results:
            signature_records[player.name] = {
                "response": response,
                "agree": agree,
            }
            if not agree:
                all_agree = False
                rejector_ids.append(player_ids[player])

        # Step 6: Play game once (with or without contract)
        if all_agree:
            contract_prompt = self.contract_mechanism_prompt.format(
                contract_description=self._contract_description(winning_contract)
            )
            additional_info = [contract_prompt] * len(players)
        else:
            rejectors_str = ", ".join(rejector_ids)
            rejection_prompt = CONTRACT_REJECTION_PROMPT.format(
                contract_description=self._contract_description(winning_contract),
                rejector_ids=rejectors_str,
            )
            additional_info = [rejection_prompt] * len(players)

        moves = self.base_game.play(
            additional_info=additional_info,
            players=players,
        )

        # Step 7: Serialize game results
        serialized_moves = [
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

        # Step 8: Log voting, signatures, and game results
        record = {
            "votes": vote_records,
            "selected_contract_index": winning_idx,
            "selected_contract_designer_uid": winning_agent.uid,
            "selected_contract_designer_name": winning_agent.name,
            "signatures": signature_records,
            "all_signed": all_agree,
            "moves": serialized_moves,
        }
        LOGGER.log_record(record=[record], file_name=self.record_file)

        # Return list with single game result (base class will call payoffs.add_profile)
        return [moves]
