"""Mechanism that lets players propose and sign payoff-altering contracts."""

import json
import random
import re
from typing import Sequence, override

from src.agents.agent_manager import Agent
from src.games.base import Action, Game, Move
from src.logger_manager import LOGGER
from src.mechanisms.base import Mechanism
from src.mechanisms.prompts import (CONTRACT_APPROVAL_VOTE_PROMPT,
                                    CONTRACT_CONFIRMATION_PROMPT,
                                    CONTRACT_DESIGN_PROMPT,
                                    CONTRACT_MECHANISM_PROMPT,
                                    CONTRACT_REJECTION_PROMPT)
from src.ranking_evaluations.payoffs_base import PayoffsBase
from src.utils.concurrency import run_tasks

Contract = dict[Action, int]

class Contracting(Mechanism):
    """Mechanism where players negotiate and optionally sign payoff contracts."""

    def __init__(
        self,
        base_game: Game,
        *,
        tournament_workers: int = 1,
    ) -> None:
        super().__init__(base_game, tournament_workers=tournament_workers)
        self.contracts: dict[str, Contract] = {}
        self._cached_agents: list[Agent] | None = None

    def _design_contract(self, designer: Agent) -> tuple[str, str, Contract]:
        """
        Design a contract from the given LLM agent.

        Returns:
            response (str): The raw response from the designer.
            contract (Contract): The contract with Action keys and integer payoff adjustments.
        """
        game_prompt = self.base_game.get_player_prompt(designer.player_id)
        base_prompt = game_prompt + "\n" + CONTRACT_DESIGN_PROMPT.format()
        response, trace_id, contract = designer.chat_with_retries(
            base_prompt=base_prompt,
            parse_func=self._parse_contract,
        )
        return response, trace_id, contract

    def _agree_to_contract(
        self, *, player: Agent, designer: Agent
    ) -> tuple[str, str, bool]:
        """
        Ask the LLM to confirm agreement to the contract with automatic retries.
        """
        game_prompt = self.base_game.get_player_prompt(player.player_id)
        base_prompt = (
            game_prompt
            + "\n"
            + CONTRACT_CONFIRMATION_PROMPT.format(
                contract_description=self._contract_description(
                    self.contracts[designer.name]
                ),
                designer_player_id=designer.player_id,
            )
        )
        response, trace_id, agreement = player.chat_with_retries(
            base_prompt=base_prompt,
            parse_func=self._parse_agreement,
        )
        return response, trace_id, agreement

    def _parse_contract(self, response: str) -> Contract:
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
        json_obj = json.loads(json_str)

        got_actions = set(
            self.base_game.action_class.from_token(k) for k in json_obj.keys()
        )
        expected_actions = set(self.base_game.action_class)
        if got_actions != expected_actions:
            raise ValueError(
                f"Action key mismatch. Expected {[a.to_token() for a in expected_actions]}, "
                f"Got {[a.to_token() for a in got_actions]}"
            )

        contract = {
            self.base_game.action_class.from_token(k): int(v)
            for k, v in json_obj.items()
        }
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
        json_obj = json.loads(json_str)

        if "sign" not in json_obj:
            raise ValueError(f"Missing 'sign' key in the response {response!r}")
        sign = json_obj["sign"]
        if not isinstance(sign, bool):
            raise ValueError(f"'sign' value must be a boolean, got {sign!r}")
        return sign

    def _contract_description(self, contract: Contract) -> str:
        """Format the prompt for the contract agent.

        Args:
            contract (dict[int]): The contract with index representing the action
                and value representing the payoff adjustment.
        """
        lines = []
        for action, payoff in contract.items():
            if payoff > 0:
                lines.append(
                    f"- If a player chooses {action.to_token()}, they receive an additional payment of {payoff} point(s), drawn equally from the other players."
                )
            elif payoff < 0:
                lines.append(
                    f"- If a player chooses {action.to_token()}, they pay an additional payment of {-payoff} point(s), distributed equally among the other players."
                )
            else:
                lines.append(
                    f"- If a player chooses {action.to_token()}, there is no additional payments in either direction."
                )
        return "\n".join(lines)

    def _all_contracts_description(self, players: Sequence[Agent]) -> str:
        """Format all contracts for the voting prompt."""
        lines = []
        for player in players:
            contract = self.contracts[player.name]
            lines.append(f"Contract proposed by Player {player.player_id}:")
            lines.append(self._contract_description(contract))
            lines.append("")
        return "\n".join(lines)

    def _collect_vote(
        self, voter: Agent, players: Sequence[Agent]
    ) -> tuple[str, dict[int, bool]]:
        """
        Ask an agent to vote on which contracts they approve.

        Returns:
            response (str): The raw response from the voter
            votes (dict[int, bool]): Mapping from contract index to approval (True/False)
        """
        game_prompt = self.base_game.get_player_prompt(voter.player_id)
        all_contracts = self._all_contracts_description(players)
        vote_prompt = (
            game_prompt
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
            json_obj = json.loads(json_str)

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

        _, trace_id, votes = voter.chat_with_retries(
            base_prompt=vote_prompt,
            parse_func=parse_votes,
        )
        return trace_id, votes

    def _select_contract(
        self, players: Sequence[Agent], all_votes: dict[Agent, dict[int, bool]]
    ) -> tuple[int, Agent]:
        """
        Select winning contract based on approval votes.

        Args:
            players: Sequence of players in the matchup
            all_votes: {voter_uid: {contract_index: approval}}

        Returns:
            (winning_index, winning_agent): Index (1-based) and Agent who designed winner
        """

        # Count approvals per contract
        approval_counts = {i: 0 for i in range(1, len(players) + 1)}
        for _voter, votes in all_votes.items():
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

    @override
    def run_tournament(self, players: list[Agent]) -> PayoffsBase:
        # Cache agents so base class reuses them
        self._cached_agents = players

        def design_fn(player: Agent) -> tuple[Agent, str, str, Contract]:
            response, trace_id, contract = self._design_contract(player)
            return player, response, trace_id, contract

        design_results = run_tasks(players, design_fn)

        self.contracts.clear()
        contract_design = {}
        for player, response, trace_id, contract in design_results:
            self.contracts[player.name] = contract
            contract_design[player.name] = {
                "response": response,
                "contract": contract,
                "trace_id": trace_id,
            }
        LOGGER.log_record(
            record=contract_design, file_name="contract_design.json"
        )

        # Now call base class - it will use our cached agents
        result = super().run_tournament(players)

        # Clear cache for next run
        self._cached_agents = None

        return result

    def _apply_contract(
        self,
        moves: list[Move],
        selected_contract: Contract,
    ) -> None:
        """
        Adjust payoffs based on the contract logic:
        A player performing Action X gets +Payoff.
        This amount is deducted equally from all other players.
        """
        for i, move in enumerate(moves):
            contract_adjustment = selected_contract[move.action]

            if contract_adjustment != 0:
                move.points += contract_adjustment
                other_moves = moves[:i] + moves[i + 1 :]

                cost_per_other = contract_adjustment / (
                    self.base_game.num_players - 1
                )
                for other_move in other_moves:
                    other_move.points -= cost_per_other

    @override
    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """
        Have players vote on contracts, select winner, get signatures, and play once.

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
        all_votes = {}  # {voter: {contract_idx: approval}}
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

        # Step 3: Select winning contract
        winning_idx, winning_agent = self._select_contract(players, all_votes)
        winning_contract = self.contracts[winning_agent.name]

        # Step 4: Collect signatures for the winning contract
        def sign_contract_fn(player: Agent) -> tuple[Agent, str, bool]:
            _, trace_id, agreement = self._agree_to_contract(
                player=player, designer=winning_agent
            )
            return player, trace_id, agreement

        sign_results = run_tasks(players, sign_contract_fn)

        # Step 5: Process signature results
        player_ids = {player: f"Player {player.player_id}" for player in players}

        all_agree = True
        rejector_ids = []
        signature_records = {}
        for player, trace_id, agree in sign_results:
            signature_records[player.name] = {
                "trace_id": trace_id,
                "agree": agree,
            }
            if not agree:
                all_agree = False
                rejector_ids.append(player_ids[player])

        # Step 6: Play game once (with or without contract)
        if all_agree:
            contract_prompt = CONTRACT_MECHANISM_PROMPT.format(
                contract_description=self._contract_description(
                    winning_contract
                ),
                designer_player_id=winning_agent.player_id,
            )
            additional_info = [contract_prompt] * len(players)
        else:
            rejectors_str = ", ".join(rejector_ids)
            rejection_prompt = CONTRACT_REJECTION_PROMPT.format(
                contract_description=self._contract_description(winning_contract),
                rejector_ids=rejectors_str,
                designer_player_id=winning_agent.player_id
            )
            additional_info = [rejection_prompt] * len(players)

        moves = self.base_game.play(
            additional_info=additional_info,
            players=players,
        )

        if all_agree:
            self._apply_contract(moves, winning_contract)

        # Step 8: Log voting, signatures, and game results
        record = {
            "votes": vote_records,
            "selected_contract_index": winning_idx,
            "selected_contract_designer_name": winning_agent.name,
            "signatures": signature_records,
            "all_signed": all_agree,
            "moves": moves,
        }
        LOGGER.log_record(record=[record], file_name=self.record_file)

        # Return list with single game result (base class will call payoffs.add_profile)
        return [moves]
