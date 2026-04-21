"""Mechanism that lets players propose and sign payoff-altering contracts."""

import asyncio
import random
from typing import Sequence, override

from coopeval.agents.agent_manager import Agent
from coopeval.games.base import Action, Game, Move
from coopeval.logger_manager import LOGGER
from coopeval.mechanisms.base import Mechanism
from coopeval.mechanisms.prompts import (
    CONTRACT_APPROVAL_VOTE_PROMPT,
    CONTRACT_CONFIRMATION_PROMPT,
    CONTRACT_DESIGN_PROMPT,
    CONTRACT_MECHANISM_PROMPT,
    CONTRACT_REJECTION_PROMPT,
)
from coopeval.mechanisms.json_parsing import (
    parse_action_value_map,
    parse_bool_votes,
)
from coopeval.ranking_evaluations.payoffs_base import PayoffsBase
from coopeval.utils.json_io import extract_json_object_from_end

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

    async def _design_contract(self, designer: Agent) -> tuple[str, Contract]:
        """
        Design a contract from the given LLM agent.

        Returns:
            response (str): The raw response from the designer.
            contract (Contract): The contract with Action keys and integer payoff adjustments.
        """
        game_prompt = self.base_game.get_player_prompt(designer.player_id)
        base_prompt = game_prompt + "\n" + CONTRACT_DESIGN_PROMPT.format()
        _response, trace_id, contract = await designer.chat_with_retries(
            base_prompt=base_prompt,
            parse_func=self._parse_contract,
        )
        return trace_id, contract

    async def _agree_to_contract(
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
        response, trace_id, agreement = await player.chat_with_retries(
            base_prompt=base_prompt,
            parse_func=self._parse_agreement,
        )
        return response, trace_id, agreement

    def _parse_contract(self, response: str) -> Contract:
        """
        Parse the contract design from the response.
        Expecting a Python dictionary in string format.
        """
        values = parse_action_value_map(
            response,
            num_actions=self.base_game.num_actions,
        )
        return {
            self.base_game.action_class.from_index(idx): value
            for idx, value in values.items()
        }

    def _parse_agreement(self, response: str) -> bool:
        """
        Parse the agreement to the contract from the response.
        Expecting a JSON object in string format.
        """
        json_obj = extract_json_object_from_end(response)

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

    async def _collect_vote(
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
            return parse_bool_votes(
                response,
                prefix="C",
                count=len(players),
            )

        _, trace_id, votes = await voter.chat_with_retries(
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

        approval_counts = {i: 0 for i in range(1, len(players) + 1)}
        for _voter, votes in all_votes.items():
            for contract_idx, approved in votes.items():
                if approved:
                    approval_counts[contract_idx] += 1

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
        self._cached_agents = players

        async def design_fn(player: Agent) -> tuple[Agent, str, Contract]:
            trace_id, contract = await self._design_contract(player)
            return player, trace_id, contract

        tasks = [design_fn(player) for player in players]
        design_results = await asyncio.gather(*tasks)

        self.contracts.clear()
        contract_design = {}
        for player, trace_id, contract in design_results:
            self.contracts[player.name] = contract
            contract_design[player.name] = {
                "contract": {str(k): v for k, v in contract.items()},
                "trace_id": trace_id,
            }
        LOGGER.log_record(
            record=contract_design, file_name="contract_design.json"
        )

        result = await super()._run_tournament_async(players)

        self._cached_agents = None

        return result

    def _apply_contract(
        self,
        moves: list[Move],
        selected_contract: Contract,
    ) -> list[Move]:
        """
        Adjust payoffs based on the contract logic:
        A player performing Action X gets +Payoff.
        This amount is deducted equally from all other players.
        """
        actions = [move.action for move in moves]
        adjustments = self.compute_contract_adjustments(
            actions, selected_contract
        )
        adjusted_moves: list[Move] = []
        for move, adjustment in zip(moves, adjustments):
            adjusted_move = Move(
                player=move.player,
                action=move.action,
                points=move.points + adjustment,
                trace_id=move.trace_id,
                mediated=move.mediated,
            )
            adjusted_moves.append(adjusted_move)

        return adjusted_moves

    def compute_contract_adjustments(
        self,
        actions: Sequence[Action],
        contract: Contract,
    ) -> list[float]:
        """Return the payoff deltas the contract induces for each player action."""

        num_players = self.base_game.num_players
        if num_players <= 1:
            raise ValueError(
                "Contract adjustments require at least two players."
            )
        if len(actions) != num_players:
            raise ValueError(
                "Actions sequence must include one entry per player."
            )

        adjustments = [0.0] * len(actions)
        for i, action in enumerate(actions):
            contract_adjustment = float(contract.get(action, 0))
            if contract_adjustment == 0:
                continue

            adjustments[i] += contract_adjustment
            cost_per_other = contract_adjustment / (num_players - 1)
            for j in range(len(actions)):
                if j != i:
                    adjustments[j] -= cost_per_other

        return adjustments

    @override
    async def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """
        Have players vote on contracts, select winner, get signatures, and play once.
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

        winning_idx, winning_agent = self._select_contract(players, all_votes)
        winning_contract = self.contracts[winning_agent.name]

        async def sign_contract_fn(player: Agent) -> tuple[Agent, str, bool]:
            _, trace_id, agreement = await self._agree_to_contract(
                player=player, designer=winning_agent
            )
            return player, trace_id, agreement

        tasks2 = [sign_contract_fn(player) for player in players]
        sign_results = await asyncio.gather(*tasks2)

        player_ids = {
            player: f"Player {player.player_id}" for player in players
        }

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
                contract_description=self._contract_description(
                    winning_contract
                ),
                rejector_ids=rejectors_str,
                designer_player_id=winning_agent.player_id,
            )
            additional_info = [rejection_prompt] * len(players)

        moves = await self.base_game.play(
            additional_info=additional_info,
            players=players,
        )

        if all_agree:
            moves = self._apply_contract(moves, winning_contract)

        record = {
            "votes": vote_records,
            "selected_contract_index": winning_idx,
            "selected_contract_designer_name": winning_agent.name,
            "signatures": signature_records,
            "all_signed": all_agree,
            "moves": moves,
        }
        LOGGER.log_record(record=[record], file_name=self.record_file)

        return [moves]
