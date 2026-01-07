"""Repeated-game mechanism that lets agents negotiate probability caps."""

import json
import re
from typing import Any, Sequence

from src.agents.agent_manager import Agent
from src.games.base import Game, Move
from src.logger_manager import LOGGER
from src.mechanisms.base import RepetitiveMechanism
from src.mechanisms.prompts import (
    DISARMAMENT_MECHANISM_PROMPT,
    DISARM_PROMPT,
)
from src.utils.concurrency import run_tasks

Caps = list[float]
CapsByPlayer = dict[int, Caps]
NegotiationResult = tuple[str, Caps]


class Disarmament(RepetitiveMechanism):
    """
    Disarmament mechanism that allows for multiple rounds of the same game.
    """

    _NO_ROOM_MESSAGE = "No room for further disarmament, keep the same cap."

    def __init__(
        self,
        base_game: Game,
        num_rounds: int,
        discount: float,
    ) -> None:
        super().__init__(base_game, num_rounds, discount)

        self.disarm_prompt = DISARM_PROMPT
        self.current_disarm_caps: CapsByPlayer = {}

        self.disarmament_mechanism_prompt = DISARMAMENT_MECHANISM_PROMPT

    def _format_prompt(
        self,
        uid: int,
    ) -> str:
        """
        Build the filled prompt:
        - caps_by_agent: {agent_name: [cap_A0, cap_A1, ...]} (ints 0..100)
        - player_id: identifier for the agent whose 'my_caps' will be shown
        - discount: continuation probability (integer percent)
        """

        if uid not in self.current_disarm_caps:
            raise KeyError(f"Caps have not been initialized for player {uid}")

        self_caps_description = self._caps_description(
            self.current_disarm_caps[uid]
        )

        opponent_labels = {
            opponent_uid: f"Player {self.uid_to_player_id[opponent_uid]}"
            for opponent_uid in self.current_disarm_caps.keys()
            if opponent_uid != uid
        }

        opp_lines = []
        for opponent_uid, opponent_label in opponent_labels.items():
            opponent_caps = self._caps_description(
                self.current_disarm_caps[opponent_uid]
            )
            opp_lines.append(f"\t{opponent_label}: {opponent_caps}")
        opponents_caps_block = "\n".join(opp_lines)

        return self.disarm_prompt.format(
            my_caps=self_caps_description,
            opponents_caps=opponents_caps_block,
            discount=round(self.discount * 100),
        )

    @staticmethod
    def _caps_description(caps: Caps) -> str:
        """Return '{"A0"=<cap0>, "A1"=<cap1>, ...}'."""
        return (
            "{"
            + ", ".join(f'"A{i}"={int(c)}' for i, c in enumerate(caps))
            + "}"
        )

    def _negotiate_disarm_caps(
        self,
        player: Agent,
    ) -> NegotiationResult:
        """Request updated caps for ``player`` and return the parsed response."""
        uid = player.uid
        base_prompt = self.base_game.prompt + "\n" + self._format_prompt(uid)

        def parse_func(resp: str) -> list[float]:
            return self._parse_disarm_caps(resp, uid)

        response, new_caps = player.chat_with_retries(
            base_prompt=base_prompt,
            parse_func=parse_func,
        )
        return response, new_caps

    def _parse_disarm_caps(
        self,
        response: str,
        uid: int,
    ) -> Caps:
        """Parse the disarmament probabilities (new caps) from the agent's response."""
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

        if not isinstance(json_obj, dict):
            raise ValueError(
                "Parsed JSON must be an object mapping actions to caps"
            )

        n = self.base_game.num_actions
        got_keys = set(json_obj.keys())
        missing = set(f"A{i}" for i in range(n)) - got_keys
        extra = got_keys - set(f"A{i}" for i in range(n))
        if extra:
            raise ValueError(f"Action key mismatch. Extra: {sorted(extra)}")
        if missing:
            raise ValueError(f"Action key mismatch. Missing: {sorted(missing)}")

        new_caps: Caps = [0.0] * n
        old_caps = self.current_disarm_caps[uid]
        for act_str, cap in json_obj.items():
            idx = int(act_str[1:])  # strip the leading 'A'
            if not 0 <= idx < n:
                raise ValueError(f"A{idx} does not exist as a valid action")
            if not isinstance(cap, (int, float)):
                raise ValueError(
                    f"Cap for {act_str} must be numeric, got {type(cap)!r}"
                )
            cap_value = float(cap)
            if not 0 <= cap_value <= 100:
                raise ValueError(f"Disarm cap {cap} out of range for A{idx}")
            if cap_value > old_caps[idx]:
                raise ValueError(
                    f"New cap {cap_value} of A{idx} greater than its old cap {old_caps[idx]}"
                )
            new_caps[idx] = cap_value

        # Rule: caps must sum to >= 100
        if sum(new_caps) < 100:
            raise ValueError(
                f"Sum of your proposed caps is {sum(new_caps)}, but must be at least 100"
            )

        return new_caps

    def _run_negotiations(
        self,
        negotiable_players: Sequence[Agent],
    ) -> dict[int, NegotiationResult]:
        """Collect fresh caps for each negotiable player (if any)."""

        if not negotiable_players:
            return {}

        results = run_tasks(negotiable_players, self._negotiate_disarm_caps)
        return {
            player.uid: result
            for player, result in zip(negotiable_players, results, strict=True)
        }

    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """
        Run the disarmament negotiation rounds.

        Returns:
            A list of move sequences (one sequence per round played).
        """
        # Ensure we start with fresh caps for this specific match
        disarmed_cap: CapsByPlayer = {
            player.uid: [100.0 for _ in range(self.base_game.num_actions)]
            for player in players
        }

        self.uid_to_player_id = {player.uid: player.player_id for player in players}

        disarmament_records: list[list[dict[str, Any]]] = []
        matchup_moves = []

        # Note: If self.history persists across matches, ensure it is cleared here
        # or managed externally. For safety, we just track local moves for the return.

        for _ in range(self.num_rounds):
            self.current_disarm_caps = disarmed_cap

            negotiable_players = [
                player
                for player in players
                if sum(disarmed_cap[player.uid]) > 100.0
            ]
            negotiation_results = self._run_negotiations(negotiable_players)

            new_disarmed_cap: CapsByPlayer = {}
            negotiation_continue = False
            disarmament_mechanisms: list[str] = []
            round_records: list[dict[str, Any]] = []

            for player in players:
                pid = player.uid
                if pid in negotiation_results:
                    disarm_rsp, player_cap = negotiation_results[pid]
                else:
                    disarm_rsp = self._NO_ROOM_MESSAGE
                    player_cap = disarmed_cap[pid]

                negotiation_continue |= player_cap != disarmed_cap[pid]
                new_disarmed_cap[pid] = player_cap

                round_records.append(
                    {
                        "player": pid,
                        "response": disarm_rsp,
                        "new_cap": player_cap,
                    }
                )
                caps_str = self._caps_description(player_cap)
                disarmament_mechanisms.append(
                    self.disarmament_mechanism_prompt.format(caps_str=caps_str)
                )

            moves = self.base_game.play(
                players=players, additional_info=disarmament_mechanisms
            )

            self.current_disarm_caps = new_disarmed_cap

            disarmament_records.append(
                [
                    {
                        **r,
                        "uid": m.uid,
                        "player_name": m.player_name,
                        "action": (
                            m.action.value
                            if hasattr(m.action, "value")
                            else str(m.action)
                        ),
                        "points": m.points,
                        "response": m.response,
                        "match_id": "|".join(sorted(p.name for p in players)),
                    }
                    for r, m in zip(round_records, moves)
                ]
            )

            self.history.append(moves)
            matchup_moves.append(moves)

            disarmed_cap = new_disarmed_cap
            if not negotiation_continue:
                break

        LOGGER.log_record(
            record=disarmament_records, file_name=self.record_file
        )

        return matchup_moves
