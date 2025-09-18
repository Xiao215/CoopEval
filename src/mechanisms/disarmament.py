"""Repeated-game mechanism that lets agents negotiate probability caps."""

import json
import re
import textwrap
from typing import Any, Sequence

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.games.base import Game
from src.mechanisms.base import RepetitiveMechanism
from src.logger_manager import LOGGER
from src.utils.concurrency import run_tasks


class Disarmament(RepetitiveMechanism):
    """
    Disarmament mechanism that allows for multiple rounds of the same game.
    """

    def __init__(
        self,
        base_game: Game,
        num_rounds: int,
        discount: float,
    ) -> None:
        super().__init__(base_game, num_rounds, discount)

        self.disarm_prompt = textwrap.dedent(
            """
        Instruction:
        You are negotiating a multi-action disarmament for this game.

        A "cap" is the maximum probability (in %) with which you may choose an action in this game.
        Your current caps:
            {my_caps}
        Opponents' current caps:
            {opponents_caps}

        Rules:
        1) For each action, you may keep the cap the same or reduce it. Increases are forbidden.
        2) Each cap must be an integer in [0, 100].
        3) All caps must be non-negative and the sum of your caps must be greater than or equal to 100.
        4) Moves are simultaneous; assume others facing the same negotiation.
        5) If at least one party reduces any cap, negotiations continue to another round with probability {discount}% (otherwise they end).
        6) If every party leaves all caps unchanged, negotiations end immediately.


        Format requirement:
        Return the new cap as a JSON object, for example:
        {{"A0": <INT>, "A1": <INT>, ...}}
        """
        )
        self.current_disarm_caps: dict[str, list[float]] = {}
        self._id_to_name: dict[str, str] = {}

        self.disarmament_mechanism_prompt = textwrap.dedent(
            """
        Additional Information:
        A "cap" is the maximum probability (in %) with which you may choose an action in the next round.
        From previous round of negotiation, you agree to have a cap of:
        {caps_str}

        Now you need to propose a new probability distribution over actions subjected to your current cap limits.
        """
        )

    def _format_prompt(
        self,
        player_id: str,
    ) -> str:
        """
        Build the filled prompt:
        - caps_by_agent: {agent_name: [cap_A0, cap_A1, ...]} (ints 0..100)
        - player_id: identifier for the agent whose 'my_caps' will be shown
        - discount: continuation probability (integer percent)
        """

        my_caps_line = self._caps_to_line(self.current_disarm_caps[player_id])

        # Sort opponents to ensure increasing opponent id
        opponents = sorted(
            pid for pid in self.current_disarm_caps.keys() if pid != player_id
        )
        opp_lines = []
        for idx, opponent_id in enumerate(opponents, start=1):
            opp_lines.append(
                f"  • Opponent {idx}: {self._caps_to_line(self.current_disarm_caps[opponent_id])}"
            )
        opponents_caps_block = "\n".join(opp_lines)

        return self.disarm_prompt.format(
            my_caps=my_caps_line,
            opponents_caps=opponents_caps_block,
            discount=self.discount * 100,
        )

    @staticmethod
    def _caps_to_line(caps: list[float]) -> str:
        """Return '{"A0"=<cap0>, "A1"=<cap1>, ...}'."""
        return "{" + ", ".join(f'"A{i}"={int(c)}' for i, c in enumerate(caps)) + "}"

    def _negotiate_disarm_caps(
        self,
        player: Agent,
    ) -> tuple[str, tuple[list[float], bool]]:
        """
        Ask the LLM for disarm caps and parse with automatic retries.
        Returns: (new_caps, changed, last_response). On total failure, returns (old_caps, False, last_response)
        and prints a warning.
        """
        player_id = player.label
        base_prompt = self.base_game.prompt + "\n" + self._format_prompt(player_id)
        parse_func = lambda resp: self._parse_disarm_caps(resp, player_id)
        response, (new_caps, changed) = player.chat_with_retries(
            base_prompt=base_prompt,
            parse_func=parse_func,
        )
        return response, (new_caps, changed)

    def _parse_disarm_caps(
        self,
        response: str,
        player_id: str,
    ) -> tuple[list[float], bool]:
        """Parse the disarmament probabilities (new caps) from the agent's response."""
        matches = re.findall(r"\{.*?\}", response, re.DOTALL)
        if not matches:
            raise ValueError(f"No JSON object found in the response {response!r}")
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

        new_caps = [0.0] * n
        old_caps = self.current_disarm_caps[player_id]
        for act_str, cap in json_obj.items():
            idx = int(act_str[1:])  # strip the leading 'A'
            if not 0 <= idx < n:
                raise ValueError(f"A{idx} does not exist as a valid action")
            if not 0 <= cap <= 100:
                raise ValueError(f"Disarm cap {cap} out of range for A{idx}")
            if cap > old_caps[idx]:
                raise ValueError(
                    f"New cap {cap} of A{idx} greater than its old cap {old_caps[idx]}"
                )
            new_caps[idx] = cap

        # Rule: caps must sum to >= 100
        if sum(new_caps) < 100:
            raise ValueError(
                f"Sum of your proposed caps is {sum(new_caps)}, but must be at least 100"
            )

        changed = any(nc < oc for nc, oc in zip(new_caps, old_caps, strict=True))
        return new_caps, changed

    def _play_matchup(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> None:
        id_to_name = {player.label: player.name for player in players}
        self._id_to_name = id_to_name

        disarmed_cap: dict[str, list[float]] = {
            player.label: [100.0 for _ in range(self.base_game.num_actions)]
            for player in players
        }
        disarmament_records = []
        for _ in range(self.num_rounds):
            new_disarmed_cap: dict[str, list[float]] = {}
            negotiation_continue = False
            disarmament_mechanisms: list[str] = []
            round_records: list[dict[str, Any]] = []

            # Prepare tasks for players who still have wiggle room on their caps
            # Sync current caps for prompt formatting
            self.current_disarm_caps = disarmed_cap

            negotiable_players = [
                player for player in players if sum(disarmed_cap[player.label]) > 100.0
            ]
            negotiation_results: dict[str, tuple[str, tuple[list[float], bool]]] = {}
            if negotiable_players:
                results = run_tasks(
                    negotiable_players,
                    self._negotiate_disarm_caps,
                )
                negotiation_results = {
                    player.label: result
                    for player, result in zip(negotiable_players, results, strict=True)
                }

            for player in players:
                pid = player.label
                if pid in negotiation_results:
                    disarm_rsp, (new_player_cap, changed) = negotiation_results[pid]
                    negotiation_continue |= changed
                    new_disarmed_cap[pid] = new_player_cap
                else:
                    disarm_rsp = "No room for further disarmament, keep the same cap."
                    new_disarmed_cap[pid] = disarmed_cap[pid]

                round_records.append(
                    {
                        "player_id": pid,
                        "player": id_to_name[pid],
                        "response": disarm_rsp,
                        "new_cap": new_disarmed_cap[pid],
                    }
                )
                caps_str = self._caps_to_line(new_disarmed_cap[pid])
                disarmament_mechanisms.append(
                    self.disarmament_mechanism_prompt.format(caps_str=caps_str)
                )

            moves = self.base_game.play(
                players=players, additional_info=disarmament_mechanisms
            )
            payoffs.add_profile(moves)
            self.current_disarm_caps = new_disarmed_cap

            disarmament_records.append(
                [
                    {
                        **r,
                        **m.to_dict(),
                        "match_id": "|".join(sorted(p.name for p in players)),
                    }
                    for r, m in zip(round_records, moves)
                ]
            )

            disarmed_cap = new_disarmed_cap
            if not negotiation_continue:
                break
        LOGGER.log_record(record=disarmament_records, file_name=self.record_file)
