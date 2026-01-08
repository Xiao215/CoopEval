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
    DISARM_PROMPT_BASE,
    DISARM_FORMAT_CAN_DISARM,
    DISARM_FORMAT_CANNOT_DISARM,
)
from src.utils.concurrency import run_tasks

Caps = list[float]
CapsByPlayer = dict[int, Caps]
NegotiationResult = tuple[str, str, Caps]  # (response, choice, caps)


class Disarmament(RepetitiveMechanism):
    """
    Disarmament mechanism that allows for multiple rounds of disarming before playing the base game under disarmament restrictions.
    """

    _NO_ROOM_MESSAGE = "No room for further disarmament, so the same upper bounds remain."

    def __init__(
        self,
        base_game: Game,
        num_rounds: int,
        discount: float,
    ) -> None:
        super().__init__(base_game, num_rounds, discount)

        self.disarm_prompt_base = DISARM_PROMPT_BASE
        self.disarm_format_can_disarm = DISARM_FORMAT_CAN_DISARM
        self.disarm_format_cannot_disarm = DISARM_FORMAT_CANNOT_DISARM
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
            raise KeyError(f"Upper bounds have not been initialized for player {uid}")

        self_caps_description = self._caps_description(
            self.current_disarm_caps[uid]
        )

        other_player_labels = {
            other_player_uid: f"Player {self.uid_to_player_id[other_player_uid]}"
            for other_player_uid in self.current_disarm_caps.keys()
            if other_player_uid != uid
        }

        opp_lines = []
        for other_player_uid, other_player_label in other_player_labels.items():
            other_player_caps = self._caps_description(
                self.current_disarm_caps[other_player_uid]
            )
            opp_lines.append(f"\t{other_player_label}: {other_player_caps}")
        other_players_caps_block = "\n".join(opp_lines)

        # Build base prompt with current state
        base_prompt = self.disarm_prompt_base.format(
            my_caps=self_caps_description,
            other_players_caps=other_players_caps_block,
            discount=round(self.discount * 100),
        )

        # Append appropriate format requirement based on whether player can disarm
        can_disarm = sum(self.current_disarm_caps[uid]) > 100.0
        if can_disarm:
            return base_prompt + "\n" + self.disarm_format_can_disarm
        else:
            return base_prompt + "\n" + self.disarm_format_cannot_disarm

    @staticmethod
    def _caps_description(caps: Caps) -> str:
        """Return '{"A0"=<upper_bounds0>, "A1"=<upper_bounds1>, ...}'."""
        return (
            "{"
            + ", ".join(f'"A{i}"={int(c)}' for i, c in enumerate(caps))
            + "}"
        )

    def _negotiate_disarm_caps(
        self,
        player: Agent,
    ) -> NegotiationResult:
        """Request updated caps for player and return parsed response.

        Returns:
            tuple[str, str, Caps]: (response, choice, new_caps)
        """
        uid = player.uid
        base_prompt = self.base_game.get_player_prompt(player.player_id) + "\n" + self._format_prompt(uid)

        def parse_func(resp: str) -> tuple[str, Caps]:
            return self._parse_disarm_caps(resp, uid)

        response, (choice, new_caps) = player.chat_with_retries(
            base_prompt=base_prompt,
            parse_func=parse_func,
        )
        return response, choice, new_caps

    def _parse_disarm_caps(
        self,
        response: str,
        uid: int,
    ) -> tuple[str, Caps]:
        """Parse the choice and disarmament caps from the agent's response.

        Returns:
            tuple[str, Caps]: (choice, new_caps) where choice is "disarm", "pass", or "end"
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

        if not isinstance(json_obj, dict):
            raise ValueError(
                "Parsed JSON must be an object with a 'choice' field"
            )

        # Validate choice field
        if "choice" not in json_obj:
            raise ValueError(
                "Missing required 'choice' field. Must include one of: "
                '{"choice": "disarm", ...}, {"choice": "pass"}, or {"choice": "end"}'
            )

        choice = json_obj["choice"]
        if not isinstance(choice, str):
            raise ValueError(
                f"'choice' field must be a string, got {type(choice).__name__}"
            )

        choice_lower = choice.lower()
        if choice_lower not in ("disarm", "pass", "end"):
            raise ValueError(
                f"'choice' field must be either 'disarm', 'pass', or 'end', got {choice!r}"
            )

        # For "pass" and "end", return current caps (no changes)
        if choice_lower in ("pass", "end"):
            # Validate no other keys present
            extra_keys = set(json_obj.keys()) - {"choice"}
            if extra_keys:
                raise ValueError(
                    f"When choice is '{choice_lower}', only the 'choice' field should be included. "
                    f"Found extra keys: {sorted(extra_keys)}"
                )
            return choice_lower, self.current_disarm_caps[uid]

        # For "disarm", check if player has room to disarm
        if sum(self.current_disarm_caps[uid]) <= 100.0:
            raise ValueError(
                'You chose "disarm" but your upper bounds already sum to 100, so you have no room to disarm further. '
                'You may only choose "pass" or "end".'
            )

        # Parse and validate the caps
        n = self.base_game.num_actions
        got_keys = set(json_obj.keys()) - {"choice"}
        missing = set(f"A{i}" for i in range(n)) - got_keys
        extra = got_keys - set(f"A{i}" for i in range(n))
        if extra:
            raise ValueError(f"Action key mismatch. Extra: {sorted(extra)}")
        if missing:
            raise ValueError(f"Action key mismatch. Missing: {sorted(missing)}")

        new_caps: Caps = [0.0] * n
        old_caps = self.current_disarm_caps[uid]
        for act_str, cap in json_obj.items():
            if act_str == "choice":
                continue
            idx = int(act_str[1:])  # strip the leading 'A'
            if not 0 <= idx < n:
                raise ValueError(f"A{idx} does not exist as a valid action")
            if not isinstance(cap, (int, float)):
                raise ValueError(
                    f"Upper bound for {act_str} must be numeric, got {type(cap)!r}"
                )
            cap_value = float(cap)
            if not 0 <= cap_value <= 100:
                raise ValueError(f"Disarm upper bound {cap} out of range for A{idx}")
            if cap_value > old_caps[idx]:
                raise ValueError(
                    f"New upper bound {cap_value} of A{idx} greater than its old upper bound {old_caps[idx]}"
                )
            new_caps[idx] = cap_value

        # Rule: caps must sum to >= 100
        if sum(new_caps) < 100:
            raise ValueError(
                f"Sum of your proposed upper bounds is {sum(new_caps)}, but must be at least 100"
            )

        # Validate: if choice is "disarm", caps must have changed
        if sum(new_caps) >= sum(old_caps) - 0.1:
            raise ValueError(
                "You chose choice='disarm' but did not change any upper bounds. "
                "Either change at least one cap, or use choice='pass' or choice='end'."
            )

        return choice_lower, new_caps

    def _run_negotiations(
        self,
        players: Sequence[Agent],
    ) -> dict[int, NegotiationResult]:
        """Prompt all players for their disarmament choices and return results."""

        results = run_tasks(players, self._negotiate_disarm_caps)
        return {
            player.uid: result
            for player, result in zip(players, results, strict=True)
        }

    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """
        Run the disarmament negotiation rounds.

        Returns:
            A list of move sequences (one sequence per round played).
        """
        # Ensure we start with fresh caps for this specific match
        self.current_disarm_caps = {
            player.uid: [100.0 for _ in range(self.base_game.num_actions)]
            for player in players
        }

        self.uid_to_player_id = {player.uid: player.player_id for player in players}

        disarmament_records: list[list[dict[str, Any]]] = []
        matchup_moves = []

        # Note: If self.history persists across matches, ensure it is cleared here
        # or managed externally. For safety, we just track local moves for the return.

        for _ in range(self.num_rounds):
            
            # Prompt ALL players (including those with no room to disarm)
            negotiation_results = self._run_negotiations(players)

            proposed_caps: CapsByPlayer = {}
            player_choices: dict[int, str] = {}
            round_records: list[dict[str, Any]] = []

            # First pass: collect all choices and proposed caps
            for player in players:
                puid = player.uid
                disarm_rsp, choice, player_cap = negotiation_results[puid]

                player_choices[puid] = choice
                proposed_caps[puid] = player_cap

                round_records.append(
                    {
                        "player": puid,
                        "response": disarm_rsp,
                        "choice": choice,
                        "proposed_upper_bound": player_cap,
                    }
                )

            # Check for veto
            veto = any(choice == "end" for choice in player_choices.values())

            # Determine which caps to use (veto discards all proposed changes)
            if veto:
                caps_to_use = self.current_disarm_caps.copy()
            else:
                caps_to_use = proposed_caps

            # Check if negotiation continues (no veto and at least one "disarm")
            active_disarm = any(choice == "disarm" for choice in player_choices.values())
            negotiation_continue = (not veto) and active_disarm

            # Calculate termination reason for this round
            # We play the game in every iteration, simulating what would happen if disarmament ended here
            if not negotiation_continue:
                # Explicit stop condition
                if veto:
                    # Find which players vetoed
                    vetoers = [
                        f"Player {self.uid_to_player_id[puid]}"
                        for puid, choice in player_choices.items()
                        if choice == "end"
                    ]
                    termination_reason = f"{', '.join(vetoers)} vetoed by choosing to end the disarmament phase."
                else:
                    # No veto, but negotiation stopped
                    # Check if anyone has room to disarm
                    any_room_to_disarm = any(
                        sum(self.current_disarm_caps[player.uid]) > 100.0
                        for player in players
                    )
                    if not any_room_to_disarm:
                        # No one can disarm further (all at sum=100)
                        termination_reason = "No players can disarm further (all at sum=100)."
                    else:
                        # No veto, people can disarm, but no one chose to
                        # This means everyone must have chosen "pass"
                        assert all(choice == "pass" for choice in player_choices.values()), \
                            "Logic error: negotiation stopped without veto, but not everyone passed"
                        termination_reason = "Everyone passed - no active disarmament occurred."
            else:
                # No explicit stop, but we're simulating what would happen if continuation probability ended it
                termination_reason = "The disarmament phase came to an end by random chance."

            # Build formatted cap lines for all players
            player_cap_lines = {}
            for player in players:
                label = f"Player {player.player_id}"
                caps_desc = self._caps_description(caps_to_use[player.uid])
                player_cap_lines[player.uid] = f"\t{label}: {caps_desc}"

            # Second pass: build mechanism prompts for each player
            disarmament_mechanisms: list[str] = []
            for player in players:
                puid = player.uid
                # Extract just the caps description for "my_caps" (without the label)
                my_caps = self._caps_description(caps_to_use[puid])

                # Other players' caps are all other players' formatted lines
                other_players_lines = [
                    player_cap_lines[other.uid]
                    for other in players
                    if other.uid != puid
                ]
                other_players_caps = "\n".join(other_players_lines)

                disarmament_mechanisms.append(
                    self.disarmament_mechanism_prompt.format(
                        my_caps=my_caps,
                        other_players_caps=other_players_caps,
                        termination_reason=termination_reason,
                    )
                )

            moves = self.base_game.play(
                players=players, additional_info=disarmament_mechanisms
            )

            self.current_disarm_caps = caps_to_use

            # Update round_records with the actual caps used and termination reason
            for record in round_records:
                record["actual_upper_bound"] = caps_to_use[record["player"]]
                record["termination_reason"] = termination_reason

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

            if not negotiation_continue:
                break

        LOGGER.log_record(
            record=disarmament_records, file_name=self.record_file
        )

        # Clear matchup-specific state to catch any accidental reuse
        self.current_disarm_caps = {}
        self.uid_to_player_id = {}

        return matchup_moves
