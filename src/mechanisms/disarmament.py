"""Repeated-game mechanism that lets agents negotiate probability caps."""

import json
import re
from typing import Any, Sequence, override

from src.agents.agent_manager import Agent
from src.games.base import Game, Move
from src.logger_manager import LOGGER
from src.mechanisms.base import RepetitiveMechanism
from src.mechanisms.prompts import (DISARM_FORMAT_CAN_DISARM,
                                    DISARM_FORMAT_CANNOT_DISARM,
                                    DISARM_PROMPT_BASE,
                                    DISARMAMENT_MECHANISM_PROMPT)
from src.utils.concurrency import run_tasks

Caps = list[float]
CapsByPlayer = dict[Agent, Caps]

class Disarmament(RepetitiveMechanism):
    """
    Disarmament mechanism that allows for multiple rounds of disarming before playing the base game under disarmament restrictions.
    """

    _NO_ROOM_MESSAGE = "No room for further disarmament, so the same upper bounds remain."

    def __init__(
        self,
        base_game: Game,
        *,
        num_rounds: int,
        discount: float,
        tournament_workers: int = 1,
    ) -> None:
        super().__init__(
            base_game, num_rounds, discount, tournament_workers=tournament_workers
        )
        self.disarmament_mechanism_prompt = DISARMAMENT_MECHANISM_PROMPT

    def _format_prompt(
        self,
        player: Agent,
        caps: CapsByPlayer,
    ) -> str:
        """
        Build the filled prompt:
        - caps_by_agent: {agent_name: [cap_A0, cap_A1, ...]} (ints 0..100)
        - player_id: identifier for the agent whose 'my_caps' will be shown
        - discount: continuation probability (integer percent)
        """

        if player not in caps:
            raise KeyError(
                f"Upper bounds have not been initialized for player {player.player_id}"
            )

        self_caps_description = self._caps_description(
            caps[player]
        )

        other_player_labels = {
            other_player: f"Player {other_player.player_id}"
            for other_player in caps.keys()
            if other_player != player
        }

        opp_lines = []
        for other_player, other_player_label in other_player_labels.items():
            other_player_caps = self._caps_description(
                caps[other_player]
            )
            opp_lines.append(f"{other_player_label}: {other_player_caps}")
        other_players_caps_block = "\n".join(opp_lines)

        # Inject both the player's caps and everyone else's latest proposals plus the continuation probability.
        base_prompt = DISARM_PROMPT_BASE.format(
            my_caps=self_caps_description,
            other_players_caps=other_players_caps_block,
            discount=round(self.discount * 100),
        )

        # Force the right response schema depending on whether this player still has slack to disarm.
        can_disarm = sum(caps[player]) > 100.0
        if can_disarm:
            return base_prompt + DISARM_FORMAT_CAN_DISARM
        else:
            return base_prompt + DISARM_FORMAT_CANNOT_DISARM

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
        caps: CapsByPlayer,
    ) -> tuple[str, str, Caps]:
        """Request updated caps for player and return parsed response.

        Returns:
            tuple[str, str, Caps]: (response, choice, new_caps)
        """
        base_prompt = (
            self.base_game.get_player_prompt(player.player_id)
            + "\n"
            + self._format_prompt(player, caps)
        )

        def parse_func(resp: str) -> tuple[str, Caps]:
            return self._parse_disarm_caps(resp, player, caps)
        _, trace_id, (choice, new_caps) = player.chat_with_retries(
            base_prompt=base_prompt,
            parse_func=parse_func,
        )
        return trace_id, choice, new_caps

    def _parse_disarm_caps(
        self,
        response: str,
        player: Agent,
        caps: CapsByPlayer,
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
        json_obj = json.loads(json_str)

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

        # For "pass" and "end", the player's bounds remain unchanged.
        if choice_lower in ("pass", "end"):
            extra_keys = set(json_obj.keys()) - {"choice"}
            if extra_keys:
                raise ValueError(
                    f"When choice is '{choice_lower}', only the 'choice' field should be included. "
                    f"Found extra keys: {sorted(extra_keys)}"
                )
            return choice_lower, caps[player]

        # Disarm requests only make sense if the player still has probability mass above 100.
        if sum(caps[player]) <= 100.0:
            raise ValueError(
                'You chose "disarm" but your upper bounds already sum to 100, so you have no room to disarm further. '
                'You may only choose "pass" or "end".'
            )

        # Otherwise parse the proposal and ensure it touches every action key.
        n = self.base_game.num_actions
        got_keys = set(json_obj.keys()) - {"choice"}
        missing = set(f"A{i}" for i in range(n)) - got_keys
        extra = got_keys - set(f"A{i}" for i in range(n))
        if extra:
            raise ValueError(f"Action key mismatch. Extra: {sorted(extra)}")
        if missing:
            raise ValueError(f"Action key mismatch. Missing: {sorted(missing)}")

        new_caps: Caps = [0.0] * n
        old_caps = caps[player]
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
        caps: CapsByPlayer,
    ) -> dict[Agent, tuple[str, str, Caps]]:
        """Prompt all players for their disarmament choices and return results."""

        def negotiate_with_caps(player: Agent) -> tuple[str, str, Caps]:
            return self._negotiate_disarm_caps(player, caps)

        results = run_tasks(players, negotiate_with_caps)
        return {
            player: result
            for player, result in zip(players, results, strict=True)
        }

    @override
    def _play_matchup(self, players: Sequence[Agent]) -> list[list[Move]]:
        """
        Run the disarmament negotiation rounds.

        Returns:
            A list of move sequences (one sequence per round played).
        """
        # Each matchup keeps its own probability caps so parallel tournaments don't bleed state.
        current_disarm_caps = {
            player: [100.0 for _ in range(self.base_game.num_actions)]
            for player in players
        }
        matchup_history = self.History(self.base_game.action_class)

        disarmament_records: list[list[dict[str, Any]]] = []
        matchup_moves = []

        for _ in range(1, self.num_rounds + 1):
            # Always solicit responses from every player so the transcript reflects unanimous consent (or lack thereof).
            negotiation_results = self._run_negotiations(players, current_disarm_caps)

            proposed_caps: CapsByPlayer = {}
            player_choices = {}
            disarming_phase_records: list[dict[str, Any]] = []

            # First pass: capture each player's choice plus the full trace for auditing.
            for player in players:
                trace_id, choice, player_cap = negotiation_results[player]

                player_choices[player] = choice
                proposed_caps[player] = player_cap

                disarming_phase_records.append(
                    {
                        "player": player,
                        "trace_id": trace_id,
                        "choice": choice,
                        "proposed_upper_bound": player_cap,
                    }
                )

            # Any explicit "end" immediately halts negotiation.
            veto = any(choice == "end" for choice in player_choices.values())

            # Veto keeps the previous caps; otherwise apply the fresh proposals.
            if veto:
                caps_to_use = current_disarm_caps.copy()
            else:
                caps_to_use = proposed_caps

            # Continue only if at least one player wants to disarm and nobody vetoed.
            active_disarm = any(choice == "disarm" for choice in player_choices.values())
            negotiation_continue = (not veto) and active_disarm

            # Emit a human-readable reason for why the phase ended (or why we simulated an end this round).
            if not negotiation_continue:
                if veto:
                    vetoers = [
                        f"Player {player.player_id}"
                        for player, choice in player_choices.items()
                        if choice == "end"
                    ]
                    termination_reason = f"{', '.join(vetoers)} vetoed by choosing to end the disarmament phase."
                else:
                    any_room_to_disarm = any(
                        sum(current_disarm_caps[player]) > 100.0
                        for player in players
                    )
                    if not any_room_to_disarm:
                        termination_reason = "No players can disarm further (all at sum=100)."
                    else:
                        assert all(choice == "pass" for choice in player_choices.values()), \
                            "Logic error: negotiation stopped without veto, but not everyone passed"
                        termination_reason = "Everyone passed - no active disarmament occurred."
            else:
                termination_reason = "The disarmament phase came to an end by random chance."

            # Precompute pretty versions of the caps so prompt construction below stays simple.
            player_cap_lines = {}
            for player in players:
                label = f"Player {player.player_id}"
                caps_desc = self._caps_description(caps_to_use[player])
                player_cap_lines[player] = f"{label}: {caps_desc}"

            # Second pass: provide each player with their own caps plus everyone else's summary.
            disarmament_mechanisms: list[str] = []
            for player in players:
                # Extract just the caps description for "my_caps" (without the label)
                my_caps = self._caps_description(caps_to_use[player])

                # Other players' caps are all other players' formatted lines
                other_players_lines = [
                    player_cap_lines[other]
                    for other in players
                    if other != player
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

            current_disarm_caps = caps_to_use

            # Update round_records with the actual caps used and termination reason
            for disarm_info in disarming_phase_records:
                disarm_info["actual_upper_bound"] = caps_to_use[
                    disarm_info["player"]
                ]
                disarm_info["termination_reason"] = termination_reason

            disarmament_records.append(
                [
                    {"disarm_info": disarm_info, "move": move}
                    for disarm_info, move in zip(disarming_phase_records, moves)
                ]
            )

            matchup_history.append(moves)
            matchup_moves.append(moves)

            if not negotiation_continue:
                break

        LOGGER.log_record(
            record=disarmament_records, file_name=self.record_file
        )

        return matchup_moves
