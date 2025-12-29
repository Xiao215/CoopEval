import random
import re
from typing import Sequence
import json

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.games.base import Game
from src.mechanisms.base import RepetitiveMechanism

random.seed(42)


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

        self.disarm_prompt = """
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

        self.game_prompt = """
        Additional Information:
        A "cap" is the maximum probability (in %) with which you may choose an action in the next round.
        From previous round of negotiation, you agree to have a cap of:
        {my_caps}

        Now you need to propose a new probability distribution over actions subjected to your current cap limits.
        """

    def _format_prompt(
        self,
        player_name: str,
        caps_by_agent: dict[str, list[float]],
    ) -> str:
        """
        Build the filled prompt:
        - caps_by_agent: {agent_name: [cap_A0, cap_A1, ...]} (ints 0..100)
        - player_name: the agent whose 'my_caps' will be shown
        - discount: continuation probability (integer percent)
        """

        my_caps_line = self._caps_to_line(caps_by_agent[player_name])

        opp_lines = []
        for name in sorted(a for a in caps_by_agent.keys() if a != player_name):
            opp_lines.append(f"\t{name}: {self._caps_to_line(caps_by_agent[name])}")
        opponents_caps_block = "\n".join(opp_lines)

        print(f'my_caps_line: {my_caps_line}')
        print(f'opponents_caps_block: {opponents_caps_block}')
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
        old_caps: list[float],
        caps_by_agent: dict[str, list[float]],
        *,
        max_retries: int = 5,
    ) -> tuple[list[float], bool]:
        """
        Ask the LLM for disarm caps and parse with automatic retries.
        Returns: (new_caps, changed, last_response). On total failure, returns (old_caps, False, last_response)
        and prints a warning.
        """
        base_prompt = self._format_prompt(player.name, caps_by_agent)
        response = ""
        error_reason = ""

        # include initial + retries
        for attempt in range(max_retries + 1):
            if attempt == 0:
                prompt = base_prompt
            else:
                prompt = self._build_retry_prompt(base_prompt, response, error_reason)

            response = self.base_game.prompt_player(player, output_instruction=prompt)

            try:
                new_caps, changed = self._parse_disarm_caps(response, old_caps)
                return new_caps, changed
            except ValueError as e:
                error_reason = str(e)
                print(
                    f"Attempt {attempt + 1} of {player.name} to parse disarm caps failed: {error_reason} from response {response!r}"
                )

        print(
            f"Warning: Failed to parse disarm caps after {1 + max_retries} attempts. "
            f"Last error: {error_reason}. Last response: {response!r}"
        )
        return old_caps[:], False

    @staticmethod
    def _build_retry_prompt(
        base_prompt: str, bad_response: str, error_reason: str
    ) -> str:
        """Restate the prompt, show prior answer and the error reason, then ask for regeneration."""
        br = bad_response.replace("\n", " ")[:500]
        return (
            f"{base_prompt}\n\n"
            f"Your previous response was:\n{br}\n\n"
            f"That response is INVALID because: {error_reason}\n\n"
            f"Please give the action cap again!"
        )

    def _parse_disarm_caps(
        self,
        response: str,
        old_caps: list[float],
    ) -> tuple[list[float], bool]:
        """Parse the disarmament probabilities (new caps) from the agent's response."""
        n = len(old_caps)
        matches = re.findall(r"\{.*?\}", response, re.DOTALL)
        if not matches:
            raise ValueError(f"No JSON object found in the response {response!r}")
        json_str = matches[-1]

        try:
            json_obj = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e

        got_keys = set(json_obj.keys())
        missing = set(f"A{i}" for i in range(n)) - got_keys
        if missing:
            raise ValueError(f"Action key mismatch. Missing: {sorted(missing)}")

        new_caps = [0.0] * n
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
        disarmed_cap = {
            player.name: [100.0 for _ in range(self.base_game.num_actions)]
            for player in players
        }
        for _ in range(self.num_rounds):
            new_disarmed_cap = {}
            negotiation_continue = False
            additional_info = []

            for player in players:
                if sum(disarmed_cap[player.name]) > 100.0:
                    new_cap, changed = self._negotiate_disarm_caps(
                        player=player,
                        old_caps=disarmed_cap[player.name],
                        caps_by_agent=disarmed_cap,
                    )
                    negotiation_continue |= changed
                    new_disarmed_cap[player.name] = new_cap
                else:
                    new_disarmed_cap[player.name] = disarmed_cap[player.name]
                capping_limitation = self._caps_to_line(new_disarmed_cap[player.name])
                additional_info.append(
                    self.game_prompt.format(my_caps=capping_limitation)
                )

            moves = self.base_game.play(
                players=players, additional_info=additional_info
            )
            payoffs.add_profile({move.name: move.points for move in moves})

            disarmed_cap = new_disarmed_cap
            if not negotiation_continue:
                break
