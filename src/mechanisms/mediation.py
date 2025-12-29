import random
import re
from typing import Callable, Sequence

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.games.base import Game
from src.mechanisms.base import Mechanism
from src.registry.agent_registry import create_agent

random.seed(42)


class Mediation(Mechanism):
    """
    Disarmament mechanism that allows for multiple rounds of the same game.
    """

    def __init__(
        self,
        base_game: Game,
        designer_cfg: dict,
    ) -> None:
        super().__init__(base_game)
        self.designer = create_agent(designer_cfg)

        self.mediator_design_prompt = """
        Instruction:
        You are tasked with designing a **mediator agent** for this game.

        - A mediator is an agent that suggests actions to players.
        - Each player may choose to **delegate** their move to the mediator or act independently.
        - The mediator observes number of players delegating to the mediator and then recommends the **same action** to all delegating players.
        - The mediator's objective is to maximize the payoff of the players who delegate to the mediator.

        Output Format:
        Return exactly **one valid Python dictionary** in a single line:
        {{1: <Action>, ..., {num_players}: <Action>}}

        - Keys: the number of players delegating (from 1 to {num_players}).
        - Values: the action the mediator will recommend (e.g., "A0", "A1", ...).
        - Ensure the dictionary is syntactically valid in Python.
        """

        self.game_prompt = """
        Additional Information:
        On top of the original game instructions, you have the option to delegate your move to a mediator agent.
        If you choose to delegate, the mediator will play an action for you based on how many players have delegated to it.
        You can also choose to act independently.

        Here is what the mediator would do for the players that delegate to it:
        {mediator_description}

        Consider A{additional_action_id} as an addtional action "Delegate to Mediator". Your final mixed strategy should include probability for all actions A0, A1, ..., A{additional_action_id}.
        """

    def _design_mediator(
        self, player: Agent, *, max_retries: int = 5
    ) -> dict[int, str]:
        """
        Design the mediator agent by prompting the designer.
        """
        base_prompt = self.mediator_design_prompt.format(
            num_players=self.base_game.num_players,
        )
        response = ""
        error_reason = ""

        # include initial + retries
        for attempt in range(max_retries + 1):
            if attempt == 0:
                prompt = base_prompt
            else:
                prompt = self._build_retry_prompt(base_prompt, response, error_reason)

            response = self.base_game.prompt_player(player, output_instruction=prompt)
            print(response)
            try:
                mediator = self._parse_mediator(response)
                print(f"Successfully designed mediator: {mediator}")
                return mediator
            except ValueError as e:
                error_reason = str(e)
                print(
                    f"Attempt {attempt + 1} of {player.name} to design mediator failed: "
                    "{error_reason} from response {response!r}"
                )
        raise ValueError(
            f"Failed to design mediator after {1 + max_retries} attempts. "
            f"Last error: {error_reason}. Last response: {response!r}"
        )

    @staticmethod
    def _build_retry_prompt(
        base_prompt: str, bad_response: str, error_reason: str
    ) -> str:
        """Restate the prompt, show prior response and ask for regeneration."""
        br = bad_response.replace("\n", " ")[:500]
        return (
            f"{base_prompt}\n\n"
            f"Your previous response was:\n{br}\n\n"
            f"That response is INVALID because: {error_reason}\n\n"
            f"Please give the action cap again!"
        )

    def _parse_mediator(self, response: str) -> dict[int, str]:
        """
        Parse the mediator design from the response.
        Expecting a Python dictionary in string format.
        """
        pairs = re.findall(r'(\d+)\s*:\s*(?:"|\'|)?A(\d+)(?:"|\'|)?', response)

        mediator = {}
        for k, v in pairs:
            if int(k) < 1 or int(k) > self.base_game.num_players:
                raise ValueError(
                    f"Invalid player number {k} for the pair {k}: A{v}, "
                    f"must be between 1 and {self.base_game.num_players}."
                )
            if not 0 <= int(v) < self.base_game.num_actions:
                raise ValueError(
                    f"Invalid action A{v} for the pair {k}: A{v}, "
                    f"must be one of {[f'A{a}' for a in range(self.base_game.num_actions)]}."
                )
            mediator[int(k)] = f"A{v}"
        if len(mediator) != self.base_game.num_players:
            raise ValueError(
                "There are missing cases in the mediator design, "
                f"you need to have cases for all number of players "
                f"from 1 to {self.base_game.num_players}."
            )
        return mediator

    def _mediator_description(self, mediator: dict[int, str]) -> str:
        """Format the prompt for the mediator agent."""
        lines = []
        for num_delegating, action in mediator.items():
            lines.append(
                f"\t• If {num_delegating} player(s) delegate to the mediator, "
                f"it will recommend action {action}."
            )
        return "\n".join(lines)

    def _play_matchup(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> None:
        mediator = self._design_mediator(self.designer)
        mediator_description = self._mediator_description(mediator)
        additional_info = self.game_prompt.format(
            mediator_description=mediator_description,
            additional_action_id=self.base_game.num_actions,
        )
        moves = self.base_game.play(
            players=players,
            additional_info=additional_info,
            action_map=self.mediator_mapping(mediator),
        )
        payoffs.add_profile({move.name: move.points for move in moves})

    def mediator_mapping(self, mediator: dict[int, str]) -> Callable:
        """
        Given the original actions and the mediator design, return the final actions
        after applying the mediator's recommendations.
        """

        def apply_mediation(actions_str: dict[str, str]) -> dict[str, str]:
            actions = {}
            num_delegating = sum(1 for a in actions_str.values() if a == "D")
            if num_delegating == 0:
                return actions_str
            recommended_action = mediator[num_delegating]
            for player_name, action_str in actions_str.items():
                if action_str == "D":
                    actions[player_name] = recommended_action
                else:
                    actions[player_name] = action_str
            return actions

        return apply_mediation
