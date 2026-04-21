"""Abstract game framework and utilities shared by all CoopEval environments."""

import asyncio
import random
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Self, Sequence, cast, override

from coopeval.agents.agent_manager import Agent
from coopeval.utils.json_io import extract_json_object_from_end


class Action(Enum):
    """Canonical set of CoopEval actions representing player moves. Provides helpers to convert between names, tokens, and indices."""

    def to_token(self) -> str:
        """Convert the action to a token (eg, A1) starting from A0 for LLM parsing."""
        idx = list(type(self)).index(self)
        return f"A{idx}"

    @classmethod
    def from_token(cls, token: str) -> Self:
        """Parse an action from a token like "A0" or "A1"."""
        try:
            idx = int(token.lstrip("A"))
            action = list(cls)[idx]
        except Exception as exp:
            raise ValueError(f"Unknown action token {token!r}") from exp
        return action

    @classmethod
    def from_index(cls, index: int) -> Self:
        """Get action from its index."""
        try:
            action = list(cls)[index]
        except Exception as exp:
            raise ValueError(f"Unknown action index {index!r}") from exp
        return action

    @classmethod
    def from_str(cls, name: str) -> "Action":
        """Convert a string like 'HEADS' or 'MatchingPenniesAction.HEADS' to an Enum."""
        member_name = name.split(".")[-1]
        return cls[member_name]

    @property
    def is_mediator(self) -> bool:
        """Check if this specific action is the Mediator action."""
        return self.name == "MEDIATOR"

    @classmethod
    def game_actions(cls) -> list[Self]:
        """Return all playable moves (excluding the Mediator action)."""
        return [act for act in cls if not act.is_mediator]

    def is_cooperate_action(self) -> bool:
        """Return True if this action represents cooperation for the game."""
        raise NotImplementedError(
            "This method should be implemented by specific game action enums."
        )


@dataclass
class Move:
    """Represents one player's realized move along with its payoff and metadata. Keeps lightweight references for logging without copying agents."""

    player: Agent
    action: Action
    points: float
    trace_id: str
    mediated: bool = False

    def serialize(self) -> dict[str, Any]:
        """Convert the Move to a dictionary, mostly for logging and record purpose."""
        # Build dict manually so we never deepcopy network clients tucked inside Agent objects.
        d = {
            "player": self.player.name,
            "action": str(self.action),
            "points": self.points,
            "trace_id": self.trace_id,
        }
        if self.mediated:
            d["mediated"] = True
        return d


class Game(ABC):
    """Abstract base for CoopEval games that defines the shared prompt/action plumbing. Handles player IO, mixed strategies, and action orchestration."""

    def __init__(
        self,
        prompt: str,
        action_class: type[Action],
        *,
        num_players: int,
        is_symmetric: bool = True,
    ) -> None:
        """Initialize shared game prompt, action metadata, and player count."""
        self.prompt = prompt
        self.num_players = num_players
        self.number_to_position = {
            1: "first",
            2: "second",
            3: "third",
            4: "fourth",
        }
        self.is_symmetric = is_symmetric
        self.action_class: type[Action] = action_class
        self.num_base_actions = len(self.action_class)
        self.num_actions = len(self.action_class)
        self.default_output_instruction = textwrap.dedent("""
        Instruction:
        - Choose a probability distribution over the provided actions each round.
        - Output must contain a valid JSON object at the end.
        - Keys must be the action names exactly as given.
        - Values must be percentage points given in integers.
        - The values must sum to exactly 100.

        Format requirement:
        Return exactly one JSON object, for example:
        {"A0": <INT>, "A1": <INT>, ...}
        """)

    def add_mediator_action(self) -> None:
        """Dynamically replace action_cls with a version containing MEDIATOR."""
        # Avoid recomputing the mediator and rebuild the aciton class, if already added.
        if any(action.is_mediator for action in self.action_class):
            return
        old_action_class = self.action_class
        members = {action.name: action.value for action in old_action_class}
        members["MEDIATOR"] = "MEDIATOR"

        new_enum = Enum(
            self.action_class.__name__,
            members,
            type=Action,
        )

        # Preserve custom helpers (e.g., is_cooperate_action) defined on the original enum.
        if hasattr(old_action_class, "is_cooperate_action"):
            setattr(new_enum, "is_cooperate_action", old_action_class.is_cooperate_action)  # type: ignore[attr-defined]

        self.action_class = cast(type[Action], new_enum)
        self.num_actions = len(self.action_class)

    def get_player_prompt(self, player_id: int) -> str:
        """Return the base game prompt from a specific player's perspective."""
        return (
            self.prompt
            + f"\nIn case player identification becomes relevant, you are playing in the position of Player {player_id} in this game.\n"
        )

    @abstractmethod
    async def play(
        self,
        additional_info: list[str] | str,
        players: Sequence[Agent],
        action_map: Callable[
            [dict[Agent, Action]], dict[Agent, Action]
        ] = lambda x: x,
    ) -> list[Move]:
        """Play one game instance and return the resulting moves."""
        raise NotImplementedError

    @abstractmethod
    def get_action_self_payoff(self, action: Action) -> float:
        """Return the payoff for taking the same action as all other players."""
        raise NotImplementedError

    @abstractmethod
    def get_actions_payoff(self, actions: Sequence[Action]) -> Sequence[float]:
        """Return player payoffs for a complete joint action profile."""
        raise NotImplementedError

    async def prompt_player_mix_probs(
        self,
        player: Agent,
        extra_info: str | None = None,
        output_instruction: str | None = None,
    ) -> tuple[str, dict[int, float]]:
        """
        Given the mechanism's additional info and the base game prompt,
        format the full prompt and query the player.

        Returns the player's raw response.
        """
        prompt = self.get_player_prompt(player.player_id)

        if extra_info:
            prompt += extra_info

        if output_instruction is None:
            output_instruction = self.default_output_instruction
        prompt += "\n" + output_instruction

        _response, trace_id, mix_probs = await player.chat_with_retries(
            prompt, self._parse_mixed_probs
        )
        return trace_id, mix_probs

    def _parse_mixed_probs(
        self,
        response: str,
    ) -> dict[int, float]:
        """
        Parse mixed strategy pairs like '<A0=60>|<A1=25>|<A2=15>'.
        Rules:
        - integers only
        - each in [0,100]
        - sum exactly 100
        """
        json_obj = extract_json_object_from_end(response)

        result: dict[int, float] = {}
        total = 0
        expected_keys = {f"A{idx}" for idx in range(self.num_actions)}
        actual_keys = set(json_obj)

        if actual_keys != expected_keys:
            raise ValueError(
                f"Expected keys {sorted(expected_keys)}, got {sorted(actual_keys)}"
            )

        for key, value in json_obj.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"Value for {key} must be an integer, got {value!r}"
                )
            if not 0 <= value <= 100:
                raise ValueError(
                    f"Value for {key} must be between 0 and 100, got {value}"
                )
            result[int(key[1:])] = value
            total += value

        if total != 100:
            raise ValueError(f"Probabilities must sum to 100 (got {total}).")

        return result

    @staticmethod
    def _choose_from_mix_strategy(probs: dict[int, float]) -> int:
        """Sample an action index from an integer percentage distribution."""
        keys = list(probs.keys())
        weights = list(probs.values())
        return random.choices(keys, weights=weights, k=1)[0]

    async def _collect_actions(
        self,
        players: Sequence[Agent],
        extra_info: Sequence[str],
    ) -> dict[Agent, tuple[Action, str]]:
        """Prompt all players concurrently and return sampled actions with trace IDs."""
        if len(players) != len(extra_info):
            raise ValueError(
                f"Count mismatch: {len(players)} vs {len(extra_info)}."
            )

        async def query(player: Agent, extra_info: str) -> tuple[int, str]:
            """Prompt one player and sample an action index from their mixed strategy."""
            trace_id, mix_probs = await self.prompt_player_mix_probs(
                player, extra_info=extra_info
            )
            action_idx = self._choose_from_mix_strategy(mix_probs)
            return action_idx, trace_id

        tasks = [query(p, info) for p, info in zip(players, extra_info)]
        results = await asyncio.gather(*tasks)

        return {
            player: (
                self.action_class.from_index(action_idx),
                trace_id,
            )
            for player, (action_idx, trace_id) in zip(players, results)
        }

    def _apply_action_map(
        self,
        players_decision: dict[Agent, tuple[Action, str]],
        action_map: Callable[[dict[Agent, Action]], dict[Agent, Action]],
    ) -> dict[Agent, tuple[Action, str, bool]]:
        """Apply a mechanism action mapping while preserving trace IDs."""
        original_actions = {
            player: decision[0] for player, decision in players_decision.items()
        }
        mapped_actions = action_map(dict(original_actions))
        if set(mapped_actions.keys()) != set(players_decision.keys()):
            raise ValueError(
                "action_map must return actions for the same player set it received."
            )
        updated: dict[Agent, tuple[Action, str, bool]] = {}
        for player, (action, trace_id) in players_decision.items():
            updated[player] = (
                mapped_actions[player],
                trace_id,
                action.is_mediator,
            )
        return updated

class GridGame(Game):
    """Symmetric 2x2 matrix-form game that embeds payoff grids into the base CoopEval framework. Generates prompts directly from the payoff matrix."""

    def __init__(
        self,
        payoff_matrix: Mapping[str, Sequence[float]],
        action_class: type[Action],
        *,
        num_players: int,
        is_symmetric: bool,
    ) -> None:
        """Initialize a symmetric payoff-matrix game and build its prompt."""
        if not is_symmetric:
            raise ValueError(
                "GridGame currently only supports symmetric games."
            )
        self.action_class = action_class
        self.raw_payoff_matrix = payoff_matrix
        self.payoff_matrix = self._parse_payoff_matrix(payoff_matrix)
        # Bake the processed payoff matrix into the prompt so each grid game prints the correct payoffs automatically.
        actions_block = "\n".join(
            [f"- {act.to_token()}" for act in action_class]
        )
        prompt = textwrap.dedent("""
        Setup:
        You are playing a decision-making game with another player.
        Your objective is to maximize your total points received in the game described in length below.

        Actions available to each player:
        {actions_block}

        Basic game rules:
        1. You and the other player each choose a probability for each action, simultaneously.
        2. After both decisions are locked in, the final action will be drawn from the probability distributions.
        3. Both players receive the points specified in the payoff description below.

        Payoff description:
        {payoff_description}
        """)

        super().__init__(
            prompt.format(
                actions_block=actions_block,
                payoff_description=self._payoff_description(),
            ),
            action_class,
            num_players=num_players,
            is_symmetric=is_symmetric,
        )

    def _parse_payoff_matrix(
        self,
        raw_payoff: Mapping[str, Sequence[float]],
    ) -> dict[
        tuple[Action, Action],
        tuple[float, float],
    ]:
        """Convert raw string-keyed payoffs into action-keyed payoff entries."""
        payoffs = {}
        for key, (p1, p2) in raw_payoff.items():
            a1 = self.action_class(key[0])
            a2 = self.action_class(key[1])
            payoffs[(a1, a2)] = (p1, p2)
        return payoffs

    def _payoff_description(self) -> str:
        """Format the payoff matrix as prompt text for players."""
        lines = []
        for (a, b), (pts_a, pts_b) in self.payoff_matrix.items():
            lines.append(
                f"\t- If you choose {a.to_token()} and the other player chooses {b.to_token()}: "
                f"you get {pts_a} points, the other player gets {pts_b} points."
            )
        return "\n".join(lines)

    @override
    def add_mediator_action(self) -> None:
        """Add the mediator action and rebuild the parsed payoff matrix."""
        super().add_mediator_action()
        self.payoff_matrix = self._parse_payoff_matrix(self.raw_payoff_matrix)

    def get_action_self_payoff(self, action: Action) -> float:
        """Return the diagonal payoff for one action in this matrix game."""
        return self.payoff_matrix[(action, action)][0]

    def get_actions_payoff(
        self, actions: Sequence[Action]
    ) -> tuple[float, float]:
        """Return both players' payoffs for a two-action profile."""
        return self.payoff_matrix[(actions[0], actions[1])]
