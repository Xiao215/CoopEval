"""Fakes and test doubles for population payoff scenarios."""

from typing import Any
from src.agents.agent_manager import Agent
from src.games.base import Action, Move


class FakeAction(Action):
    """Minimal action enum for population payoff tests."""

    HOLD = "H"
    PASS = "P"


class FakeAgent(Agent):
    """Minimal stand-in for ``Agent`` with attributes needed for payoff tracking."""

    def __init__(self, llm_config: dict | None = None) -> None:
        if llm_config is None:
            llm_config = {}
        if "model" not in llm_config:
            llm_config["model"] = "fake-agent"
        if "provider" not in llm_config:
            llm_config["provider"] = "TestInstance"
        agent_config = {"llm": llm_config, "type": "IOAgent"}

        super().__init__(agent_config)
    def chat(self, messages: Any) -> str:
        return f"fake response to {messages}"

    @property
    def name(self) -> str:
        return f"fake-agent-{self.uid}"

    def __lt__(self, other: "FakeAgent") -> bool:
        return self.uid < other.uid


def make_fake_move(
    uid: int, points: float, action: FakeAction = FakeAction.HOLD
) -> Move:
    """Build a ``Move`` instance without relying on concrete game implementations."""
    return Move(
        uid=uid,
        player_name=f"fake-agent-{uid}",
        action=action,
        points=points,
        response="",
    )
