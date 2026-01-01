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

    def __init__(self, uid: int, model_type: str = "Unset") -> None:
        # We don't call super().__init__ to avoid needing real dependencies
        self.uid = uid
        self.model_type = model_type
        self._name = f"FakeAgent-{uid}"

    def chat(self, messages: Any) -> str:
        return f"fake response to {messages}"

    @property
    def name(self) -> str:
        return self._name

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
