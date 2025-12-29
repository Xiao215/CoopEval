"""Helpers and test doubles for population payoff scenarios."""

from src.agents.agent_manager import Agent
from src.games.base import Action, Move


class MockAction(Action):
    """Minimal action enum for population payoff tests."""

    HOLD = "H"
    PASS = "P"


def make_move(uid: int, points: float, action: MockAction = MockAction.HOLD) -> Move:
    """Build a ``Move`` instance without relying on concrete game implementations."""
    return Move(
        uid=uid,
        player_name=f"agent-{uid}",
        action=action,
        points=points,
        response="",
    )


class MockAgent(Agent):
    """Minimal stand-in for ``Agent`` with the attributes used in tests."""

    def __init__(self, uid: int, model_type: str="Unset") -> None:
        self.uid = uid
        self.model_type = model_type

    def chat(self, messages: str) -> str:
        return f"mock response to {messages}"

    @property
    def name(self) -> str:
        return f"MockAgent-{self.uid}"
