from src.games.base import Move
from tests.fakes.general_fakes import FakeAction

class FakeMove(Move):
    """A fake move class for testing purposes."""

    def __init__(
        self, uid: int, points: float, action: FakeAction = FakeAction.HOLD
    ) -> None:
        super().__init__(
            uid=uid,
            player_name=f"fake-agent-{uid}",
            action=action,
            points=points,
            response="",
        )
