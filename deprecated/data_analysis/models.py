from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .io import RunPaths


@dataclass
class MoveRecord:
    """Normalized record of a single agent move in any mechanism."""

    agent: str
    action: str
    points: float
    response: str
    round_index: int | None = None
    matchup_index: int | None = None
    mediator: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunData:
    """Parsed artefacts for a single run directory."""

    run: RunPaths
    config: Dict[str, Any]
    game: str
    mechanism: str
    agents: List[str]
    moves: List[MoveRecord]
    matchups: Dict[str, List[MoveRecord]]
    rounds: List[List[List[MoveRecord]]]
    expected_payoffs: Dict[str, float]
    mechanism_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return self.run.root

