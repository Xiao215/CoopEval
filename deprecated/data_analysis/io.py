from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class RunPaths:
    root: Path
    config_path: Path | None
    payoffs_path: Path | None
    history_paths: list[Path]


def find_runs(root: str | Path) -> list[RunPaths]:
    base = Path(root)
    runs: list[RunPaths] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        config_path = child / "config.json"
        payoffs_path = child / "payoffs.json"
        history_paths = sorted(child.glob("*.jsonl"))
        runs.append(
            RunPaths(
                root=child,
                config_path=config_path if config_path.exists() else None,
                payoffs_path=payoffs_path if payoffs_path.exists() else None,
                history_paths=history_paths,
            )
        )
    return runs


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str | Path) -> Iterable[Any]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_config(run: RunPaths) -> dict[str, Any] | None:
    return load_json(run.config_path) if run.config_path else None


def load_payoffs(run: RunPaths) -> dict[str, Any] | None:
    return load_json(run.payoffs_path) if run.payoffs_path else None


def load_histories(run: RunPaths) -> list[list[dict[str, Any]]]:
    histories: list[list[dict[str, Any]]] = []
    for hp in run.history_paths:
        # Each line may be a list of rounds (each round = list of move dicts)
        for entry in iter_jsonl(hp):
            if isinstance(entry, list) and entry and isinstance(entry[0], list):
                # entry is a list of rounds
                for round_moves in entry:
                    if isinstance(round_moves, list):
                        histories.append(round_moves)
            elif isinstance(entry, list):
                # entry is a single round
                histories.append(entry)
    return histories


