"""Helpers for discovering and reading CoopEval experiment result folders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from coopeval.utils.json_io import iter_jsonl, load_json

DEFAULT_SKIP_GAMES = ("MatchingPennies", "StagHunt")


def should_skip_game_name(
    game_name: str, skip_games: Iterable[str] | None = None
) -> bool:
    """Return True if the provided game name appears in the skip list."""
    source = (
        DEFAULT_SKIP_GAMES
        if skip_games is None
        else tuple(
            value.strip() for value in skip_games if value and value.strip()
        )
    )
    if not source:
        return False
    return game_name.strip() in set(source)


def iter_action_nodes(
    payload: Any,
    *,
    require_trace_id: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield serialized action nodes from arbitrary nested record payloads."""

    if isinstance(payload, Mapping):
        if "player" in payload and "action" in payload:
            trace_id = payload.get("trace_id")
            if not require_trace_id or isinstance(trace_id, str):
                yield dict(payload)
        for value in payload.values():
            yield from iter_action_nodes(
                value, require_trace_id=require_trace_id
            )
        return

    if isinstance(payload, list):
        for value in payload:
            yield from iter_action_nodes(
                value, require_trace_id=require_trace_id
            )


def discover_experiment_subfolders(experiment_dir: Path) -> list[Path]:
    """Find experiment subfolders, excluding configs and slurm directories."""
    return [
        d
        for d in experiment_dir.iterdir()
        if d.is_dir() and d.name not in {"configs", "slurm"}
    ]


@dataclass(frozen=True)
class ExperimentInfo:
    """Minimal metadata describing a single experiment directory."""

    path: Path
    game: str
    mechanism: str


@dataclass(frozen=True)
class TournamentData:
    """Metadata and selected artifact paths for one tournament experiment."""

    path: Path
    config: dict[str, Any]
    game: str
    mechanism: str
    artifacts: dict[str, Path]

    def artifact(self, filename: str) -> Path:
        """Return the selected artifact path for a filename."""
        return self.artifacts[filename]

    def load_json(self, filename: str) -> Any:
        """Load a selected JSON artifact by filename."""
        return load_json(self.artifact(filename))


def _discover_experiment_dirs(root: Path) -> Iterator[Path]:
    """Yield directories under root that contain config.json."""

    resolved = root.resolve()
    for cfg in resolved.rglob("config.json"):
        yield cfg.parent


def find_artifact_paths(
    base_dir: Path,
    artifact_candidates: Mapping[str, Iterable[Path | str]],
    *,
    required: Iterable[str] = (),
) -> dict[str, Path | None]:
    """Return the first existing candidate path for each named artifact.

    Candidate paths may be absolute or relative to base_dir. Missing optional
    artifacts map to None; missing required artifacts raise FileNotFoundError
    with the searched paths.
    """

    root = base_dir.resolve()

    def candidate_path(value: Path | str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    searched_paths = {
        name: tuple(candidate_path(candidate) for candidate in candidates)
        for name, candidates in artifact_candidates.items()
    }
    artifact_paths = {
        name: next(
            (path.resolve() for path in candidates if path.exists()),
            None,
        )
        for name, candidates in searched_paths.items()
    }

    missing_required = [
        name for name in required if artifact_paths.get(name) is None
    ]
    if missing_required:
        searched = []
        for name in missing_required:
            searched.append(f"{name}:")
            searched.extend(
                f"- {path}" for path in searched_paths.get(name, ())
            )
        searched_text = "\n".join(searched)
        raise FileNotFoundError(
            "Could not locate required artifact(s) in "
            f"{root}. Looked in:\n{searched_text}"
        )

    return artifact_paths


def _select_artifact_paths(
    run_dir: Path,
    *,
    artifacts: Iterable[str],
) -> dict[str, Path] | None:
    """Return selected artifact paths, or None if a required file is missing."""

    selected_paths = find_artifact_paths(
        run_dir,
        {filename: (filename,) for filename in artifacts},
    )
    if any(path is None for path in selected_paths.values()):
        return None

    return {
        name: path
        for name, path in selected_paths.items()
        if path is not None
    }


def iter_tournament_data(
    root: Path,
    *,
    artifacts: Iterable[str],
    skip_games: Iterable[str] | None = None,
) -> Iterator[TournamentData]:
    """Yield tournament metadata plus requested artifact paths under root."""
    path = root.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise NotADirectoryError(path)

    requested_artifacts = tuple(artifacts)

    for run_dir in _discover_experiment_dirs(path):
        config = load_json(run_dir / "config.json")
        game_name = config["game"]["type"].strip()
        if should_skip_game_name(game_name, skip_games):
            continue

        artifact_paths = _select_artifact_paths(
            run_dir,
            artifacts=requested_artifacts,
        )
        if artifact_paths is None:
            continue

        mechanism = config["mechanism"]["type"].strip()
        yield TournamentData(
            path=run_dir,
            config=config,
            game=game_name,
            mechanism=mechanism,
            artifacts=artifact_paths,
        )


def iter_experiments(
    root: Path,
    *,
    skip_games: Iterable[str] | None = None,
) -> Iterator[ExperimentInfo]:
    """Yield ExperimentInfo entries for runs under root, applying skip filters."""

    for data in iter_tournament_data(
        root,
        artifacts=("records.jsonl",),
        skip_games=skip_games,
    ):
        yield ExperimentInfo(
            path=data.path,
            game=data.game,
            mechanism=data.mechanism,
        )


def load_json_lines(path: Path) -> list[Any]:
    """Load a newline-delimited JSON file."""
    return list(iter_jsonl(path))
