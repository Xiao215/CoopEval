"""Shared helpers for llm_judge plotting scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from coopeval.config import JUDGE_OUTPUT_DIR

RAW_SUBDIR = "raw"
NORMALIZED_SUBDIR = "normalized"
DATASET_SUBDIR = "dataset"
FIGURES_SUBDIR = "figures"
REPORTS_SUBDIR = "reports"

NORMALIZED_JSON_FILENAME = "normalized.jsonl"
DATASET_SHARE_FILENAME = "taxonomy_by_game_mechanism_model_player.share_pct.csv"
DATASET_COUNTS_FILENAME = "taxonomy_by_game_mechanism_model_player.counts.csv"
MECHANISM_DIFFERENCES_FILENAME = "mechanism_label_differences.csv"


def validate_input_name(value: str) -> str:
    """Ensure the provided run name is a simple slug."""
    candidate = value.strip()
    if not candidate:
        raise ValueError("input_name cannot be empty.")
    if Path(candidate).name != candidate:
        raise ValueError("input_name must not include directory separators.")
    return candidate


def _ensure_exists(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path.resolve()


def _first_existing(description: str, candidates: Iterable[Path]) -> Path:
    tried: list[Path] = []
    for path in candidates:
        tried.append(path)
        if path.exists():
            return path.resolve()
    tried_list = "\n".join(f"- {p}" for p in tried)
    raise FileNotFoundError(f"Missing {description}. Looked in:\n{tried_list}")


def run_dir(input_name: str) -> Path:
    """Return the base outputs/judge/<input_name> directory."""
    return (JUDGE_OUTPUT_DIR / input_name).resolve()


def normalized_jsonl_path(input_name: str) -> Path:
    """Return normalized JSONL for a run."""
    return _ensure_exists(
        run_dir(input_name) / NORMALIZED_SUBDIR / NORMALIZED_JSON_FILENAME,
        f"normalized JSONL for '{input_name}'",
    )


def dataset_share_csv_path(input_name: str) -> Path:
    """Return canonical taxonomy share CSV for a run."""
    return _ensure_exists(
        run_dir(input_name) / DATASET_SUBDIR / DATASET_SHARE_FILENAME,
        f"dataset share CSV for '{input_name}'",
    )


def mechanism_differences_csv_path(input_name: str) -> Path:
    """Locate mechanism_label_differences.csv for a run."""
    candidates = [
        run_dir(input_name) / MECHANISM_DIFFERENCES_FILENAME,
        run_dir(input_name) / FIGURES_SUBDIR / MECHANISM_DIFFERENCES_FILENAME,
        JUDGE_OUTPUT_DIR
        / REPORTS_SUBDIR
        / input_name
        / MECHANISM_DIFFERENCES_FILENAME,
    ]
    return _first_existing(
        f"mechanism_label_differences.csv for '{input_name}'", candidates
    )


def prepare_figure_subdir(input_name: str, relative: str) -> Path:
    """Ensure outputs/judge/<run>/figures/<relative>/ exists and return it."""
    path = run_dir(input_name) / FIGURES_SUBDIR / relative
    path.mkdir(parents=True, exist_ok=True)
    return path
