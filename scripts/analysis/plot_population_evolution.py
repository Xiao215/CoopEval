#!/usr/bin/env python3
"""Generate population-evolution figures from CoopEval replicator dynamics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.typing import ColorType

from coopeval.config import FIGURE_DIR
from coopeval.script_utils.colors import MODEL_COLOR_PALETTE
from coopeval.script_utils.display_helper import format_model_name, sort_agents
from coopeval.script_utils.figure_exports import save_matplotlib_figure
from coopeval.script_utils.result_loader import (
    DEFAULT_SKIP_GAMES,
    TournamentData,
    iter_tournament_data,
)
from coopeval.utils.json_io import clean_path

POPULATION_EVOLUTION_ARTIFACTS = ("population_history.json",)


def _slugify(value: str) -> str:
    """Return a filesystem-friendly lowercase identifier."""

    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "unknown"


def population_history_to_matrix(
    population_history: Sequence[dict[str, float]],
    agent_names: Sequence[str],
) -> np.ndarray:
    """Convert list-of-dicts population history to a dense matrix."""

    mat = np.zeros((len(population_history), len(agent_names)))
    for timestep, population in enumerate(population_history):
        for agent_idx, agent in enumerate(agent_names):
            mat[timestep, agent_idx] = population.get(agent, 0.0)
    return mat


def draw_population_evolution(
    ax: Axes,
    trajectory: np.ndarray,
    *,
    labels: Sequence[str],
    colors: Sequence[ColorType],
) -> None:
    """Draw a stacked area chart of population shares over time."""

    if trajectory.ndim != 2:
        raise ValueError(
            "Expected population trajectory with shape "
            "(time_steps, agents)."
        )

    steps, n_agents = trajectory.shape
    sums = trajectory.sum(axis=1)
    if not np.allclose(sums, 1, rtol=1e-3):
        print(
            "Warning: population distributions sum to "
            f"[{sums.min():.4f}, {sums.max():.4f}]"
        )

    cumulative = np.cumsum(trajectory, axis=1)
    base = np.hstack([np.zeros((steps, 1)), cumulative])
    x_values = np.arange(steps)

    for agent_idx in range(n_agents):
        ax.fill_between(
            x_values,
            base[:, agent_idx],
            base[:, agent_idx + 1],
            color=colors[agent_idx],
            label=labels[agent_idx],
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("Population Share", fontsize=11)
    ax.set_xlim(0, max(steps - 1, 1))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, linestyle="--")


def create_population_evolution_figure(
    population_history: Sequence[dict[str, float]],
    *,
    title: str,
) -> Figure:
    """Create a population-evolution figure for one experiment."""

    if not population_history:
        raise ValueError("population_history.json is empty.")

    agent_names = sort_agents(
        {agent for timestep in population_history for agent in timestep}
    )
    trajectory = population_history_to_matrix(population_history, agent_names)
    labels = [format_model_name(agent) for agent in agent_names]
    colors = [
        MODEL_COLOR_PALETTE.get(label, "#888888") for label in labels
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    draw_population_evolution(
        ax, trajectory, labels=labels, colors=colors
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    fig.tight_layout()
    return fig


def output_stem_for_experiment(data: TournamentData) -> Path:
    """Return a deterministic figure stem for a tournament experiment."""

    run_name = data.path.parent.name
    experiment_name = data.path.name
    return Path(
        f"{_slugify(run_name)}_"
        f"{_slugify(experiment_name)}_"
        "population_evolution"
    )


def plot_population_evolution(
    data: TournamentData,
    *,
    output_dir: Path,
) -> list[Path]:
    """Create and save the population-evolution plot for one experiment."""

    population_history = data.load_json("population_history.json")
    title = (
        "Population Evolution: "
        f"{data.mechanism} / {data.game} / {data.path.parent.name}"
    )
    fig = create_population_evolution_figure(
        population_history, title=title
    )
    saved_paths = save_matplotlib_figure(
        fig,
        output_dir
        / "population_evolution"
        / output_stem_for_experiment(data),
        ("png",),
        dpi=150,
        bbox_inches="tight",
        format_subdirs=False,
    )
    plt.close(fig)
    return saved_paths


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for population-evolution plotting."""

    parser = argparse.ArgumentParser(
        description="Generate population-evolution figures from CoopEval runs."
    )
    parser.add_argument(
        "--tournament_result_dirs",
        nargs="+",
        type=clean_path,
        required=True,
        help="Tournament result batch to scan.",
    )
    parser.add_argument(
        "--skip-games",
        nargs="*",
        default=DEFAULT_SKIP_GAMES,
        help=(
            "Games to skip before aggregation "
            "(default: %(default)s; pass with no values to include all)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=clean_path,
        default=FIGURE_DIR,
        help="Directory where figures are written.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    plot_count = 0

    for tournament_result_batch in args.tournament_result_dirs:
        for data in iter_tournament_data(
            tournament_result_batch,
            artifacts=POPULATION_EVOLUTION_ARTIFACTS,
            skip_games=args.skip_games,
        ):
            print(f"Plotting: {data.path}")
            saved_paths = plot_population_evolution(
                data, output_dir=args.output_dir
            )
            for path in saved_paths:
                print(f"Saved: {path}")
            plot_count += 1

    if plot_count == 0:
        raise RuntimeError(
            "No population_history.json files matched the requested games."
        )


if __name__ == "__main__":
    main()
