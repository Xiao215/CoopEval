#!/usr/bin/env python3
"""Discover and plot evolutionary degradation cases from CoopEval runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from coopeval.analysis.evolutionary_degradation import (
    AnalyzedExperiment,
    CandidateExperiment,
    DegradedAgent,
    ExperimentData,
    analyze_tournament_data,
    candidate_from_analysis,
    degradation_summary_rows,
)
from coopeval.config import FIGURE_DIR
from coopeval.script_utils.display_helper import format_model_name, sort_agents
from coopeval.script_utils.result_loader import (
    DEFAULT_SKIP_GAMES,
    TournamentData,
    iter_tournament_data,
)
from coopeval.utils.json_io import clean_path
from coopeval.visualization.analysis_utils import NormalizeScore
from coopeval.script_utils.colors import MODEL_COLOR_PALETTE, hex_to_rgba
from coopeval.script_utils.figure_exports import save_matplotlib_figure

EVO_DEGRADATION_ARTIFACTS = (
    "agent_average_payoff.json",
    "replicator_dynamics_fitness.json",
    "population_history.json",
    "matchup_payoffs.json",
)


@dataclass(frozen=True)
class AgentDisplayContext:
    """Display names and highlight set for plot rendering."""

    agent_types: list[str]
    short_names: dict[str, str]
    degraded_names: set[str]


@dataclass(frozen=True)
class FitnessAxisBounds:
    """Fitness axis range and broken-axis decisions."""

    y_min: float
    y_max: float
    y_range: float
    margin: float
    data_min: float
    data_max: float
    use_top_break: bool
    use_bottom_break: bool


def build_agent_display_context(
    fitness_trajectories: dict[str, list[float]],
    degraded_agents: list[DegradedAgent],
) -> AgentDisplayContext:
    """Build reusable model display metadata for plot rendering."""

    agent_types = sort_agents(list(fitness_trajectories))
    return AgentDisplayContext(
        agent_types=agent_types,
        short_names={agent: format_model_name(agent) for agent in agent_types},
        degraded_names={agent.agent_name for agent in degraded_agents},
    )


def compute_fitness_axis_bounds(
    exp_data: ExperimentData,
    fitness_trajectories: dict[str, list[float]],
    *,
    gap_threshold: float = 0.15,
) -> FitnessAxisBounds:
    """Compute y-axis limits and whether a broken axis is useful."""

    normalizer = NormalizeScore(exp_data.game, exp_data.game_config)
    y_min = normalizer.ne_payoff
    y_max = normalizer.coop_payoff
    y_range = y_max - y_min
    margin = y_range * 0.05
    all_vals = [
        value for traj in fitness_trajectories.values() for value in traj
    ]
    data_min = min(all_vals)
    data_max = max(all_vals)
    return FitnessAxisBounds(
        y_min=y_min,
        y_max=y_max,
        y_range=y_range,
        margin=margin,
        data_min=data_min,
        data_max=data_max,
        use_top_break=(y_max - data_max) / y_range > gap_threshold,
        use_bottom_break=(data_min - y_min) / y_range > gap_threshold,
    )


def create_population_subplot(
    ax: Axes,
    population_history: list[dict[str, float]],
    agent_types: list[str],
    degraded_agent_names: set[str],
    short_names: dict[str, str],
) -> None:
    """Create a stacked area chart for population evolution."""

    n_steps = len(population_history)
    n_agents = len(agent_types)
    mat = np.zeros((n_steps, n_agents))

    for timestep, pop_dict in enumerate(population_history):
        for agent_idx, agent in enumerate(agent_types):
            mat[timestep, agent_idx] = pop_dict[agent]

    sums = mat.sum(axis=1)
    if not np.allclose(sums, 1, rtol=1e-2):
        print(
            "Warning: distributions sum to "
            f"[{sums.min():.4f}, {sums.max():.4f}]"
        )

    cumsum = np.cumsum(mat, axis=1)
    base = np.hstack([np.zeros((n_steps, 1)), cumsum])
    x_values = np.arange(n_steps)

    colors = []
    labels = []
    for agent in agent_types:
        base_color = MODEL_COLOR_PALETTE.get(short_names[agent], "#888888")
        colors.append(
            base_color
            if agent in degraded_agent_names
            else hex_to_rgba(base_color, alpha=0.5)
        )
        labels.append(
            f"{short_names[agent]} *"
            if agent in degraded_agent_names
            else short_names[agent]
        )

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
    ax.set_title(
        "Population Evolution Over Time", fontsize=12, fontweight="bold"
    )
    ax.set_xlim(0, max(n_steps - 1, 1))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, linestyle="--")


def _plot_fitness_lines(
    ax: Axes,
    fitness_trajectories: dict[str, list[float]],
    agent_types: list[str],
    degraded_names: set[str],
    short_names: dict[str, str],
) -> None:
    """Draw fitness trajectory lines on an axes."""

    for agent in agent_types:
        fitness_vals = fitness_trajectories[agent]
        timesteps = np.arange(len(fitness_vals))
        short = short_names[agent]
        base_color = MODEL_COLOR_PALETTE.get(short, "#888888")

        if agent in degraded_names:
            ax.plot(
                timesteps,
                fitness_vals,
                color=base_color,
                linewidth=2.5,
                label=f"{short} *",
                alpha=1.0,
            )
            ax.scatter(
                [0],
                [fitness_vals[0]],
                color=base_color,
                s=50,
                zorder=5,
                marker="o",
            )
            ax.scatter(
                [len(fitness_vals) - 1],
                [fitness_vals[-1]],
                color=base_color,
                s=50,
                zorder=5,
                marker="s",
            )
        else:
            ax.plot(
                timesteps,
                fitness_vals,
                color=base_color,
                linewidth=1.0,
                linestyle="--",
                alpha=1.0,
                label=short,
            )


def create_fitness_subplot(
    ax: Axes,
    fitness_trajectories: dict[str, list[float]],
    agent_types: list[str],
    degraded_agents: list[DegradedAgent],
    short_names: dict[str, str],
    game: str,
    game_config: dict[str, Any],
) -> None:
    """Create a line plot for fitness trajectories."""

    degraded_names = {agent.agent_name for agent in degraded_agents}
    _plot_fitness_lines(
        ax,
        fitness_trajectories,
        agent_types,
        degraded_names,
        short_names,
    )

    normalizer = NormalizeScore(game, game_config)
    y_range = normalizer.coop_payoff - normalizer.ne_payoff
    margin = y_range * 0.1
    ax.set_ylim(normalizer.ne_payoff - margin, normalizer.coop_payoff + margin)

    n_steps = max(len(values) for values in fitness_trajectories.values())
    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("Fitness Value", fontsize=11)
    ax.set_title(
        "Fitness Trajectories Over Time", fontsize=12, fontweight="bold"
    )
    ax.set_xlim(0, max(n_steps - 1, 1))
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=9)


def plot_evo_degradation(
    exp_data: ExperimentData,
    degraded_agents: list[DegradedAgent],
    fitness_trajectories: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Create the appendix-style population and fitness degradation plot."""

    display = build_agent_display_context(fitness_trajectories, degraded_agents)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"hspace": 0.35}
    )

    create_population_subplot(
        ax1,
        exp_data.population_history,
        display.agent_types,
        display.degraded_names,
        display.short_names,
    )
    create_fitness_subplot(
        ax2,
        fitness_trajectories,
        display.agent_types,
        degraded_agents,
        display.short_names,
        exp_data.game,
        exp_data.game_config,
    )

    plt.tight_layout(pad=0.5)
    renderer = fig.canvas.get_renderer()
    bbox = ax1.get_tightbbox(renderer)
    y_top = bbox.y1 / (fig.get_figheight() * fig.dpi)
    fig.text(
        0.5,
        y_top + 0.015,
        f"{exp_data.mechanism} - {exp_data.game}",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        transform=fig.transFigure,
    )

    saved_paths = save_matplotlib_figure(
        fig,
        output_path.with_suffix(""),
        [output_path.suffix.lstrip(".")],
        dpi=150,
        bbox_inches="tight",
        format_subdirs=False,
    )
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


def _add_break_marks(ax_above: Axes, ax_below: Axes, d: float = 0.015) -> None:
    """Add diagonal break indicators between two broken y axes."""

    kwargs = dict(
        transform=ax_above.transAxes, color="k", clip_on=False, lw=1.5
    )
    ax_above.plot((-d, +d), (-d, +d), **kwargs)
    ax_above.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_below.transAxes)
    ax_below.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_below.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


def create_mainbody_axes(
    bounds: FitnessAxisBounds,
) -> tuple[plt.Figure, Axes, Axes, Axes | None, Axes | None]:
    """Create population and fitness axes for the main-body plot."""

    population_height = 4
    fitness_height = 4
    stub_height = 1
    total_fit_height = (
        fitness_height
        + (stub_height if bounds.use_top_break else 0)
        + (stub_height if bounds.use_bottom_break else 0)
    )
    fig = plt.figure(figsize=(7, 10))
    outer_gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[population_height, total_fit_height],
        hspace=0.10,
    )
    population_ax = fig.add_subplot(outer_gs[0])

    fit_row_heights = []
    if bounds.use_top_break:
        fit_row_heights.append(stub_height)
    fit_row_heights.append(fitness_height)
    if bounds.use_bottom_break:
        fit_row_heights.append(stub_height)

    inner_gs = GridSpecFromSubplotSpec(
        len(fit_row_heights),
        1,
        subplot_spec=outer_gs[1],
        height_ratios=fit_row_heights,
        hspace=0.05,
    )

    row_idx = 0
    top_stub_ax: Axes | None = None
    bottom_stub_ax: Axes | None = None
    if bounds.use_top_break:
        top_stub_ax = fig.add_subplot(inner_gs[row_idx], sharex=population_ax)
        row_idx += 1

    main_fitness_ax: Axes = fig.add_subplot(
        inner_gs[row_idx], sharex=population_ax
    )
    row_idx += 1

    if bounds.use_bottom_break:
        bottom_stub_ax = fig.add_subplot(
            inner_gs[row_idx], sharex=population_ax
        )

    return fig, population_ax, main_fitness_ax, top_stub_ax, bottom_stub_ax


def configure_mainbody_population_axis(ax: Axes, font_size: int) -> None:
    """Apply main-body specific population-axis formatting."""

    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Population Share", fontsize=font_size)
    ax.set_title(
        "Population Evolution Over Time",
        fontsize=font_size + 1,
        fontweight="bold",
    )


def configure_mainbody_fitness_axes(
    *,
    main_ax: Axes,
    top_stub_ax: Axes | None,
    bottom_stub_ax: Axes | None,
    bounds: FitnessAxisBounds,
    font_size: int,
) -> None:
    """Apply y-limits, labels, and break marks to main-body fitness axes."""

    if bounds.use_bottom_break or bounds.use_top_break:
        main_ax.set_ylim(
            bounds.data_min - bounds.margin,
            bounds.data_max + bounds.margin,
        )

        if bounds.use_top_break and top_stub_ax is not None:
            stub_margin = bounds.y_range * 0.03
            top_stub_ax.set_ylim(
                bounds.y_max - stub_margin,
                bounds.y_max + stub_margin,
            )
            top_stub_ax.set_yticks([bounds.y_max])
            top_stub_ax.spines["bottom"].set_visible(False)
            main_ax.spines["top"].set_visible(False)
            top_stub_ax.tick_params(bottom=False)
            _add_break_marks(top_stub_ax, main_ax)

        if bounds.use_bottom_break and bottom_stub_ax is not None:
            stub_margin = bounds.y_range * 0.03
            bottom_stub_ax.set_ylim(
                bounds.y_min - stub_margin,
                bounds.y_min + stub_margin,
            )
            bottom_stub_ax.set_yticks([bounds.y_min])
            main_ax.spines["bottom"].set_visible(False)
            bottom_stub_ax.spines["top"].set_visible(False)
            main_ax.tick_params(bottom=False)
            _add_break_marks(main_ax, bottom_stub_ax)
    else:
        full_margin = bounds.y_range * 0.1
        main_ax.set_ylim(
            bounds.y_min - full_margin,
            bounds.y_max + full_margin,
        )

    main_ax.set_ylabel("Fitness Value", fontsize=font_size)
    topmost_ax = (
        top_stub_ax
        if bounds.use_top_break and top_stub_ax is not None
        else main_ax
    )
    topmost_ax.set_title(
        "Fitness Trajectories Over Time",
        fontsize=font_size + 1,
        fontweight="bold",
    )

    bottommost_ax = (
        bottom_stub_ax
        if bounds.use_bottom_break and bottom_stub_ax is not None
        else main_ax
    )
    bottommost_ax.set_xlabel("Time Step", fontsize=font_size)
    bottommost_ax.tick_params(labelbottom=True)
    main_ax.legend(loc="lower right", fontsize=10)


def plot_evo_degradation_mainbody(
    exp_data: ExperimentData,
    degraded_agents: list[DegradedAgent],
    fitness_trajectories: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Create the compact main-body variant of the degradation plot."""

    display = build_agent_display_context(fitness_trajectories, degraded_agents)
    bounds = compute_fitness_axis_bounds(exp_data, fitness_trajectories)
    font_size = 13
    fig, ax1, ax2_main, ax2_top_stub, ax2_bot_stub = create_mainbody_axes(
        bounds
    )

    create_population_subplot(
        ax1,
        exp_data.population_history,
        display.agent_types,
        display.degraded_names,
        display.short_names,
    )
    configure_mainbody_population_axis(ax1, font_size)

    all_fit_axes = [
        ax for ax in [ax2_top_stub, ax2_main, ax2_bot_stub] if ax is not None
    ]
    n_steps = max(len(values) for values in fitness_trajectories.values())
    for ax in all_fit_axes:
        _plot_fitness_lines(
            ax,
            fitness_trajectories,
            display.agent_types,
            display.degraded_names,
            display.short_names,
        )
        ax.set_xlim(0, max(n_steps - 1, 1))
        ax.grid(alpha=0.3, linestyle="--")
        ax.tick_params(labelbottom=False)

    configure_mainbody_fitness_axes(
        main_ax=ax2_main,
        top_stub_ax=ax2_top_stub,
        bottom_stub_ax=ax2_bot_stub,
        bounds=bounds,
        font_size=font_size,
    )

    saved_paths = save_matplotlib_figure(
        fig,
        output_path.with_suffix(""),
        [output_path.suffix.lstrip(".")],
        dpi=150,
        bbox_inches="tight",
        format_subdirs=False,
    )
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


def discover_degraded_experiments(
    result_dirs: Iterable[Path],
    *,
    skip_games: Iterable[str] | None,
    min_initial_rank: float,
    max_final_pop: float,
) -> tuple[list[CandidateExperiment], int, int]:
    """Scan result directories and return ranked degradation candidates."""

    candidates = []
    total_experiments = 0
    total_with_rd = 0

    for result_dir in result_dirs:
        print(f"\nScanning {result_dir}...")
        all_data = list(
            iter_tournament_data(
                result_dir,
                artifacts=(),
                skip_games=skip_games,
            )
        )
        evo_data = list(
            iter_tournament_data(
                result_dir,
                artifacts=EVO_DEGRADATION_ARTIFACTS,
                skip_games=skip_games,
            )
        )
        total_experiments += len(all_data)

        for data in evo_data:
            if data.mechanism.lower() == "nomechanism":
                continue

            total_with_rd += 1
            try:
                analyzed = analyze_tournament_data(
                    data,
                    min_initial_rank=min_initial_rank,
                    max_final_pop=max_final_pop,
                )
            except (KeyError, RuntimeError, ValueError) as exc:
                print(
                    "  Warning: Failed to compute fitness for "
                    f"{data.path.name}: {exc}"
                )
                continue
            if analyzed is None:
                continue

            candidate = candidate_from_analysis(analyzed)
            candidates.append(candidate)
            print(
                "  Found degraded agents in "
                f"{data.path.name}: {candidate.num_degraded} agents, "
                f"max collapse={candidate.max_collapse_index:.3f}"
            )

    candidates.sort(
        key=lambda candidate: (
            -candidate.max_collapse_index,
            -candidate.num_degraded,
            -candidate.avg_collapse_index,
        )
    )
    return candidates, total_experiments, total_with_rd


def write_candidate_outputs(
    candidates: list[CandidateExperiment], output_dir: Path, top_n: int
) -> None:
    """Write the candidate path list and detail CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    top_candidates = candidates[:top_n]

    candidate_file = output_dir / "candidate_experiments.txt"
    with candidate_file.open("w", encoding="utf-8") as file_obj:
        for candidate in top_candidates:
            file_obj.write(f"{candidate.exp_path}\n")
    print(f"\nSaved candidate list to: {candidate_file}")

    csv_file = output_dir / "candidates_detailed.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "rank",
                "path",
                "game",
                "mechanism",
                "num_degraded",
                "max_collapse",
                "avg_collapse",
                "degraded_agents",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(top_candidates, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "path": str(candidate.exp_path),
                    "game": candidate.game,
                    "mechanism": candidate.mechanism,
                    "num_degraded": candidate.num_degraded,
                    "max_collapse": f"{candidate.max_collapse_index:.4f}",
                    "avg_collapse": f"{candidate.avg_collapse_index:.4f}",
                    "degraded_agents": "; ".join(
                        agent.agent_name for agent in candidate.degraded_agents
                    ),
                }
            )
    print(f"Saved detailed CSV to: {csv_file}")


def print_candidate_report(candidates: list[CandidateExperiment]) -> None:
    """Print ranked candidate details to stdout."""

    print(
        "\n" f"TOP {len(candidates)} CANDIDATE EXPERIMENTS WITH LLM DEGRADATION"
    )
    print("=" * 80)
    for rank, candidate in enumerate(candidates, start=1):
        print(f"\n{rank}. {candidate.exp_path}")
        print(f"   Game: {candidate.game}, Mechanism: {candidate.mechanism}")
        print(f"   Degraded agents: {candidate.num_degraded}")
        print(f"   Max collapse index: {candidate.max_collapse_index:.3f}")
        print(
            "   Agents: "
            + ", ".join(agent.agent_name for agent in candidate.degraded_agents)
        )


def run_discovery_phase(
    result_dirs: list[Path],
    output_dir: Path,
    *,
    skip_games: Iterable[str] | None,
    min_initial_rank: float,
    max_final_pop: float,
    top_n: int,
) -> None:
    """Discover experiments with degraded agents and save candidate files."""

    print("=" * 80)
    print("DISCOVERY & IDENTIFICATION")
    print("=" * 80)
    print(
        "Criteria: "
        f"initial rank >= {min_initial_rank * 100:.0f}%, "
        f"final pop < {max_final_pop * 100:.0f}%"
    )

    candidates, total_experiments, total_with_rd = (
        discover_degraded_experiments(
            result_dirs,
            skip_games=skip_games,
            min_initial_rank=min_initial_rank,
            max_final_pop=max_final_pop,
        )
    )

    print("\n" + "=" * 80)
    print("Discovery complete.")
    print(f"  Total experiments scanned: {total_experiments}")
    print(f"  Experiments with replicator dynamics: {total_with_rd}")
    print(f"  Candidates with degraded agents: {len(candidates)}")
    print("=" * 80)

    if not candidates:
        print("\nNo candidates found matching criteria.")
        return

    top_candidates = candidates[:top_n]
    print_candidate_report(top_candidates)
    write_candidate_outputs(candidates, output_dir, top_n)


def read_candidate_paths(candidate_file: Path) -> list[Path]:
    """Read candidate experiment directories from a newline-delimited file."""

    if not candidate_file.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {candidate_file}. "
            "Run the discover phase first."
        )
    return [
        Path(line.strip()).expanduser().resolve()
        for line in candidate_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def iter_candidate_tournament_data(
    candidate_paths: Iterable[Path],
) -> Iterable[TournamentData]:
    """Yield evo degradation tournament data for candidate path roots."""

    for exp_folder in candidate_paths:
        if not exp_folder.exists():
            raise FileNotFoundError(f"Candidate path not found: {exp_folder}")

        found_data = False
        for data in iter_tournament_data(
            exp_folder, artifacts=EVO_DEGRADATION_ARTIFACTS
        ):
            found_data = True
            yield data

        if not found_data:
            raise RuntimeError(
                "Candidate path is missing required evo degradation artifacts: "
                f"{exp_folder}"
            )


def analyze_candidate_data(
    data: TournamentData,
    *,
    min_initial_rank: float,
    max_final_pop: float,
) -> AnalyzedExperiment:
    """Analyze one candidate tournament data object."""

    analyzed = analyze_tournament_data(
        data,
        min_initial_rank=min_initial_rank,
        max_final_pop=max_final_pop,
    )
    if analyzed is None:
        raise ValueError(f"No degraded agents found in experiment: {data.path}")
    return analyzed


def plot_analyzed_experiment(
    analyzed: AnalyzedExperiment, output_dir: Path, candidate_index: int
) -> None:
    """Generate appendix and main-body plots for one analyzed experiment."""

    exp_data = analyzed.exp_data
    run_name = exp_data.exp_path.parent.name
    base_name = (
        f"{candidate_index:02d}_{exp_data.mechanism}_{exp_data.game}_{run_name}"
    )

    plot_evo_degradation(
        exp_data,
        analyzed.degraded_agents,
        analyzed.fitness_trajectories,
        output_dir / "candidate_plots_appendix" / f"{base_name}_appendix.png",
    )
    plot_evo_degradation_mainbody(
        exp_data,
        analyzed.degraded_agents,
        analyzed.fitness_trajectories,
        output_dir / "candidate_plots_mainbody" / f"{base_name}_mainbody.png",
    )


def write_degradation_summary(
    summary_rows: list[dict[str, Any]], output_dir: Path
) -> None:
    """Write the degradation summary CSV when rows are available."""

    if not summary_rows:
        return
    summary_file = output_dir / "degradation_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved summary to: {summary_file}")


def run_plot_phase(
    output_dir: Path,
    *,
    candidate_file: Path | None,
    min_initial_rank: float,
    max_final_pop: float,
) -> None:
    """Generate plots for experiments listed in a candidate file."""

    print("\n" + "=" * 80)
    print("VISUALIZATION")
    print("=" * 80)

    if candidate_file is None:
        candidate_file = output_dir / "candidate_experiments.txt"

    candidate_paths = read_candidate_paths(candidate_file)
    print(f"Loading {len(candidate_paths)} candidates from: {candidate_file}\n")

    summary_rows = []
    plot_count = 0
    candidate_data = iter_candidate_tournament_data(candidate_paths)
    for idx, data in enumerate(candidate_data, start=1):
        print(f"{idx}. Processing: {data.path.name}")
        analyzed = analyze_candidate_data(
            data,
            min_initial_rank=min_initial_rank,
            max_final_pop=max_final_pop,
        )
        plot_analyzed_experiment(analyzed, output_dir, idx)
        summary_rows.extend(degradation_summary_rows(analyzed))
        plot_count += 1

    write_degradation_summary(summary_rows, output_dir)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evolutionary degradation plotting."""

    parser = argparse.ArgumentParser(
        description="Discover and plot LLM evolutionary degradation patterns."
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
        default=FIGURE_DIR / "evo_dynamics",
        help="Output directory for plots and candidate summaries.",
    )
    parser.add_argument(
        "--phase",
        choices=["discover", "plot"],
        nargs="+",
        default=["discover", "plot"],
        help=(
            "Phases to run. Provide multiple phases, e.g. "
            "--phase discover plot."
        ),
    )
    parser.add_argument(
        "--candidate-list",
        type=clean_path,
        help="Path to candidate list file for the plot phase.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top candidates to keep from discovery.",
    )
    parser.add_argument(
        "--max-final-pop",
        type=float,
        default=0.10,
        help="Maximum final population share for degraded agents.",
    )
    parser.add_argument(
        "--min-initial-rank",
        type=float,
        default=0.5,
        help="Minimum initial rank percentile for degraded agents.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()

    phases = set(args.phase)

    if "discover" in phases:
        run_discovery_phase(
            args.tournament_result_dirs,
            args.output_dir,
            skip_games=args.skip_games,
            min_initial_rank=args.min_initial_rank,
            max_final_pop=args.max_final_pop,
            top_n=args.top_n,
        )

    if "plot" in phases:
        run_plot_phase(
            args.output_dir,
            candidate_file=args.candidate_list,
            min_initial_rank=args.min_initial_rank,
            max_final_pop=args.max_final_pop,
        )


if __name__ == "__main__":
    main()
