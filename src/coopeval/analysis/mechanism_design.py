"""Reusable mechanism-design analysis helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Sequence, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from coopeval.games.base import Game
from coopeval.registry.game_registry import GAME_REGISTRY
from coopeval.script_utils.display_helper import (
    format_model_name,
    sort_agents,
    sort_games,
)
from coopeval.script_utils.result_loader import (
    should_skip_game_name,
)
from coopeval.utils.json_io import load_json
from coopeval.script_utils.colors import (
    HEATMAP_NEGATIVE_COLOR,
    HEATMAP_POSITIVE_COLOR,
    PLOT_SEPARATOR_COLOR,
)

DesignT = TypeVar("DesignT")
ContextT = TypeVar("ContextT")

EPSILON = 1e-9


@dataclass
class AgentMetrics:
    """Count how often designs make the target action satisfy each criterion."""

    total: int
    weak_dominance: int
    nash_equilibrium: int

    @property
    def weak_dominance_rate(self) -> float:
        """Return the weak-dominance rate for the target action."""
        if self.total == 0:
            return 0.0
        return self.weak_dominance / self.total

    @property
    def nash_equilibrium_rate(self) -> float:
        """Return the Nash equilibrium rate for the target action."""
        if self.total == 0:
            return 0.0
        return self.nash_equilibrium / self.total


@dataclass
class DesignAnalyzer(Generic[DesignT, ContextT]):
    """Callbacks describing how to load and score a mechanism design."""

    name: str
    design_filename: str
    item_noun_singular: str
    item_noun_plural: str
    build_designs: Callable[
        [Path, dict[str, Any], dict[str, Any], Game],
        Iterable[tuple[str, DesignT]],
    ]
    context_factory: Callable[[Game], ContextT]
    evaluate_design: Callable[[DesignT, ContextT], tuple[bool, bool]]
    configure_game: Callable[[Game], None] | None = None
    figure_subdir: str | None = None
    figure_stem: str | None = None
    figure_title_prefix: str | None = None


def find_design_dirs(root: Path, design_filename: str) -> list[Path]:
    """Return directories that include both config.json and the target design file."""

    matches: list[Path] = []
    for design_path in root.rglob(design_filename):
        run_dir = design_path.parent
        if (run_dir / "config.json").exists():
            matches.append(run_dir)
    return sorted(matches)


def format_count(amount: int, singular: str, plural: str) -> str:
    """Human-friendly helper (e.g., 1 design vs 2 designs)."""

    return f"{amount} {singular if amount == 1 else plural}"


def extract_mechanism_entries(
    result_dir: Path,
    analyzer: DesignAnalyzer[DesignT, ContextT],
    *,
    skip_games: Sequence[str] | None,
) -> tuple[dict[str, dict[str, list[DesignT]]], dict[str, Game]]:
    """Load all mechanism designs keyed by (game, agent)."""

    print(f"Scanned {result_dir} for {analyzer.item_noun_plural}.\n")

    design_index: dict[str, dict[str, list[DesignT]]] = defaultdict(
        lambda: defaultdict(list)
    )
    game_cache: dict[str, Game] = {}
    game_kwargs_cache: dict[str, dict[str, Any]] = {}

    for run_dir in find_design_dirs(result_dir, analyzer.design_filename):
        config_path = run_dir / "config.json"
        design_path = run_dir / analyzer.design_filename

        config = load_json(config_path)
        game_name = config["game"]["type"]
        if should_skip_game_name(game_name, skip_games):
            continue
        game_kwargs = config["game"].get("kwargs", {})

        if game_name not in game_cache:
            game_cache[game_name] = GAME_REGISTRY[game_name](**game_kwargs)
            game_kwargs_cache[game_name] = game_kwargs
            if analyzer.configure_game is not None:
                analyzer.configure_game(game_cache[game_name])
        elif game_kwargs_cache[game_name] != game_kwargs:
            raise ValueError(
                f"Conflicting kwargs for game {game_name}: "
                f"{game_kwargs_cache[game_name]} vs {game_kwargs}"
            )

        game = game_cache[game_name]
        payload = load_json(design_path)

        for agent_name, design in analyzer.build_designs(
            run_dir, config, payload, game
        ):
            design_index[game_name][agent_name].append(design)

    return design_index, game_cache


def collect_mechanism_designs(
    result_dirs: Sequence[Path],
    analyzer: DesignAnalyzer[DesignT, ContextT],
    *,
    skip_games: Sequence[str] | None,
) -> tuple[dict[str, dict[str, list[DesignT]]], dict[str, Game]]:
    """Load mechanism designs from multiple result directories."""

    design_index: dict[str, dict[str, list[DesignT]]] = defaultdict(
        lambda: defaultdict(list)
    )
    games: dict[str, Game] = {}

    for result_dir in result_dirs:
        batch_design_index, batch_games = extract_mechanism_entries(
            result_dir, analyzer, skip_games=skip_games
        )
        games.update(batch_games)
        for game_name, agent_map in batch_design_index.items():
            for agent_name, designs in agent_map.items():
                design_index[game_name][agent_name].extend(designs)

    return design_index, games


def compute_metrics(
    design_index: dict[str, dict[str, list[DesignT]]],
    games: dict[str, Game],
    analyzer: DesignAnalyzer[DesignT, ContextT],
) -> dict[str, dict[str, AgentMetrics]]:
    """Score dominance/stability for each agent's submitted designs."""

    metrics: dict[str, dict[str, AgentMetrics]] = {}
    for game_name, agent_map in sorted(design_index.items()):
        if game_name not in games:
            raise ValueError(f"Missing cached Game instance for {game_name}.")
        game = games[game_name]
        context = analyzer.context_factory(game)

        agent_metrics: dict[str, AgentMetrics] = {}
        for agent_name in sort_agents(agent_map):
            designs = agent_map[agent_name]
            weak_dominance_count = 0
            nash_equilibrium_count = 0
            for design in designs:
                weak_dominance, nash_equilibrium = analyzer.evaluate_design(
                    design, context
                )
                if weak_dominance:
                    weak_dominance_count += 1
                if nash_equilibrium:
                    nash_equilibrium_count += 1
            agent_metrics[agent_name] = AgentMetrics(
                total=len(designs),
                weak_dominance=weak_dominance_count,
                nash_equilibrium=nash_equilibrium_count,
            )
        metrics[game_name] = agent_metrics

    return metrics


def print_summary(
    design_index: dict[str, dict[str, list[DesignT]]],
    analyzer: DesignAnalyzer[DesignT, ContextT],
) -> None:
    """Pretty-print how many designs each agent submitted per game."""

    for game_name, agent_map in sorted(design_index.items()):
        total_items = sum(len(entries) for entries in agent_map.values())
        total_label = format_count(
            total_items,
            analyzer.item_noun_singular,
            analyzer.item_noun_plural,
        )
        print(
            f"Game: {game_name} ("
            f"{len(agent_map)} agents, "
            f"{total_label}"
            ")"
        )
        for agent_name in sort_agents(agent_map):
            entries = agent_map[agent_name]
            amount = format_count(
                len(entries),
                analyzer.item_noun_singular,
                analyzer.item_noun_plural,
            )
            print(f"  {agent_name}: {amount}")
        print()


def print_metric_table(
    metrics: dict[str, dict[str, AgentMetrics]],
    analyzer: DesignAnalyzer[DesignT, ContextT],
) -> None:
    """Print weak-dominance and Nash equilibrium rates in a readable table."""

    print(
        "Weak Dominance and Nash Equilibrium Rates "
        f"({analyzer.item_noun_plural}):\n"
    )
    for game_name, agent_map in sorted(metrics.items()):
        if not agent_map:
            continue
        print(f"{game_name}:")
        for agent_name in sort_agents(agent_map):
            stats = agent_map[agent_name]
            print(
                f"  {agent_name}: Weak Dominance "
                f"{stats.weak_dominance_rate:.1%} | Nash Equilibrium "
                f"{stats.nash_equilibrium_rate:.1%} | n={stats.total}"
            )
        print()


def plot_tiered_rates(
    metrics: dict[str, dict[str, AgentMetrics]],
    title_prefix: str = "",
) -> Figure:
    """Return a Matplotlib figure plotting Nash equilibrium vs weak dominance."""

    data = []
    for game_name, agents in metrics.items():
        for agent_name, stats in agents.items():
            n = stats.total
            p_nash = stats.nash_equilibrium_rate
            p_weak_dom = stats.weak_dominance_rate
            err_nash = (
                1.96 * np.sqrt((p_nash * (1 - p_nash)) / n) if n > 0 else 0
            )
            err_weak_dom = (
                1.96 * np.sqrt((p_weak_dom * (1 - p_weak_dom)) / n)
                if n > 0
                else 0
            )

            data.append(
                {
                    "Game": game_name,
                    "AgentKey": agent_name,
                    "Agent": format_model_name(agent_name),
                    "Nash Equilibrium (%)": p_nash * 100,
                    "Weak Dominance (%)": p_weak_dom * 100,
                    "Nash Equilibrium Err": err_nash * 100,
                    "Weak Dominance Err": err_weak_dom * 100,
                    "N": n,
                }
            )

    df = pd.DataFrame(data)

    games = sort_games(df["Game"].unique().tolist())
    num_games = len(games)

    fig, axes = plt.subplots(
        nrows=num_games, ncols=1, figsize=(10, 5 * num_games)
    )
    if num_games == 1:
        axes = [axes]

    for ax, game in zip(axes, games):
        game_df = df[df["Game"] == game].copy()
        ordered_agents = sort_agents(game_df["AgentKey"].unique().tolist())
        game_df["AgentKey"] = pd.Categorical(
            game_df["AgentKey"], categories=ordered_agents, ordered=True
        )
        game_df = game_df.sort_values("AgentKey")

        total_designs = int(game_df["N"].sum())
        if total_designs > 0:
            avg_nash = float(
                np.average(
                    game_df["Nash Equilibrium (%)"], weights=game_df["N"]
                )
            )
            avg_weak_dom = float(
                np.average(game_df["Weak Dominance (%)"], weights=game_df["N"])
            )
            avg_nash_err = (
                1.96
                * np.sqrt(
                    (avg_nash / 100) * (1 - (avg_nash / 100)) / total_designs
                )
                * 100
            )
            avg_weak_dom_err = (
                1.96
                * np.sqrt(
                    (avg_weak_dom / 100)
                    * (1 - (avg_weak_dom / 100))
                    / total_designs
                )
                * 100
            )
        else:
            avg_nash = 0.0
            avg_weak_dom = 0.0
            avg_nash_err = 0.0
            avg_weak_dom_err = 0.0

        avg_row = pd.DataFrame(
            [
                {
                    "Game": game,
                    "AgentKey": "__average__",
                    "Agent": "Average",
                    "Nash Equilibrium (%)": avg_nash,
                    "Weak Dominance (%)": avg_weak_dom,
                    "Nash Equilibrium Err": avg_nash_err,
                    "Weak Dominance Err": avg_weak_dom_err,
                    "N": total_designs,
                }
            ]
        )
        plot_df = pd.concat([avg_row, game_df], ignore_index=True)

        average_gap = 0.75
        x_positions = np.concatenate(
            ([0.0], np.arange(len(game_df), dtype=float) + 1.0 + average_gap)
        )
        bar_width = 0.35
        x_nash = x_positions - bar_width / 2
        x_weak_dom = x_positions + bar_width / 2

        bars_nash = ax.bar(
            x_nash,
            plot_df["Nash Equilibrium (%)"],
            width=bar_width,
            label="Nash Equilibrium",
            color=HEATMAP_NEGATIVE_COLOR,
            edgecolor=HEATMAP_NEGATIVE_COLOR,
        )
        bars_weak_dom = ax.bar(
            x_weak_dom,
            plot_df["Weak Dominance (%)"],
            width=bar_width,
            label="Weak Dominance",
            color=HEATMAP_POSITIVE_COLOR,
            edgecolor=HEATMAP_POSITIVE_COLOR,
        )

        nash_upper = np.minimum(
            plot_df["Nash Equilibrium Err"],
            100 - plot_df["Nash Equilibrium (%)"],
        )
        weak_dom_upper = np.minimum(
            plot_df["Weak Dominance Err"],
            100 - plot_df["Weak Dominance (%)"],
        )
        nash_lower = np.minimum(
            plot_df["Nash Equilibrium Err"], plot_df["Nash Equilibrium (%)"]
        )
        weak_dom_lower = np.minimum(
            plot_df["Weak Dominance Err"], plot_df["Weak Dominance (%)"]
        )

        ax.errorbar(
            x_nash,
            plot_df["Nash Equilibrium (%)"],
            yerr=[nash_lower, nash_upper],
            fmt="none",
            ecolor=HEATMAP_NEGATIVE_COLOR,
            capsize=4,
            elinewidth=1.5,
        )
        ax.errorbar(
            x_weak_dom,
            plot_df["Weak Dominance (%)"],
            yerr=[weak_dom_lower, weak_dom_upper],
            fmt="none",
            ecolor=HEATMAP_POSITIVE_COLOR,
            capsize=4,
            elinewidth=1.5,
        )

        ax.bar_label(
            bars_nash,
            labels=[f"{val:.1f}%" for val in plot_df["Nash Equilibrium (%)"]],
            padding=3,
            fontsize=8,
        )
        ax.bar_label(
            bars_weak_dom,
            labels=[f"{val:.1f}%" for val in plot_df["Weak Dominance (%)"]],
            padding=3,
            fontsize=8,
        )

        title = f"{title_prefix} in {game}" if title_prefix else game
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_ylabel("Frequency of Criterion (%)", fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(plot_df["Agent"], rotation=15, ha="right")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        if len(x_positions) > 1:
            separator_x = (x_positions[0] + x_positions[1]) / 2
            ax.axvline(
                separator_x,
                color=PLOT_SEPARATOR_COLOR,
                linestyle=":",
                linewidth=1,
            )

        if ax == axes[0]:
            ax.legend(loc="upper right")

    fig.tight_layout()
    return fig
