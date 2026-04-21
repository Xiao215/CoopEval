#!/usr/bin/env python3
"""Generate action-frequency figures from cleaned CoopEval runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from .strategic_behavior_common import stacked_barh_with_se

from coopeval.config import FIGURE_DIR
from coopeval.script_utils.display_helper import (
    build_action_color_map,
    build_action_display_names,
    clean_action,
    format_mechanism_name,
    format_model_name,
    sort_agents,
    sort_games,
    sort_mechanisms,
    strip_player_suffix,
)
from coopeval.script_utils.result_loader import (
    DEFAULT_SKIP_GAMES,
    ExperimentInfo,
    iter_action_nodes,
    iter_experiments,
    load_json_lines,
)
from coopeval.utils.json_io import load_json, clean_path
from coopeval.script_utils.figure_exports import save_matplotlib_figure

ActionFrequencyIndex = dict[str, dict[str, list[dict[str, Counter[str]]]]]


def analyze_action_frequencies(
    experiments: list[ExperimentInfo],
    *,
    consolidate_players: bool = True,
) -> ActionFrequencyIndex:
    """Extract per-run action counters grouped by mechanism, game, and player."""

    action_freq: ActionFrequencyIndex = {}

    for experiment in experiments:
        run_action_freq: defaultdict[str, Counter[str]] = defaultdict(Counter)

        for match in load_json_lines(experiment.path / "records.jsonl"):
            for move in iter_action_nodes(match):
                run_action_freq[
                    (
                        strip_player_suffix(move["player"])
                        if consolidate_players
                        else move["player"]
                    )
                ][move["action"]] += 1

        mechanism_runs = action_freq.setdefault(experiment.mechanism, {})
        game_runs = mechanism_runs.setdefault(experiment.game, [])
        game_runs.append(dict(run_action_freq))

    return action_freq


def plot_action_frequencies(
    action_freq: ActionFrequencyIndex,
    game_configs: dict[str, dict[str, Any]],
    *,
    games_evaluated: list[str],
    output_dir: Path,
) -> None:
    """Plot action distributions pooled over all models."""

    games = [game for game in games_evaluated if game in game_configs]
    if not games:
        print("No game configs found.")
        return

    n_games = len(games)
    max_mechs = max(
        len(
            sort_mechanisms(
                mechanism
                for mechanism, game_runs in action_freq.items()
                if game in game_runs
            )
        )
        for game in games
    )
    fig, axes = plt.subplots(
        1,
        n_games,
        figsize=(5 * n_games, max(4, max_mechs * 0.55 + 1.5)),
        squeeze=False,
    )
    axes = axes[0]
    fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.18)

    for col_idx, game in enumerate(games):
        ax = axes[col_idx]
        action_colors = build_action_color_map(game, game_configs[game])
        display_names = build_action_display_names(game, game_configs[game])
        ordered_actions = list(action_colors)
        mechs_present = sort_mechanisms(
            mechanism
            for mechanism, game_runs in action_freq.items()
            if game in game_runs
        )

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, len(mechs_present) - 0.5)
        ax.set_title(game, fontsize=11)
        ax.set_xlabel("action probabilities in %", fontsize=9)
        ax.invert_yaxis()

        for mechanism_idx, mechanism in enumerate(mechs_present):
            runs = action_freq[mechanism][game]
            run_pcts = {action: [] for action in ordered_actions}

            for run_data in runs:
                run_counts: Counter[str] = Counter()
                for action_counter in run_data.values():
                    for raw_action, count in action_counter.items():
                        run_counts[clean_action(raw_action)] += count
                total = sum(run_counts.values())
                if total == 0:
                    continue
                for action in ordered_actions:
                    run_pcts[action].append(
                        run_counts.get(action, 0) / total * 100
                    )

            stacked_barh_with_se(
                ax, mechanism_idx, run_pcts, ordered_actions, action_colors
            )

        ax.set_yticks(range(len(mechs_present)))
        ax.set_yticklabels(
            (
                [
                    format_mechanism_name(mechanism)
                    for mechanism in mechs_present
                ]
                if col_idx == 0
                else [""] * len(mechs_present)
            ),
            fontsize=8,
        )
        patches = [
            mpatches.Patch(
                color=action_colors[action],
                label=display_names.get(action, action),
            )
            for action in ordered_actions
        ]
        ax.legend(
            handles=patches,
            fontsize=7,
            loc="lower right",
            title="NE -> Coop",
            title_fontsize=7,
        )

    fig.suptitle(
        "Action Frequencies across Mechanisms", fontsize=13, fontweight="bold"
    )
    saved_paths = save_matplotlib_figure(
        fig,
        output_dir / "action_frequency" / "action_freq_all_games",
        ["png"],
        dpi=150,
        bbox_inches="tight",
        format_subdirs=False,
    )
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


def plot_action_frequencies_by_model(
    action_freq: ActionFrequencyIndex,
    game_configs: dict[str, dict[str, Any]],
    *,
    games_evaluated: list[str],
    output_dir: Path,
) -> None:
    """Plot action distributions for each model within each mechanism."""

    games = [game for game in games_evaluated if game in game_configs]
    if not games:
        print("No game configs found.")
        return

    all_models_set: set[str] = set()
    for mechanism in sort_mechanisms(action_freq):
        for game in games:
            for run_data in action_freq.get(mechanism, {}).get(game, []):
                all_models_set.update(run_data)
    sorted_models = sort_agents(list(all_models_set))
    n_models = len(sorted_models)

    max_mechs = max(
        len(
            sort_mechanisms(
                mechanism
                for mechanism, game_runs in action_freq.items()
                if game in game_runs
            )
        )
        for game in games
    )
    group_height = n_models + 1
    fig_height = max(6, max_mechs * group_height * 0.38 + 2)

    fig, axes = plt.subplots(
        1, len(games), figsize=(5 * len(games), fig_height), squeeze=False
    )
    axes = axes[0]
    fig.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.12)

    for col_idx, game in enumerate(games):
        ax = axes[col_idx]
        action_colors = build_action_color_map(game, game_configs[game])
        display_names = build_action_display_names(game, game_configs[game])
        ordered_actions = list(action_colors)
        mechs_present = sort_mechanisms(
            mechanism
            for mechanism, game_runs in action_freq.items()
            if game in game_runs
        )
        n_rows = len(mechs_present) * group_height

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.set_title(game, fontsize=11)
        ax.set_xlabel("action probabilities in %", fontsize=9)
        ax.invert_yaxis()

        ytick_pos, ytick_labels = [], []

        for group_idx, mechanism in enumerate(mechs_present):
            group_base = group_idx * group_height
            if col_idx == 0:
                group_mid = group_base + (n_models - 1) / 2.0
                ax.text(
                    -0.22,
                    group_mid,
                    format_mechanism_name(mechanism),
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    clip_on=False,
                )

            runs = action_freq[mechanism][game]
            for model_idx, player in enumerate(sorted_models):
                y = group_base + model_idx
                ytick_pos.append(y)
                ytick_labels.append(
                    format_model_name(player) if col_idx == 0 else ""
                )

                run_pcts = {action: [] for action in ordered_actions}
                for run_data in runs:
                    player_counts: Counter[str] = Counter()
                    for raw_action, count in run_data.get(player, {}).items():
                        player_counts[clean_action(raw_action)] += count
                    total = sum(player_counts.values())
                    if total == 0:
                        continue
                    for action in ordered_actions:
                        run_pcts[action].append(
                            player_counts.get(action, 0) / total * 100
                        )

                if not any(run_pcts[action] for action in ordered_actions):
                    ax.barh(y, 100, height=0.8, left=0, color="#e8e8e8")
                    continue

                stacked_barh_with_se(
                    ax, y, run_pcts, ordered_actions, action_colors
                )

        for group_idx in range(1, len(mechs_present)):
            ax.axhline(
                group_idx * group_height - 0.5,
                color="gray",
                linewidth=0.5,
                linestyle="--",
                alpha=0.5,
            )

        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_labels, fontsize=7.5)
        patches = [
            mpatches.Patch(
                color=action_colors[action],
                label=display_names.get(action, action),
            )
            for action in ordered_actions
        ]
        ax.legend(
            handles=patches,
            fontsize=7,
            loc="lower right",
            title="NE -> Coop",
            title_fontsize=7,
        )

    fig.suptitle(
        "Action Frequencies by Model across Mechanisms",
        fontsize=13,
        fontweight="bold",
    )
    saved_paths = save_matplotlib_figure(
        fig,
        output_dir / "action_frequency" / "action_freq_by_model",
        ["png"],
        dpi=150,
        bbox_inches="tight",
        format_subdirs=False,
    )
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for action-frequency plotting."""
    parser = argparse.ArgumentParser(
        description="Generate pooled and per-model action-frequency figures."
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
    parser.add_argument(
        "--keep-player-positions",
        action="store_true",
        help="Keep #P1/#P2 player suffixes instead of pooling seats.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    experiments = []
    for tournament_result_batch in args.tournament_result_dirs:
        for experiment in iter_experiments(
            tournament_result_batch, skip_games=args.skip_games
        ):
            experiments.append(experiment)
    if not experiments:
        raise RuntimeError(
            "No records.jsonl files matched the requested games."
        )

    game_configs = {}
    for experiment in experiments:
        game_configs.setdefault(
            experiment.game, load_json(experiment.path / "config.json")
        )
    games = sort_games({experiment.game for experiment in experiments})
    action_freq = analyze_action_frequencies(
        experiments, consolidate_players=not args.keep_player_positions
    )

    plot_action_frequencies(
        action_freq,
        game_configs,
        games_evaluated=games,
        output_dir=args.output_dir,
    )
    plot_action_frequencies_by_model(
        action_freq,
        game_configs,
        games_evaluated=games,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
