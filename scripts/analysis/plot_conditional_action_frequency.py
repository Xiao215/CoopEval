#!/usr/bin/env python3
"""Generate conditional action-frequency figures from cleaned CoopEval runs."""

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
    format_action_name,
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
    iter_experiments,
    load_json_lines,
)
from coopeval.utils.json_io import clean_path, load_json
from coopeval.script_utils.figure_exports import save_matplotlib_figure

ConditionalFrequencyIndex = dict[
    str, dict[str, list[dict[str, dict[tuple[str, ...], Counter[str]]]]]
]

REPEATED_ACTION_MECHANISMS = {
    "Repetition",
    "Reputation",
    "ReputationFirstOrder",
}


def clean_prior_label(prior_tuple: tuple[str, ...]) -> str:
    """Return a readable label for a tuple of previous co-player actions."""

    if all(action == "NoHistory" for action in prior_tuple):
        return "No History"
    return ", ".join(
        format_action_name(clean_action(action)) for action in prior_tuple
    )


def analyze_conditional_action_frequencies(
    experiments: list[ExperimentInfo],
    *,
    consolidate_players: bool = True,
) -> ConditionalFrequencyIndex:
    """Extract action counters conditioned on co-player previous actions."""

    cond_freq: ConditionalFrequencyIndex = {}

    for experiment in experiments:
        if experiment.mechanism not in REPEATED_ACTION_MECHANISMS:
            continue
        run_cond_freq: defaultdict[
            str, defaultdict[tuple[str, ...], Counter[str]]
        ] = defaultdict(lambda: defaultdict(Counter))

        for match in load_json_lines(experiment.path / "records.jsonl"):
            last_known_action: dict[str, str] = {}

            for round_moves in match:
                current_players = [move["player"] for move in round_moves]

                for move in round_moves:
                    player_raw = move["player"]
                    player = (
                        strip_player_suffix(player_raw)
                        if consolidate_players
                        else player_raw
                    )
                    opponents = [
                        candidate
                        for candidate in current_players
                        if candidate != player_raw
                    ]
                    prior_actions = tuple(
                        sorted(
                            last_known_action.get(opponent, "NoHistory")
                            for opponent in opponents
                        )
                    )
                    run_cond_freq[player][prior_actions][move["action"]] += 1

                for move in round_moves:
                    last_known_action[move["player"]] = move["action"]

        mechanism_runs = cond_freq.setdefault(experiment.mechanism, {})
        game_runs = mechanism_runs.setdefault(experiment.game, [])
        game_runs.append(
            {
                player: dict(prior_counts)
                for player, prior_counts in run_cond_freq.items()
            }
        )

    return cond_freq


def plot_conditional_actions(
    cond_freq: ConditionalFrequencyIndex,
    *,
    target_game: str,
    game_config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot action distributions conditioned on previous co-player actions."""

    action_colors = build_action_color_map(target_game, game_config)
    display_names = build_action_display_names(target_game, game_config)

    all_models_set: set[str] = set()
    all_priors_set: set[tuple[str, ...]] = set()
    all_actions_set: set[str] = set()

    mechanisms = sort_mechanisms(
        mechanism
        for mechanism, game_runs in cond_freq.items()
        if target_game in game_runs
    )
    for mechanism in mechanisms:
        for run_data in cond_freq.get(mechanism, {}).get(target_game, []):
            for player, prior_dict in run_data.items():
                all_models_set.add(player)
                for prior_tuple, action_counter in prior_dict.items():
                    all_priors_set.add(prior_tuple)
                    for raw_action in action_counter:
                        all_actions_set.add(clean_action(raw_action))

    if not all_models_set:
        print(f"No conditional data found for game: {target_game}")
        return

    sorted_models = sort_agents(list(all_models_set))
    n_models = len(sorted_models)
    ordered_actions = [
        action for action in action_colors if action in all_actions_set
    ]
    ordered_actions += sorted(
        action for action in all_actions_set if action not in action_colors
    )
    action_coop_index = {
        action: idx for idx, action in enumerate(ordered_actions)
    }

    def prior_sort_key(prior_tuple: tuple[str, ...]) -> tuple[int, float]:
        if all(action == "NoHistory" for action in prior_tuple):
            return (0, 0.0)
        clean = tuple(clean_action(action) for action in prior_tuple)
        avg_coop = sum(
            action_coop_index.get(action, 0) for action in clean
        ) / len(clean)
        return (1, -avg_coop)

    sorted_priors = sorted(all_priors_set, key=prior_sort_key)
    n_priors = len(sorted_priors)
    group_height = n_models + 1
    n_rows = n_priors * group_height
    fig_height = max(6, n_rows * 0.38 + 2)

    fig, axes = plt.subplots(
        1, len(mechanisms), figsize=(6 * len(mechanisms), fig_height)
    )
    if len(mechanisms) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.12)

    for col_idx, mechanism in enumerate(mechanisms):
        ax = axes[col_idx]
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.set_title(format_mechanism_name(mechanism), fontsize=11)
        ax.set_xlabel("action probabilities in %", fontsize=9)
        ax.invert_yaxis()

        mechanism_runs = cond_freq.get(mechanism, {}).get(target_game, [])

        for group_idx, prior_tuple in enumerate(sorted_priors):
            group_base = group_idx * group_height

            if col_idx == 0:
                group_mid = group_base + (n_models - 1) / 2.0
                ax.text(
                    -0.22,
                    group_mid,
                    f"[{clean_prior_label(prior_tuple)}]",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=7.5,
                    style="italic",
                    clip_on=False,
                )

            for model_idx, player in enumerate(sorted_models):
                y = group_base + model_idx
                run_counters = [
                    {
                        clean_action(action): count
                        for action, count in run_data.get(player, {})
                        .get(prior_tuple, Counter())
                        .items()
                    }
                    for run_data in mechanism_runs
                ]
                run_totals = [sum(counter.values()) for counter in run_counters]

                if all(total == 0 for total in run_totals):
                    ax.barh(y, 100, height=0.8, left=0, color="#e8e8e8")
                    continue

                run_pcts = {
                    action: [
                        counter.get(action, 0) / total * 100
                        for counter, total in zip(run_counters, run_totals)
                        if total > 0
                    ]
                    for action in ordered_actions
                }
                stacked_barh_with_se(
                    ax, y, run_pcts, ordered_actions, action_colors
                )

        for group_idx in range(1, n_priors):
            ax.axhline(
                group_idx * group_height - 0.5,
                color="gray",
                linewidth=0.5,
                linestyle="--",
                alpha=0.5,
            )

        ytick_pos = [
            group * group_height + model
            for group in range(n_priors)
            for model in range(n_models)
        ]
        ytick_labels = (
            [
                format_model_name(player)
                for _ in range(n_priors)
                for player in sorted_models
            ]
            if col_idx == 0
            else [""] * len(ytick_pos)
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
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=len(ordered_actions),
        fontsize=9,
        title="Action (NE -> Cooperative)",
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        f"Conditional Action Distributions - {target_game}",
        fontsize=13,
        fontweight="bold",
    )
    saved_paths = save_matplotlib_figure(
        fig,
        output_dir
        / "conditional_action_frequency"
        / f"conditional_stacked_{target_game}",
        ["png"],
        dpi=150,
        bbox_inches="tight",
        format_subdirs=False,
    )
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for conditional action-frequency plotting."""
    parser = argparse.ArgumentParser(
        description="Generate prior-action conditioned action-frequency figures."
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
    cond_freq = analyze_conditional_action_frequencies(
        experiments, consolidate_players=not args.keep_player_positions
    )

    for game in games:
        if game not in game_configs:
            print(f"Skipping {game}: no game config found.")
            continue
        plot_conditional_actions(
            cond_freq,
            target_game=game,
            game_config=game_configs[game],
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
