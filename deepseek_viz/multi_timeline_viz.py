#!/usr/bin/env python3
"""
Render a grid of timeline heatmaps for multiple matches.

Example:
    python deepseek_viz/multi_timeline_viz.py \
        --inputs \
            outputs/2025/11/03/18:06_probabilistic_deepseek/Repetition_PrisonersDilemma_readable.json \
            outputs/2025/11/03/18:44_direct_deepseek/Repetition_PrisonersDilemmaDirect_readable.json \
        --output figures/deepseek_runs_panel.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns

from timeline_viz import (
    ACTION_COLORS,
    ACTION_TO_VALUE,
    build_dataframe,
    iter_players,
    load_match,
    make_annotation,
)


def load_single_match(payload: Any, match_index: int | None) -> dict[str, Any]:
    """
    Wrapper around timeline_viz.load_match that defaults to the first entry when the
    payload is a singleton list and no match index is provided.
    """
    if isinstance(payload, list) and len(payload) == 1 and match_index is None:
        return payload[0]
    return load_match(payload, match_index=match_index)


def _ordered_slots(
    action_matrix: pd.DataFrame, base_names: Iterable[str]
) -> Sequence[str]:
    ordered: list[str] = []
    for name in base_names:
        slots = [slot for slot in action_matrix.index if slot.startswith(f"{name}#")]
        ordered.extend(sorted(slots))
    return ordered


def plot_timeline_panel(
    ax: plt.Axes,
    match: dict[str, Any],
    *,
    title: str,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    base_names = list(iter_players(match))
    df = make_annotation(build_dataframe(match))

    if df["action_value"].isna().any():
        missing = df[df["action_value"].isna()]
        actions = missing["action"].unique()
        raise ValueError(f"Encountered unsupported actions in timeline: {actions}")

    action_matrix = df.pivot(index="player_slot", columns="round", values="action_value")
    label_matrix = df.pivot(index="player_slot", columns="round", values="annotation")
    low_matrix = df.pivot(index="player_slot", columns="round", values="low_confidence")

    ordered_slots = _ordered_slots(action_matrix, base_names)
    action_matrix = action_matrix.loc[ordered_slots]
    label_matrix = label_matrix.loc[ordered_slots]
    low_matrix = low_matrix.loc[ordered_slots]

    sns.heatmap(
        action_matrix,
        cmap=mcolors.ListedColormap(ACTION_COLORS),
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        square=False,
        annot=label_matrix.values,
        fmt="",
        annot_kws={
            "fontsize": 7,
            "ha": "center",
            "va": "center",
            "fontfamily": "monospace",
        },
        ax=ax,
    )

    if show_xlabel:
        ax.set_xlabel("Round", fontsize=10, fontweight="bold", labelpad=6)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)

    if show_ylabel:
        ax.set_ylabel("Player", fontsize=10, fontweight="bold", labelpad=6)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticklabels(action_matrix.columns, fontsize=8)
    ax.set_yticklabels(action_matrix.index, rotation=0, fontsize=8)
    ax.set_axisbelow(True)

    for row_idx, slot in enumerate(low_matrix.index):
        for col_idx, rnd in enumerate(low_matrix.columns):
            if pd.notna(low_matrix.loc[slot, rnd]) and low_matrix.loc[slot, rnd]:
                rect = Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    fill=False,
                    edgecolor="#FFD700",
                    linewidth=2.0,
                )
                ax.add_patch(rect)


def build_legend_handles() -> list[plt.Line2D]:
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markersize=12,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=label,
        )
        for color, label in zip(ACTION_COLORS, ("Cooperate (C)", "Defect (D)"))
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markersize=12,
            markerfacecolor="white",
            markeredgecolor="#FFD700",
            markeredgewidth=2.0,
            label="Low Confidence (!)",
        )
    )
    return handles


def compute_layout(n_plots: int, cols: int | None) -> tuple[int, int]:
    if cols is None:
        cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)
    return rows, cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        required=True,
        nargs="+",
        type=Path,
        help="Readable JSON files (one per match).",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional titles for each subplot. Defaults to the parent directory name.",
    )
    parser.add_argument(
        "--match-index",
        type=int,
        help="Select a single match when the JSON contains multiple entries.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        help="Number of columns in the grid. Defaults to ceil(sqrt(n_plots)).",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Optional overarching title for the figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination file for the panel figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("Number of labels must match number of inputs.")

    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 9

    matches: list[dict[str, Any]] = []
    titles: list[str] = []
    for idx, path in enumerate(args.inputs):
        payload = json.loads(path.read_text(encoding="utf-8"))
        match = load_single_match(payload, match_index=args.match_index)
        matches.append(match)
        if args.labels:
            titles.append(args.labels[idx])
        else:
            titles.append(path.parent.name or path.stem)

    rows, cols = compute_layout(len(matches), args.cols)
    fig_width = max(4.0 * cols, 8.0)
    fig_height = max(2.8 * rows, 6.0)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)

    for idx, (match, subplot_title) in enumerate(zip(matches, titles, strict=True)):
        ax = axes[idx // cols][idx % cols]
        show_xlabel = idx // cols == rows - 1
        show_ylabel = idx % cols == 0
        plot_timeline_panel(
            ax,
            match,
            title=subplot_title,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
        )

    # Hide any unused axes
    for idx in range(len(matches), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    if args.title:
        fig.suptitle(args.title, fontsize=14, fontweight="bold", y=0.995)

    legend_handles = build_legend_handles()
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_handles),
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Saved panel to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
