#!/usr/bin/env python3
"""
Render a grid of timeline heatmaps for multiple matches with enhanced styling.

Example:
    python deepseek_viz/multi_timeline_viz_alt.py \
        --inputs \
            outputs/2025/11/03/18:06_probabilistic_deepseek/Repetition_PrisonersDilemma_readable.json \
            outputs/2025/11/03/18:44_direct_deepseek/Repetition_PrisonersDilemmaDirect_readable.json \
        --output figures/deepseek_runs_panel_alt.png
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
import matplotlib.patheffects as path_effects
import pandas as pd
import seaborn as sns

from timeline_viz import (
    build_dataframe,
    iter_players,
    load_match,
    make_annotation,
)


def load_single_match(payload: Any, match_index: int | None) -> dict[str, Any]:
    """Wrapper around timeline_viz.load_match with singleton fallback."""
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


def estimate_subplot_dimensions(match: dict[str, Any]) -> dict[str, float]:
    """Estimate subplot width/height heuristics based on annotations."""
    df = make_annotation(build_dataframe(match))
    action_matrix = df.pivot(index="player_slot", columns="round", values="action_value")

    n_players = action_matrix.shape[0]
    n_rounds = action_matrix.shape[1]

    max_line_len = 0
    max_lines = 0
    for annotation in df["annotation"]:
        lines = annotation.splitlines() or [annotation]
        max_line_len = max(max_line_len, *(len(line) for line in lines))
        max_lines = max(max_lines, len(lines))

    width = max(6.0, 0.55 * n_rounds + 3.5, 0.25 * max_line_len + 3.0)
    height = max(3.0, 0.7 * n_players + 1.8, 0.5 * max_lines + 2.0)

    return {
        "width": width,
        "height": height,
        "n_rounds": n_rounds,
        "n_players": n_players,
    }


def calculate_font_size(n_rounds: int, n_players: int) -> float:
    """Calculate a dynamic annotation font size based on grid dimensions."""
    base_size = 8.5
    size_factor = max(0.55, 1.0 - (n_rounds - 10) * 0.015 - (n_players - 4) * 0.03)
    return base_size * size_factor


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

    cmap = mcolors.ListedColormap(["#27ae60", "#e74c3c"])

    n_rounds = len(action_matrix.columns)
    n_players = len(action_matrix.index)
    font_size = calculate_font_size(n_rounds, n_players)

    sns.heatmap(
        action_matrix,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=1.2,
        linecolor="white",
        square=False,
        annot=label_matrix.values,
        fmt="",
        annot_kws={
            "fontsize": font_size,
            "ha": "center",
            "va": "center",
            "fontfamily": "monospace",
            "weight": "bold",
            "color": "white",
        },
        ax=ax,
    )

    for text in ax.texts:
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=1.5, foreground="black", alpha=0.5),
                path_effects.Normal(),
            ]
        )

    if show_xlabel:
        ax.set_xlabel("Round", fontsize=11, fontweight="bold", labelpad=8, color="#2c3e50")
        ax.tick_params(axis="x", labelsize=9, length=4, width=1.2, colors="#34495e", pad=4)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False, length=0)

    if show_ylabel:
        ax.set_ylabel("Player", fontsize=11, fontweight="bold", labelpad=8, color="#2c3e50")
        ax.tick_params(axis="y", labelsize=8.5, length=4, width=1.2, colors="#34495e", pad=4)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False, length=0)

    title_text = ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color="#2c3e50")
    title_text.set_bbox(
        dict(
            boxstyle="round,pad=0.5",
            facecolor="#ecf0f1",
            edgecolor="#bdc3c7",
            linewidth=1.5,
            alpha=0.8,
        )
    )

    ax.set_xticklabels(action_matrix.columns, fontsize=9, weight="medium")
    ax.set_yticklabels(action_matrix.index, rotation=0, fontsize=8.5, weight="medium")

    for spine in ax.spines.values():
        spine.set_edgecolor("#95a5a6")
        spine.set_linewidth(1.5)

    for row_idx, slot in enumerate(low_matrix.index):
        for col_idx, rnd in enumerate(low_matrix.columns):
            if pd.notna(low_matrix.loc[slot, rnd]) and low_matrix.loc[slot, rnd]:
                rect = Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    fill=False,
                    edgecolor="#f39c12",
                    linewidth=2.5,
                    linestyle="--",
                    alpha=0.9,
                )
                ax.add_patch(rect)


def build_legend_handles() -> list[plt.Line2D]:
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markersize=14,
            markerfacecolor="#27ae60",
            markeredgecolor="#1e8449",
            markeredgewidth=1.5,
            label="Cooperate (C)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markersize=14,
            markerfacecolor="#e74c3c",
            markeredgecolor="#c0392b",
            markeredgewidth=1.5,
            label="Defect (D)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markersize=14,
            markerfacecolor="white",
            markeredgecolor="#f39c12",
            markeredgewidth=2.5,
            linestyle="--",
            label="Low Confidence (!)",
        ),
    ]


def compute_layout(n_plots: int, cols: int | None) -> tuple[int, int]:
    if cols is None:
        if n_plots <= 2:
            cols = n_plots
        elif n_plots <= 4:
            cols = 2
        else:
            cols = min(3, n_plots)
    rows = math.ceil(n_plots / cols)
    return rows, cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, nargs="+", type=Path, help="Readable JSON files.")
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional titles for each subplot (defaults to parent directory name).",
    )
    parser.add_argument(
        "--match-index",
        type=int,
        help="Select a single match when the JSON contains multiple entries.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        help="Number of columns in the grid. Auto-computed if omitted.",
    )
    parser.add_argument("--title", type=str, help="Optional figure-level title.")
    parser.add_argument("--output", type=Path, help="Destination file for the panel figure.")
    parser.add_argument(
        "--judgements",
        type=Path,
        help=(
            "Aggregated strategy judgements JSON (from run_strategy_judge.sh). "
            "If provided, subplot titles include per-player strategy labels."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("Number of labels must match number of inputs.")

    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.facecolor"] = "#fafafa"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#95a5a6"
    plt.rcParams["grid.color"] = "#ecf0f1"
    plt.rcParams["grid.linewidth"] = 0.8

    judgement_map: dict[str, list[tuple[str, str]]] = {}
    if args.judgements:
        payload = json.loads(args.judgements.read_text(encoding="utf-8"))
        for block in payload:
            source = block.get("source")
            entries = block.get("entries", [])
            if not source:
                continue
            parent_dir = str(Path(source).resolve().parent)
            labels: list[tuple[str, str]] = []
            for entry in entries:
                history = entry.get("history") or {}
                slot = history.get("player_slot") or "unknown"
                judgement = entry.get("judgement") or {}
                label = judgement.get("label")
                if label:
                    labels.append((slot, label))
            if labels:
                judgement_map[parent_dir] = labels

    matches: list[dict[str, Any]] = []
    titles: list[str] = []
    metas: list[dict[str, float]] = []
    for idx, path in enumerate(args.inputs):
        payload = json.loads(path.read_text(encoding="utf-8"))
        match = load_single_match(payload, match_index=args.match_index)
        matches.append(match)
        metas.append(estimate_subplot_dimensions(match))
        if args.labels:
            titles.append(args.labels[idx])
        else:
            dir_name = path.parent.name or path.stem
            cleaned = dir_name.replace("_", " ").title()
            run_dir = str(path.parent.resolve())
            labels = judgement_map.get(run_dir)
            if labels:
                summary = "; ".join(
                    f"{slot.split('#')[-1]}: {label}" for slot, label in labels
                )
                cleaned = f"{cleaned}\n[{summary}]"
            titles.append(cleaned)

    rows, cols = compute_layout(len(matches), args.cols)

    col_widths = [1.0] * cols
    row_heights = [1.0] * rows
    for idx, meta in enumerate(metas):
        row_idx, col_idx = divmod(idx, cols)
        col_widths[col_idx] = max(col_widths[col_idx], meta["width"])
        row_heights[row_idx] = max(row_heights[row_idx], meta["height"])

    fig_width = sum(col_widths)
    fig_height = sum(row_heights)
    if args.title:
        fig_height += 0.8
    fig_height += 1.0

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=False,
        gridspec_kw={"width_ratios": col_widths, "height_ratios": row_heights},
    )

    for idx, (match, subplot_title) in enumerate(zip(matches, titles, strict=True)):
        row_idx, col_idx = divmod(idx, cols)
        ax = axes[row_idx][col_idx]
        plot_timeline_panel(
            ax,
            match,
            title=subplot_title,
            show_xlabel=row_idx == rows - 1,
            show_ylabel=col_idx == 0,
        )

    for idx in range(len(matches), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    if args.title:
        fig.suptitle(
            args.title,
            fontsize=16,
            fontweight="bold",
            color="#2c3e50",
            y=0.99,
        )

    legend_handles = build_legend_handles()
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_handles),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10.5,
        title="Action Legend",
        title_fontsize=11,
        edgecolor="#95a5a6",
        facecolor="white",
        framealpha=0.95,
    )
    legend.get_frame().set_linewidth(1.5)

    top_margin = 0.96 if args.title else 0.98
    fig.tight_layout(rect=[0, 0.05, 1, top_margin])
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Saved panel to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
