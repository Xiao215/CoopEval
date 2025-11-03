#!/usr/bin/env python3
"""
Visualise a match timeline from the readable JSON format using seaborn.

Example:
    python deepseek_viz/timeline_viz.py \
        --input outputs/2025/11/03/14:32/deepseek_self_play.json \
        --output outputs/2025/11/03/14:32/deepseek_self_play_timeline.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns

ACTION_TO_VALUE = {"C": 0, "D": 1}
ACTION_COLORS = ["#2ca02c", "#d62728"]


def load_match(payload: Any, *, match_index: int | None) -> dict[str, Any]:
    """Return the match structure with rounds and player info."""
    if isinstance(payload, list):
        if match_index is None:
            raise ValueError(
                "Readable JSON contains multiple matches. "
                "Provide --match-index to select one."
            )
        if not (1 <= match_index <= len(payload)):
            raise ValueError(
                f"match-index {match_index} out of range "
                f"(contains {len(payload)} matches)."
            )
        return payload[match_index - 1]

    if isinstance(payload, dict):
        if match_index is not None:
            payload_idx = payload.get("match_index")
            if payload_idx is not None and payload_idx != match_index:
                raise ValueError(
                    f"Requested match_index={match_index}, "
                    f"but JSON contains match_index={payload_idx}."
                )
        return payload

    raise TypeError("Unsupported JSON format: expected object or list of objects.")


def iter_players(match: dict[str, Any]) -> Iterable[str]:
    """Yield player names in display order."""
    if "players" in match and match["players"]:
        return list(match["players"])

    return sorted(
        {player["player_name"] for rnd in match["rounds"] for player in rnd["players"]}
    )


def build_dataframe(match: dict[str, Any]) -> pd.DataFrame:
    records = []
    for rnd in match["rounds"]:
        round_idx = rnd["round"]
        for player in rnd["players"]:
            probs = player.get("probabilities") or {}
            player_slot = player.get("player_slot") or f"{player['player_name']}#{player['uid']}"
            prob_c = probs.get("A0")
            prob_d = probs.get("A1")
            action = player.get("action")
            if action == "C":
                chosen_prob = prob_c
                alt_prob = prob_d
            elif action == "D":
                chosen_prob = prob_d
                alt_prob = prob_c
            else:
                chosen_prob = alt_prob = None
            low_confidence = (
                chosen_prob is not None and alt_prob is not None and chosen_prob < alt_prob
            )
            records.append(
                {
                    "round": round_idx,
                    "player_name": player["player_name"],
                    "player_slot": player_slot,
                    "action": action,
                    "action_value": ACTION_TO_VALUE.get(action),
                    "points": player.get("points"),
                    "prob_cooperate": prob_c,
                    "prob_defect": prob_d,
                    "chosen_prob": chosen_prob,
                    "alt_prob": alt_prob,
                    "low_confidence": low_confidence,
                }
            )
    df = pd.DataFrame.from_records(records)
    df.sort_values(["player_slot", "round"], inplace=True)
    return df


def make_annotation(df: pd.DataFrame) -> pd.DataFrame:
    labels = []
    for _, row in df.iterrows():
        label_parts = [row.get("action") or "?"]
        if pd.notna(row.get("points")):
            label_parts.append(f"pts={row['points']}")
        if pd.notna(row.get("prob_cooperate")) and pd.notna(row.get("prob_defect")):
            label_parts.append(f"P(C)={int(row['prob_cooperate'])}%")
            label_parts.append(f"P(D)={int(row['prob_defect'])}%")
        if row.get("low_confidence"):
            label_parts.append("(!)")
        labels.append("\n".join(label_parts))
    df = df.copy()
    df["annotation"] = labels
    return df


def render_timeline(match: dict[str, Any], *, title: str | None = None) -> plt.Figure:
    base_names = list(iter_players(match))
    df = make_annotation(build_dataframe(match))
    if df["action_value"].isna().any():
        missing = df[df["action_value"].isna()]
        actions = missing["action"].unique()
        raise ValueError(f"Encountered unsupported actions in timeline: {actions}")

    action_matrix = df.pivot(index="player_slot", columns="round", values="action_value")
    label_matrix = df.pivot(index="player_slot", columns="round", values="annotation")
    low_matrix = df.pivot(index="player_slot", columns="round", values="low_confidence")

    ordered_slots: list[str] = []
    for name in base_names:
        slots = [slot for slot in action_matrix.index if slot.startswith(f"{name}#")]
        ordered_slots.extend(sorted(slots))

    action_matrix = action_matrix.loc[ordered_slots]
    label_matrix = label_matrix.loc[ordered_slots]
    low_matrix = low_matrix.loc[ordered_slots]

    # Styling
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    cmap = mcolors.ListedColormap(ACTION_COLORS)
    fig_width = max(6.0, 1.0 * action_matrix.shape[1] + 2.5)
    fig_height = max(3.5, 1.2 * len(ordered_slots))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        action_matrix,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=1.0,
        linecolor="white",
        square=False,
        annot=label_matrix.values,
        fmt="",
        annot_kws={
            "fontsize": 8,
            "ha": "center",
            "va": "center",
            "fontfamily": "monospace",
        },
        ax=ax,
    )

    ax.set_xlabel("Round", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Player", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(
        title or match.get("description", "Match Timeline"),
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticklabels(action_matrix.columns, fontsize=10)
    ax.set_yticklabels(action_matrix.index, rotation=0, fontsize=9)

    for row_idx, slot in enumerate(low_matrix.index):
        for col_idx, rnd in enumerate(low_matrix.columns):
            if pd.notna(low_matrix.loc[slot, rnd]) and low_matrix.loc[slot, rnd]:
                rect = Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    fill=False,
                    edgecolor="#FFD700",
                    linewidth=3.0,
                )
                ax.add_patch(rect)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markersize=14,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=label,
        )
        for color, label in zip(ACTION_COLORS, ("Cooperate (C)", "Defect (D)"))
    ]
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markersize=14,
            markerfacecolor="white",
            markeredgecolor="#FFD700",
            markeredgewidth=3.0,
            label="Low Confidence (!)",
        )
    )

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
        title="Actions",
        title_fontsize=11,
    )

    ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Readable JSON produced by post_processing/jsonl_to_readable_json.py.",
    )
    parser.add_argument(
        "--match-index",
        type=int,
        help="Select a single match when the JSON contains multiple entries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save the figure (PNG/PDF/etc). If omitted, the plot is shown.",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Optional title for the figure. Defaults to the match description.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    match = load_match(payload, match_index=args.match_index)
    title = args.title or match.get("description")
    fig = render_timeline(match, title=title)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Saved timeline to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
