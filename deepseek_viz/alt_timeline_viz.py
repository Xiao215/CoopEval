#!/usr/bin/env python3
"""
Alternative multi-panel visualisation for match timelines.

Example:
    python deepseek_viz/alt_timeline_viz.py \
        --input outputs/2025/11/03/14:32/deepseek_self_play.json \
        --output outputs/2025/11/03/14:32/deepseek_self_play_timeline_alt.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_match(payload: Any, *, match_index: int | None) -> dict[str, Any]:
    """Return a single match dict from readable JSON produced by the converter."""
    if isinstance(payload, list):
        if match_index is None:
            raise ValueError(
                "Input JSON contains multiple matches. Provide --match-index to select one."
            )
        if not (1 <= match_index <= len(payload)):
            raise ValueError(
                f"match-index {match_index} out of range (found {len(payload)} matches)."
            )
        return payload[match_index - 1]

    if isinstance(payload, dict):
        if match_index is not None:
            payload_idx = payload.get("match_index")
            if payload_idx is not None and payload_idx != match_index:
                raise ValueError(
                    f"Requested match_index={match_index}, but JSON contains match_index={payload_idx}."
                )
        return payload

    raise TypeError("Unsupported JSON structure: expected object or list of objects.")


def iter_player_slots(match: dict[str, Any]) -> Iterable[str]:
    """Yield unique player slots (name#uid) in the order they appear."""
    seen: set[str] = set()
    ordered: list[str] = []
    for rnd in match["rounds"]:
        for player in rnd["players"]:
            slot = player.get("player_slot") or f"{player['player_name']}#{player['uid']}"
            if slot not in seen:
                seen.add(slot)
                ordered.append(slot)
    return ordered


def prepare_series(match: dict[str, Any]) -> dict[str, Any]:
    slots = list(iter_player_slots(match))
    if len(slots) != 2:
        raise ValueError(
            f"Visualization expects exactly two player slots; found {len(slots)}."
        )

    rounds = []
    series = {
        "rounds": rounds,
        "slots": slots,
        "actions": {slot: [] for slot in slots},
        "points": {slot: [] for slot in slots},
        "prob_a0": {slot: [] for slot in slots},
    }

    for round_entry in match["rounds"]:
        rounds.append(round_entry["round"])
        slot_to_player = {
            player.get("player_slot") or f"{player['player_name']}#{player['uid']}": player
            for player in round_entry["players"]
        }
        for slot in slots:
            player = slot_to_player.get(slot)
            if player is None:
                raise ValueError(
                    f"Round {round_entry['round']} missing entry for player slot {slot}."
                )
            series["actions"][slot].append(player.get("action"))
            series["points"][slot].append(player.get("points") or 0)
            probs = player.get("probabilities") or {}
            series["prob_a0"][slot].append(probs.get("A0"))

    return series


def render_fig(series: dict[str, Any], *, title: str | None = None) -> plt.Figure:
    rounds = series["rounds"]
    slot1, slot2 = series["slots"]
    actions1 = series["actions"][slot1]
    actions2 = series["actions"][slot2]
    points1 = np.array(series["points"][slot1], dtype=float)
    points2 = np.array(series["points"][slot2], dtype=float)
    coop1 = series["prob_a0"][slot1]
    coop2 = series["prob_a0"][slot2]

    cumulative1 = np.cumsum(points1)
    cumulative2 = np.cumsum(points2)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(title or "Prisoner's Dilemma Timeline", fontsize=16, fontweight="bold")

    ax1, ax2, ax3 = axes

    # Actions timeline
    for r, a1, a2 in zip(rounds, actions1, actions2):
        color1 = "#2ecc71" if a1 == "C" else "#e74c3c"
        color2 = "#2ecc71" if a2 == "C" else "#e74c3c"
        ax1.scatter(
            r,
            1,
            s=300,
            c=color1,
            marker="s",
            edgecolors="black",
            linewidth=2,
            zorder=3,
        )
        ax1.text(r, 1, a1 or "?", ha="center", va="center", fontweight="bold", fontsize=10)

        ax1.scatter(
            r,
            0,
            s=300,
            c=color2,
            marker="s",
            edgecolors="black",
            linewidth=2,
            zorder=3,
        )
        ax1.text(r, 0, a2 or "?", ha="center", va="center", fontweight="bold", fontsize=10)

        if a1 == "C" and a2 == "C":
            linestyle, linecolor, alpha = "-", "#27ae60", 0.3
        elif a1 == "D" and a2 == "D":
            linestyle, linecolor, alpha = "--", "#c0392b", 0.3
        else:
            linestyle, linecolor, alpha = ":", "#95a5a6", 0.5

        ax1.plot([r, r], [0, 1], linestyle=linestyle, color=linecolor, alpha=alpha, linewidth=2)

    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(
        [f"{slot2}\n(Player 2)", f"{slot1}\n(Player 1)"],
        fontsize=11,
    )
    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_title("Actions Taken (C = Cooperate, D = Defect)", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)
    ax1.set_xlim(0.5, len(rounds) + 0.5)

    # Cooperation probabilities
    ax2.plot(
        rounds,
        coop1,
        marker="o",
        linewidth=2,
        markersize=8,
        label=f"{slot1} (Player 1)",
        color="#3498db",
    )
    ax2.plot(
        rounds,
        coop2,
        marker="s",
        linewidth=2,
        markersize=8,
        label=f"{slot2} (Player 2)",
        color="#e67e22",
    )
    ax2.fill_between(rounds, coop1, alpha=0.2, color="#3498db")
    ax2.fill_between(rounds, coop2, alpha=0.2, color="#e67e22")
    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("Cooperation Probability (%)", fontsize=12)
    ax2.set_title("Cooperation Intent (Probability of Choosing A0)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, len(rounds) + 0.5)
    ax2.set_ylim(-5, 105)

    # Cumulative score progression
    ax3.plot(
        rounds,
        cumulative1,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label=f"{slot1} (Player 1)",
        color="#3498db",
    )
    ax3.plot(
        rounds,
        cumulative2,
        marker="s",
        linewidth=2.5,
        markersize=8,
        label=f"{slot2} (Player 2)",
        color="#e67e22",
    )
    ax3.fill_between(rounds, cumulative1, alpha=0.2, color="#3498db")
    ax3.fill_between(rounds, cumulative2, alpha=0.2, color="#e67e22")

    for idx, (r, pt1, pt2) in enumerate(zip(rounds, points1, points2)):
        ax3.annotate(
            f"+{int(pt1)}",
            xy=(r, cumulative1[idx]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            alpha=0.7,
            color="#3498db",
        )
        ax3.annotate(
            f"+{int(pt2)}",
            xy=(r, cumulative2[idx]),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            alpha=0.7,
            color="#e67e22",
        )

    ax3.set_xlabel("Round", fontsize=12)
    ax3.set_ylabel("Cumulative Points", fontsize=12)
    ax3.set_title("Score Progression", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=11, loc="upper left")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.5, len(rounds) + 0.5)

    ax3.text(
        rounds[-1],
        cumulative1[-1],
        f" Final: {int(cumulative1[-1])}",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#3498db",
        ha="left",
    )
    ax3.text(
        rounds[-1],
        cumulative2[-1],
        f" Final: {int(cumulative2[-1])}",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#e67e22",
        ha="left",
    )

    fig.tight_layout()
    return fig


def print_summary(series: dict[str, Any]) -> None:
    rounds = len(series["rounds"])
    slot1, slot2 = series["slots"]
    actions1 = series["actions"][slot1]
    actions2 = series["actions"][slot2]
    points1 = np.array(series["points"][slot1], dtype=float)
    points2 = np.array(series["points"][slot2], dtype=float)
    cumulative1 = np.cumsum(points1)
    cumulative2 = np.cumsum(points2)

    mutual_coop = sum(a1 == "C" and a2 == "C" for a1, a2 in zip(actions1, actions2))
    mutual_def = sum(a1 == "D" and a2 == "D" for a1, a2 in zip(actions1, actions2))

    print("\n=== Game Summary ===")
    print(f"Total Rounds: {rounds}")
    print(f"\nPlayer 1 ({slot1}):")
    print(f"  Final Score: {int(cumulative1[-1])}")
    print(f"  Cooperations: {actions1.count('C')}")
    print(f"  Defections: {actions1.count('D')}")
    print(f"\nPlayer 2 ({slot2}):")
    print(f"  Final Score: {int(cumulative2[-1])}")
    print(f"  Cooperations: {actions2.count('C')}")
    print(f"  Defections: {actions2.count('D')}")
    print(f"\nMutual Cooperation rounds: {mutual_coop}")
    print(f"Mutual Defection rounds: {mutual_def}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Readable JSON produced by post_processing/jsonl_to_readable_json.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save the generated figure. Defaults to <input>_alt_timeline.png.",
    )
    parser.add_argument(
        "--match-index",
        type=int,
        help="Select a single match when the JSON contains multiple entries.",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Override the figure title (defaults to match description).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    match = load_match(payload, match_index=args.match_index)
    title = args.title or match.get("description")

    series = prepare_series(match)
    fig = render_fig(series, title=title)

    output_path = args.output
    if output_path is None:
        suffix = "_alt_timeline.png"
        output_path = args.input.with_suffix("")
        output_path = output_path.with_name(output_path.name + suffix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_path}")

    print_summary(series)


if __name__ == "__main__":
    main()
