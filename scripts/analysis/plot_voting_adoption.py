#!/usr/bin/env python3
"""Generate voting/adoption figures from cleaned CoopEval runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from coopeval.config import FIGURE_DIR
from coopeval.script_utils.display_helper import (
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
from coopeval.utils.json_io import clean_path
from coopeval.script_utils.colors import MECHANISM_COLORS
from coopeval.script_utils.figure_exports import save_matplotlib_figure

VotingStatsIndex = dict[str, dict[str, list[dict[str, Any]]]]
VOTING_MECHANISM_TYPES = {"Contracting", "Mediation"}


def analyze_voting_adoption_stats(
    experiments: list[ExperimentInfo],
    *,
    consolidate_players: bool = True,
) -> VotingStatsIndex:
    """Extract voting and adoption statistics for contracting and mediation."""

    voting_stats: VotingStatsIndex = {}

    for experiment in experiments:
        if experiment.mechanism not in VOTING_MECHANISM_TYPES:
            continue

        run_stats: dict[str, Any] = {
            "max_votes_distribution": Counter(),
            "unanimous_adoption_count": 0,
            "total_matches": 0,
            "adoption_counts_distribution": Counter(),
            "proposal_votes_received": defaultdict(Counter),
            "adoption_by_player": defaultdict(Counter),
        }

        for match in load_json_lines(experiment.path / "records.jsonl"):
            match_data = match[0]
            moves = match_data["moves"]

            run_stats["total_matches"] += 1
            votes_list = match_data["votes"]
            proposal_vote_counts = Counter()

            for voter_record in votes_list:
                for proposal_id, voted_yes in voter_record["votes"].items():
                    if voted_yes:
                        proposal_vote_counts[proposal_id] += 1

            max_votes_received = max(proposal_vote_counts.values(), default=0)
            run_stats["max_votes_distribution"][max_votes_received] += 1

            index_to_designer = {
                str(idx + 1): (
                    strip_player_suffix(votes_list[idx]["voter_name"])
                    if consolidate_players
                    else votes_list[idx]["voter_name"]
                )
                for idx in range(len(votes_list))
            }
            for proposal_id, designer in index_to_designer.items():
                run_stats["proposal_votes_received"][designer][
                    proposal_vote_counts[proposal_id]
                ] += 1

            if experiment.mechanism == "Contracting":
                signatures = match_data["signatures"]
                accepted_count = sum(
                    1 for sig in signatures.values() if sig["agree"]
                )
                run_stats["adoption_counts_distribution"][accepted_count] += 1
                if match_data["all_signed"]:
                    run_stats["unanimous_adoption_count"] += 1

                for player_name, sig in signatures.items():
                    run_stats["adoption_by_player"][
                        (
                            strip_player_suffix(player_name)
                            if consolidate_players
                            else player_name
                        )
                    ][sig["agree"]] += 1

            elif experiment.mechanism == "Mediation":
                accepted_count = sum(
                    1 for move in moves if move.get("mediated", False)
                )
                run_stats["adoption_counts_distribution"][accepted_count] += 1
                if accepted_count == len(moves):
                    run_stats["unanimous_adoption_count"] += 1

                for move in moves:
                    run_stats["adoption_by_player"][
                        (
                            strip_player_suffix(move["player"])
                            if consolidate_players
                            else move["player"]
                        )
                    ][move.get("mediated", False)] += 1

        mechanism_stats = voting_stats.setdefault(experiment.mechanism, {})
        game_stats = mechanism_stats.setdefault(experiment.game, [])
        game_stats.append(run_stats)

    return voting_stats


def _plot_empty_axis(ax: Any, title: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_axis_off()


def plot_voting_and_adoption(
    voting_stats: VotingStatsIndex,
    *,
    target_game: str,
    output_dir: Path,
) -> None:
    """Plot voting and adoption summaries for one game."""

    mechanisms = sort_mechanisms(
        [
            mechanism
            for mechanism, game_stats in voting_stats.items()
            if game_stats.get(target_game)
        ]
    )
    if not mechanisms:
        print(f"No voting/adoption data found for game: {target_game}")
        return

    display_mechs = [
        format_mechanism_name(mechanism) for mechanism in mechanisms
    ]
    mech_palette = {
        format_mechanism_name(mechanism): MECHANISM_COLORS[mechanism]
        for mechanism in mechanisms
    }
    fig, axes = plt.subplots(1, 4, figsize=(26, 5))

    rows = []
    for mechanism in mechanisms:
        for run_idx, run_stats in enumerate(
            voting_stats[mechanism].get(target_game, [])
        ):
            total = run_stats["total_matches"]
            if total == 0:
                continue
            for n_votes, frequency in run_stats[
                "max_votes_distribution"
            ].items():
                rows.append(
                    {
                        "Mechanism": format_mechanism_name(mechanism),
                        "Max Votes": n_votes,
                        "Percentage": frequency / total * 100,
                        "Run": run_idx,
                    }
                )
    df1 = pd.DataFrame(rows)
    if df1.empty:
        _plot_empty_axis(axes[0], "Max Votes a Proposal Received")
    else:
        sns.barplot(
            data=df1.sort_values("Max Votes"),
            x="Max Votes",
            y="Percentage",
            hue="Mechanism",
            hue_order=display_mechs,
            palette=mech_palette,
            estimator="mean",
            errorbar="se",
            ax=axes[0],
        )
        axes[0].set_title("Max Votes a Proposal Received")
        axes[0].set_xlabel("Number of Votes")
        axes[0].set_ylabel("Frequency (%)")
        axes[0].set_ylim(0, 100)

    rows = []
    for mechanism in mechanisms:
        for run_idx, run_stats in enumerate(
            voting_stats[mechanism].get(target_game, [])
        ):
            total = run_stats["total_matches"]
            if total == 0:
                continue
            for n_players, frequency in run_stats[
                "adoption_counts_distribution"
            ].items():
                rows.append(
                    {
                        "Mechanism": format_mechanism_name(mechanism),
                        "Players": n_players,
                        "Percentage": frequency / total * 100,
                        "Run": run_idx,
                    }
                )
    df2 = pd.DataFrame(rows)
    if df2.empty:
        _plot_empty_axis(axes[1], "Total Players Accepted / Delegated")
    else:
        sns.barplot(
            data=df2.sort_values("Players"),
            x="Players",
            y="Percentage",
            hue="Mechanism",
            hue_order=display_mechs,
            palette=mech_palette,
            estimator="mean",
            errorbar="se",
            ax=axes[1],
        )
        axes[1].set_title("Total Players Accepted / Delegated")
        axes[1].set_xlabel("Number of Players")
        axes[1].set_ylabel("Frequency (%)")
        axes[1].set_ylim(0, 100)

    num_players = max(
        (
            n_votes
            for mechanism in mechanisms
            for run_stats in voting_stats[mechanism].get(target_game, [])
            for n_votes in run_stats["max_votes_distribution"]
        ),
        default=1,
    )

    all_players: set[str] = set()
    for mechanism in mechanisms:
        for run_stats in voting_stats[mechanism].get(target_game, []):
            all_players.update(run_stats["proposal_votes_received"])
            all_players.update(run_stats["adoption_by_player"])
    model_order = list(
        dict.fromkeys(
            format_model_name(player)
            for player in sort_agents(list(all_players))
        )
    )

    rows = []
    for mechanism in mechanisms:
        for run_idx, run_stats in enumerate(
            voting_stats[mechanism].get(target_game, [])
        ):
            total = run_stats["total_matches"]
            if total == 0:
                continue
            for player, vote_dist in run_stats[
                "proposal_votes_received"
            ].items():
                model = format_model_name(player)
                avg_votes = (
                    sum(n_votes * count for n_votes, count in vote_dist.items())
                    / total
                    / num_players
                    * 100
                )
                rows.append(
                    {
                        "Model": model,
                        "Mechanism": format_mechanism_name(mechanism),
                        "Avg Votes": avg_votes,
                        "Run": run_idx,
                    }
                )
    df3 = pd.DataFrame(rows)
    if df3.empty:
        _plot_empty_axis(axes[2], "Avg Votes Received per Model's Proposal")
    else:
        sns.barplot(
            data=df3,
            x="Model",
            y="Avg Votes",
            hue="Mechanism",
            order=model_order,
            hue_order=display_mechs,
            palette=mech_palette,
            estimator="mean",
            errorbar="se",
            ax=axes[2],
        )
        axes[2].set_title("Avg Votes Received per Model's Proposal")
        axes[2].set_xlabel("Model")
        axes[2].set_ylabel("Avg % of Voters per Match")
        axes[2].set_ylim(0, 100)
        plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right")

    rows = []
    for mechanism in mechanisms:
        for run_idx, run_stats in enumerate(
            voting_stats[mechanism].get(target_game, [])
        ):
            for player, decision_dist in run_stats[
                "adoption_by_player"
            ].items():
                model = format_model_name(player)
                total_decisions = sum(decision_dist.values())
                if total_decisions == 0:
                    continue
                accept_rate = decision_dist[True] / total_decisions * 100
                rows.append(
                    {
                        "Model": model,
                        "Mechanism": format_mechanism_name(mechanism),
                        "Acceptance Rate (%)": accept_rate,
                        "Run": run_idx,
                    }
                )
    df4 = pd.DataFrame(rows)
    if df4.empty:
        _plot_empty_axis(axes[3], "Acceptance / Delegation Rate per Model")
    else:
        sns.barplot(
            data=df4,
            x="Model",
            y="Acceptance Rate (%)",
            hue="Mechanism",
            order=model_order,
            hue_order=display_mechs,
            palette=mech_palette,
            estimator="mean",
            errorbar="se",
            ax=axes[3],
        )
        axes[3].set_title("Acceptance / Delegation Rate per Model")
        axes[3].set_xlabel("Model")
        axes[3].set_ylabel("Acceptance Rate (%)")
        axes[3].set_ylim(0, 100)
        plt.setp(axes[3].get_xticklabels(), rotation=45, ha="right")

    fig.suptitle(
        f"Voting & Adoption - {target_game}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    saved_paths = save_matplotlib_figure(
        fig,
        output_dir / "voting_adoption" / f"voting_adoption_{target_game}",
        ["png"],
        dpi=300,
        bbox_inches="tight",
        format_subdirs=False,
    )
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for voting/adoption plotting."""
    parser = argparse.ArgumentParser(
        description="Generate voting/adoption figures for contracting and mediation."
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

    games = sort_games({experiment.game for experiment in experiments})
    voting_stats = analyze_voting_adoption_stats(
        experiments, consolidate_players=not args.keep_player_positions
    )

    for game in games:
        plot_voting_and_adoption(
            voting_stats,
            target_game=game,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
