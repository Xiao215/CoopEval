#!/usr/bin/env python3
"""Plot mechanism-vs-baseline taxonomy deltas as a single diverging bar chart."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

from coopeval.llm_judge.plotting_utils import (
    dataset_share_csv_path,
    prepare_figure_subdir,
    validate_input_name,
)
from coopeval.llm_judge.taxonomy_dataset import TaxonomyDataset
from coopeval.script_utils.display_helper import (
    format_mechanism_name,
    sort_mechanisms,
)
from coopeval.script_utils.colors import MECHANISM_COLORS
from coopeval.script_utils.figure_exports import save_matplotlib_figure

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "axes.titlesize": 10,
    }
)

_LABEL_DISPLAY: dict[str, str] = {
    "individual utility maximization": "Utility\nMaximization",
    "strategic equilibrium focus": "Strategic Equilibrium\nFocus",
    "strategic influence": "Strategic\nInfluence",
    "uncertainty evaluation": "Uncertainty\nEvaluation",
    "multidimensional reasoning": "Multidimensional\nReasoning",
    "trust evaluation": "Trust\nEvaluation",
    "risk aversion": "Risk Aversion",
    "reciprocity": "Reciprocity",
    "social welfare maximization": "Social Welfare\nMaximization",
    "competitiveness": "Competitiveness",
    "social norm conformity": "Social Norm\nConformity",
    "rule misunderstanding": "Rule\nMisunderstanding",
    "exploration-exploitation trade-off": "Exploration-\nExploitation",
    "inequity aversion": "Inequity\nAversion",
    "strategy legibility": "Strategy\nLegibility",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the mechanism delta bar chart."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot single-panel mechanism deltas against NoMechanism for "
            "taxonomy label shares."
        )
    )
    parser.add_argument(
        "input_name",
        type=str,
        help="Judge run identifier (auto-discovers dataset share CSV).",
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=8,
        help=(
            "Number of top labels per mechanism to union together "
            "(0 = all labels, default: 8)."
        ),
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="taxonomy_delta_bars_vs_nomechanism",
        help=(
            "Output filename stem "
            "(default: taxonomy_delta_bars_vs_nomechanism)."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=["horizontal", "vertical", "both"],
        default="both",
        help=("Which layout(s) to export " "(default: both)."),
    )
    args = parser.parse_args()
    args.input_name = validate_input_name(args.input_name)
    return args


def load_share_data(path: Path) -> pd.DataFrame:
    """Load the dataset share CSV and coerce numeric columns."""
    df = pd.read_csv(path)
    required = {
        "game",
        "mechanism",
        "model",
        "player",
        "label",
        "share_pct",
        "group_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {sorted(missing)}"
        )
    df["share_pct"] = pd.to_numeric(df["share_pct"], errors="coerce")
    df["group_count"] = pd.to_numeric(df["group_count"], errors="coerce")
    df = df.dropna(subset=["share_pct", "group_count"]).copy()
    return df


def build_mechanism_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to a mechanism-by-label share matrix."""
    grouped = df.groupby(["mechanism", "label"], as_index=False)[
        "share_pct"
    ].mean()
    matrix = grouped.pivot(
        index="mechanism", columns="label", values="share_pct"
    ).fillna(0.0)
    ordered = [
        mechanism
        for mechanism in sort_mechanisms(matrix.index)
        if mechanism in matrix.index
    ]
    return matrix.loc[ordered]


def select_labels(dataset: TaxonomyDataset, top_labels: int) -> list[str]:
    """Choose labels via the union of each mechanism's top-N labels."""
    if top_labels > 0:
        return dataset.union_top_labels("mechanism", top_labels)
    return dataset.top_labels()


def compute_deltas_and_tests(
    df: pd.DataFrame,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Compute mechanism-vs-NoMechanism deltas with Bonferroni correction."""
    if "NoMechanism" not in set(df["mechanism"]):
        raise RuntimeError("NoMechanism rows are required for delta plotting.")

    work = df.copy()
    work["positive"] = work["share_pct"] / 100.0 * work["group_count"]

    cells = (
        work.groupby(["game", "mechanism", "model", "player"])["group_count"]
        .first()
        .reset_index()
    )
    totals = cells.groupby("mechanism")["group_count"].sum()

    agg = (
        work.groupby(["mechanism", "label"], as_index=False)
        .agg(positives=("positive", "sum"))
        .merge(totals.rename("n_total").reset_index(), on="mechanism")
    )

    baseline = agg[agg["mechanism"] == "NoMechanism"][
        ["label", "positives", "n_total"]
    ].rename(columns={"positives": "pos_baseline", "n_total": "n_baseline"})

    results: dict[tuple[str, str], tuple[float, float]] = {}
    mechanisms = [
        mechanism
        for mechanism in sort_mechanisms(agg["mechanism"].unique())
        if mechanism != "NoMechanism"
    ]
    for mechanism in mechanisms:
        mech_data = agg[agg["mechanism"] == mechanism][
            ["label", "positives", "n_total"]
        ].rename(columns={"positives": "pos_mech", "n_total": "n_mech"})
        merged = mech_data.merge(baseline, on="label", how="outer").fillna(0)
        for _, row in merged.iterrows():
            n_mech = int(row["n_mech"])
            n_baseline = int(row["n_baseline"])
            pos_mech = float(row["pos_mech"])
            pos_baseline = float(row["pos_baseline"])
            p_mech = pos_mech / n_mech if n_mech > 0 else 0.0
            p_baseline = pos_baseline / n_baseline if n_baseline > 0 else 0.0
            delta = (p_mech - p_baseline) * 100.0
            p_value = 1.0
            if n_mech > 0 and n_baseline > 0:
                pooled = (pos_mech + pos_baseline) / (n_mech + n_baseline)
                if 0.0 < pooled < 1.0:
                    se = np.sqrt(
                        pooled
                        * (1.0 - pooled)
                        * (1.0 / n_mech + 1.0 / n_baseline)
                    )
                    z_score = (p_mech - p_baseline) / se
                    p_value = 2 * stats.norm.sf(abs(z_score))
            results[(mechanism, str(row["label"]))] = (delta, p_value)

    n_tests = max(1, len(results))
    for key, (delta, p_value) in list(results.items()):
        results[key] = (delta, min(p_value * n_tests, 1.0))
    return results


def significance_marker(p_value_bonf: float) -> str:
    """Return significance stars for a corrected p-value."""
    if p_value_bonf < 0.001:
        return "***"
    if p_value_bonf < 0.01:
        return "**"
    if p_value_bonf < 0.05:
        return "*"
    return ""


def format_axis_label(label: str, width: int = 22) -> str:
    """Format a taxonomy label for display on the y-axis."""
    key = label.strip().lower()
    if key in _LABEL_DISPLAY:
        return _LABEL_DISPLAY[key]
    return "\n".join(textwrap.wrap(label.title(), width=width))


def plot_delta_bars(
    matrix: pd.DataFrame,
    sig_data: dict[tuple[str, str], tuple[float, float]],
    labels: list[str],
    output_path: Path,
    root_dir: Path,
    *,
    layout: str,
) -> None:
    """Render a single-panel diverging bar chart vs NoMechanism."""
    if "NoMechanism" not in matrix.index:
        raise RuntimeError("NoMechanism row is required for delta plotting.")

    baseline = matrix.loc["NoMechanism"]
    mechanisms = [
        mechanism
        for mechanism in sort_mechanisms(matrix.index)
        if mechanism != "NoMechanism"
    ]
    delta = matrix.loc[mechanisms].subtract(baseline, axis="columns")[labels]

    n_mechanisms = len(mechanisms)
    if n_mechanisms == 0:
        raise RuntimeError("Need at least one non-baseline mechanism to plot.")

    bar_height = 0.7 / n_mechanisms
    group_gap = 0.42
    if layout == "horizontal":
        fig_width = 9.0
        fig_height = max(
            5.2, len(labels) * (n_mechanisms * bar_height + group_gap)
        )
        legend_columns = min(len(mechanisms), 3)
        legend_anchor = (0.5, -0.12)
    else:
        fig_width = max(11.0, len(labels) * 1.45)
        fig_height = 6.5
        legend_columns = min(len(mechanisms), 6)
        legend_anchor = (0.5, -0.16)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if layout == "horizontal":
        for i, label in enumerate(reversed(labels)):
            y_center = i * (n_mechanisms * bar_height + group_gap)
            for j, mechanism in enumerate(mechanisms):
                y_pos = y_center + j * bar_height
                value = float(delta.loc[mechanism, label])
                color = MECHANISM_COLORS.get(mechanism, "#888888")
                ax.barh(
                    y_pos,
                    value,
                    height=bar_height * 0.86,
                    color=color,
                    alpha=0.9,
                    edgecolor="white",
                    linewidth=0.35,
                )
                marker = significance_marker(
                    sig_data.get((mechanism, label), (value, 1.0))[1]
                )
                if abs(value) >= 1.5:
                    offset = 1.25 if value >= 0 else -1.25
                    ha = "left" if value >= 0 else "right"
                    text = (
                        f"{value:+.0f}{marker}" if marker else f"{value:+.0f}"
                    )
                    ax.text(
                        value + offset,
                        y_pos,
                        text,
                        va="center",
                        ha=ha,
                        fontsize=7,
                        color="#333333",
                    )

        ytick_positions: list[float] = []
        ytick_labels: list[str] = []
        for i, label in enumerate(reversed(labels)):
            center = (
                i * (n_mechanisms * bar_height + group_gap)
                + (n_mechanisms - 1) * bar_height / 2.0
            )
            ytick_positions.append(center)
            ytick_labels.append(format_axis_label(label))

        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize=9)
        ax.axvline(0, color="#444444", linewidth=0.8, zorder=0)
        ax.axvspan(-5, 5, color="#f0f0f0", alpha=0.6, zorder=0)
        ax.grid(axis="x", alpha=0.2, zorder=0)
        ax.set_xlabel("Δ Share (pp) vs. NoMechanism", fontsize=9)
    else:
        group_width = 0.82
        bar_width = group_width / n_mechanisms
        x_centers = np.arange(len(labels), dtype=float)
        y_limit = max(25.0, float(np.nanmax(np.abs(delta.to_numpy()))))

        for j, mechanism in enumerate(mechanisms):
            x_pos = x_centers - group_width / 2.0 + (j + 0.5) * bar_width
            values = delta.loc[mechanism, labels].astype(float).to_numpy()
            color = MECHANISM_COLORS.get(mechanism, "#888888")
            ax.bar(
                x_pos,
                values,
                width=bar_width * 0.88,
                color=color,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.35,
            )
            for x_value, label, value in zip(x_pos, labels, values):
                marker = significance_marker(
                    sig_data.get((mechanism, label), (value, 1.0))[1]
                )
                if abs(value) >= 1.5:
                    offset = 0.9 if value >= 0 else -0.9
                    va = "bottom" if value >= 0 else "top"
                    text = (
                        f"{value:+.0f}{marker}" if marker else f"{value:+.0f}"
                    )
                    ax.text(
                        x_value,
                        value + offset,
                        text,
                        ha="center",
                        va=va,
                        fontsize=7,
                        color="#333333",
                    )

        ax.set_xticks(x_centers)
        ax.set_xticklabels(
            [format_axis_label(label, width=16) for label in labels],
            fontsize=8,
            rotation=35,
            ha="right",
        )
        ax.axhline(0, color="#444444", linewidth=0.8, zorder=0)
        ax.axhspan(-5, 5, color="#f0f0f0", alpha=0.6, zorder=0)
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.set_ylabel("Δ Share (pp) vs. NoMechanism", fontsize=9)
        ax.set_ylim(-y_limit * 1.18, y_limit * 1.18)
        ax.set_xlim(-0.6, len(labels) - 0.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "How Mechanisms Shift Justification Profiles",
        fontsize=10,
        fontweight="bold",
        pad=10,
    )

    handles = [
        Patch(
            facecolor=MECHANISM_COLORS.get(mechanism, "#888888"),
            label=format_mechanism_name(mechanism),
        )
        for mechanism in mechanisms
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=legend_anchor,
        ncol=legend_columns,
        framealpha=0.95,
        edgecolor="#cccccc",
    )

    if layout == "horizontal":
        x_max = (
            float(np.nanmax(np.abs(delta.to_numpy()))) if not delta.empty else 0
        )
        x_max = max(25.0, x_max)
        ax.set_xlim(-x_max * 1.15, x_max * 1.15)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18 if layout == "horizontal" else 0.25)
    saved_paths = save_matplotlib_figure(
        fig,
        output_path,
        ("png",),
        dpi=300,
        root_dir=root_dir,
        bbox_inches="tight",
    )
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


def main() -> None:
    """CLI entry point for the taxonomy delta bar chart."""
    args = parse_args()
    input_name = args.input_name
    share_csv = dataset_share_csv_path(input_name)
    dataset = TaxonomyDataset.from_share_csv(share_csv)
    df = load_share_data(share_csv)
    matrix = build_mechanism_matrix(df)
    labels = select_labels(dataset, args.top_labels)
    sig_data = compute_deltas_and_tests(df)

    output_root = prepare_figure_subdir(input_name, "taxonomy_delta_bars")
    layouts = (
        ["horizontal", "vertical"] if args.layout == "both" else [args.layout]
    )
    for layout in layouts:
        stem = args.output_stem
        if layout == "vertical":
            stem = f"{stem}_vertical"
        plot_delta_bars(
            matrix=matrix,
            sig_data=sig_data,
            labels=labels,
            output_path=output_root / stem,
            root_dir=output_root,
            layout=layout,
        )


if __name__ == "__main__":
    main()
