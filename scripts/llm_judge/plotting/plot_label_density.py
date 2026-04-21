#!/usr/bin/env python3
"""Small-multiple violin plots for confidence/response-length by label."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from coopeval.llm_judge.plotting_utils import (
    dataset_share_csv_path,
    normalized_jsonl_path,
    prepare_figure_subdir,
    validate_input_name,
)
from coopeval.llm_judge.taxonomy_dataset import TaxonomyDataset
from coopeval.script_utils.display_helper import (
    format_mechanism_name,
    format_model_name,
    sort_agents,
    sort_games,
    sort_mechanisms,
)
from coopeval.script_utils.llm_judge_helpers import classification_labels
from coopeval.utils.json_io import iter_jsonl
from coopeval.script_utils.figure_exports import save_matplotlib_figure


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for taxonomy label density violins."""
    parser = argparse.ArgumentParser(
        description="Plot metric distributions per label and group."
    )
    parser.add_argument(
        "input_name",
        type=str,
        help="Judge run identifier (auto-discovers normalized JSONL).",
    )
    parser.add_argument(
        "--metric",
        choices=["classification_confidence", "response_chars"],
        default="classification_confidence",
        help="Metric to visualize.",
    )
    parser.add_argument(
        "--group-field",
        choices=["mechanism", "game", "model"],
        default="mechanism",
        help="Grouping dimension (default: mechanism).",
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=6,
        help=(
            "Number of top labels per mechanism to union together "
            "(default: 6)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["count", "frequency"],
        default="frequency",
        help="Label axis mode (default: frequency).",
    )
    args = parser.parse_args()
    args.input_name = validate_input_name(args.input_name)
    return args


def load_records(
    path: str | Path, metric: str, group_field: str
) -> pd.DataFrame:
    """Load label/metric rows from the normalized JSONL."""
    rows = []
    for payload in iter_jsonl(Path(path)):
        labels = classification_labels(payload)
        if not labels:
            continue
        value = payload.get(metric)
        if value is None and metric == "response_chars":
            text = payload.get("response_text") or ""
            value = len(text)
        if value is None:
            continue
        group = payload.get(group_field, "Unknown")
        for label in labels:
            rows.append(
                {"label": label, "metric": float(value), "group": group}
            )
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point for label density visualization."""
    args = parse_args()
    input_name = args.input_name
    jsonl_path = normalized_jsonl_path(input_name)
    share_csv = dataset_share_csv_path(input_name)
    dataset = TaxonomyDataset.from_share_csv(share_csv)
    df = load_records(jsonl_path, args.metric, args.group_field)
    if df.empty:
        raise RuntimeError("No data to plot; check --metric and input file.")
    if args.top_labels > 0:
        top_labels = dataset.union_top_labels("mechanism", args.top_labels)
    else:
        top_labels = dataset.top_labels()
    df = df[df["label"].isin(top_labels)]
    nrows = len(top_labels)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(9, 3 * nrows),
        sharex=True,
    )
    if nrows == 1:
        axes = [axes]
    for ax, label in zip(axes, top_labels):
        subset = df[df["label"] == label]
        groups = subset["group"].unique().tolist()

        if args.group_field == "mechanism":
            groups = sort_mechanisms(groups)
        elif args.group_field == "model":
            groups = sort_agents(groups)
        elif args.group_field == "game":
            groups = sort_games(groups)
        else:
            groups = sorted(groups)

        data = [subset[subset["group"] == g]["metric"].values for g in groups]
        ax.violinplot(data, showmeans=True, showextrema=False)
        ax.set_xticks(range(1, len(groups) + 1))

        if args.group_field == "mechanism":
            display_groups = [format_mechanism_name(str(g)) for g in groups]
        elif args.group_field == "model":
            display_groups = [format_model_name(str(g)) for g in groups]
        else:
            display_groups = [str(g) for g in groups]

        ax.set_xticklabels(display_groups, rotation=25, ha="right")
        ax.set_ylabel(label)
    axes[-1].set_xlabel(args.metric.replace("_", " ").title())
    fig.tight_layout()
    density_root = prepare_figure_subdir(input_name, "taxonomy_label_density")
    output_prefix = (
        density_root
        / f"taxonomy_label_density_{args.group_field}_{args.metric}_{args.mode}"
    )
    saved_paths = save_matplotlib_figure(
        fig,
        output_prefix,
        ("png",),
        dpi=300,
        root_dir=density_root,
    )
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
