#!/usr/bin/env python3
"""Mechanism → Game → Label Sankey/alluvial visualization (Plotly)."""

import argparse

import plotly.graph_objects as go

from coopeval.llm_judge.plotting_utils import (
    dataset_share_csv_path,
    normalized_jsonl_path,
    prepare_figure_subdir,
    validate_input_name,
)
from coopeval.llm_judge.taxonomy_dataset import TaxonomyDataset
from coopeval.script_utils.display_helper import (
    format_mechanism_name,
    sort_games,
    sort_mechanisms,
)
from coopeval.script_utils.llm_judge_helpers import classification_labels
from coopeval.utils.json_io import iter_jsonl
from coopeval.script_utils.figure_exports import save_plotly_figure


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the mechanism→game→label Sankey plot."""
    parser = argparse.ArgumentParser(
        description="Mechanism → Game → Label Sankey diagram."
    )
    parser.add_argument(
        "input_name",
        type=str,
        help="Judge run identifier (auto-discovers normalized JSONL).",
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=12,
        help=(
            "Number of top labels per mechanism to union together "
            "(default: 12)."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "html"],
        default=["png"],
        help="Formats to export.",
    )
    args = parser.parse_args()
    args.input_name = validate_input_name(args.input_name)
    return args


def main() -> None:
    """CLI entry point for generating the taxonomy Sankey diagram."""

    args = parse_args()
    input_name = args.input_name
    jsonl_path = normalized_jsonl_path(input_name)
    share_csv = dataset_share_csv_path(input_name)
    dataset = TaxonomyDataset.from_share_csv(share_csv)
    counts = {}
    for row in iter_jsonl(jsonl_path):
        labels = classification_labels(row)
        if not labels:
            continue
        mech = row.get("mechanism", "Unknown")
        game = row.get("game", "Unknown")
        for label in labels:
            counts[(mech, game, label)] = counts.get((mech, game, label), 0) + 1
    if args.top_labels > 0:
        label_order = dataset.union_top_labels("mechanism", args.top_labels)
    else:
        label_order = dataset.top_labels()
    keep_labels = set(label_order)
    counts = {k: v for k, v in counts.items() if k[2] in keep_labels}
    mechanisms = sort_mechanisms({k[0] for k in counts})
    games = sort_games({k[1] for k in counts})
    labels = [label for label in label_order if label in keep_labels]
    nodes = mechanisms + games + labels
    display_nodes = [
        format_mechanism_name(str(n)) if n in mechanisms else str(n)
        for n in nodes
    ]
    index = {name: idx for idx, name in enumerate(nodes)}
    sources = []
    targets = []
    values = []
    for (mech, game, label), value in counts.items():
        sources.append(index[mech])
        targets.append(index[game])
        values.append(value)
        sources.append(index[game])
        targets.append(index[label])
        values.append(value)
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=display_nodes),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig.update_layout(title="Mechanism → Game → Label Flow")
    sankey_root = prepare_figure_subdir(input_name, "taxonomy_sankey")
    output_prefix = sankey_root / "taxonomy_sankey"
    saved_paths = save_plotly_figure(
        fig, output_prefix, args.formats, root_dir=sankey_root
    )
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
