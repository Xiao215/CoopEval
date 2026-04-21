#!/usr/bin/env python3
"""Mechanism → Label treemap (Plotly)."""

import argparse

import pandas as pd
import plotly.express as px

from coopeval.llm_judge.plotting_utils import (
    normalized_jsonl_path,
    prepare_figure_subdir,
    validate_input_name,
)
from coopeval.script_utils.display_helper import format_mechanism_name
from coopeval.script_utils.llm_judge_helpers import classification_labels
from coopeval.utils.json_io import iter_jsonl
from coopeval.script_utils.figure_exports import save_plotly_figure


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the mechanism→label treemap."""
    parser = argparse.ArgumentParser(
        description="Mechanism → Label treemap from normalized JSONL."
    )
    parser.add_argument(
        "input_name",
        type=str,
        help="Judge run identifier (auto-discovers normalized JSONL).",
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
    """CLI entry point for generating the taxonomy treemap."""
    args = parse_args()
    input_name = args.input_name
    jsonl_path = normalized_jsonl_path(input_name)
    rows = []
    for row in iter_jsonl(jsonl_path):
        labels = classification_labels(row)
        if not labels:
            continue
        mech = row.get("mechanism", "Unknown")
        for label in labels:
            rows.append({"mechanism": mech, "label": label})
    if not rows:
        raise RuntimeError("No labels found in input JSONL.")

    df = pd.DataFrame(rows)
    counts = df.groupby(["mechanism", "label"]).size().reset_index(name="count")
    counts["mechanism_total"] = counts.groupby("mechanism")["count"].transform(
        "sum"
    )
    counts["mechanism"] = counts["mechanism"].apply(
        lambda x: format_mechanism_name(str(x))
    )

    counts["share_pct"] = 100.0 * counts["count"] / counts["mechanism_total"]
    fig = px.treemap(
        counts,
        path=["mechanism", "label"],
        values="share_pct",
        hover_data={"share_pct": ":.2f", "count": True},
    )
    fig.update_layout(
        title="Mechanism → Label Treemap (Share % within mechanism)"
    )
    treemap_root = prepare_figure_subdir(input_name, "taxonomy_treemap")
    output_prefix = treemap_root / "taxonomy_treemap"
    saved_paths = save_plotly_figure(
        fig, output_prefix, args.formats, root_dir=treemap_root
    )
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
