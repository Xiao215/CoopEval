#!/usr/bin/env python3
"""Label co-occurrence network."""

import argparse
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from coopeval.llm_judge.plotting_utils import (
    normalized_jsonl_path,
    prepare_figure_subdir,
    validate_input_name,
)
from coopeval.script_utils.llm_judge_helpers import classification_labels
from coopeval.utils.json_io import iter_jsonl
from coopeval.script_utils.figure_exports import save_matplotlib_figure


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the label co-occurrence network plot."""
    parser = argparse.ArgumentParser(
        description="Draw a co-occurrence network over taxonomy labels."
    )
    parser.add_argument(
        "input_name",
        type=str,
        help="Judge run identifier (auto-discovers normalized JSONL).",
    )
    parser.add_argument(
        "--min-cooc",
        type=int,
        default=200,
        help="Minimum co-occurrence count to draw an edge (default: 200).",
    )
    args = parser.parse_args()
    args.input_name = validate_input_name(args.input_name)
    return args


def main() -> None:
    """CLI entry point for drawing the taxonomy co-occurrence network."""
    args = parse_args()
    input_name = args.input_name
    jsonl_path = normalized_jsonl_path(input_name)
    cooc = Counter()
    label_counts = Counter()
    for row in iter_jsonl(jsonl_path):
        labels = classification_labels(row)
        if not labels:
            continue
        uniq = sorted(set(labels))
        label_counts.update(uniq)
        for a, b in combinations(uniq, 2):
            cooc[(a, b)] += 1
    edges = [(a, b, w) for (a, b), w in cooc.items() if w >= args.min_cooc]
    if not edges:
        raise RuntimeError("No edges meet the --min-cooc threshold.")
    G = nx.Graph()
    for a, b, w in edges:
        G.add_edge(a, b, weight=w)
    for label, count in label_counts.items():
        if label in G:
            G.nodes[label]["size"] = count
    pos = nx.spring_layout(G, seed=42, weight="weight")
    sizes = np.array([G.nodes[n].get("size", 1) for n in G.nodes()])
    fig, ax = plt.subplots(figsize=(10, 8))
    edge_widths = [G[u][v]["weight"] / args.min_cooc for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_widths,
        edge_color="#888888",
        alpha=0.25,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=40 + (sizes / sizes.max()) * 320,
        node_color="tab:blue",
    )
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=8,
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5),
    )
    ax.set_axis_off()
    ax.set_title("Taxonomy Label Co-occurrence Network")
    fig.tight_layout()
    network_root = prepare_figure_subdir(input_name, "taxonomy_label_network")
    output_prefix = network_root / "taxonomy_label_network"
    saved_paths = save_matplotlib_figure(
        fig,
        output_prefix,
        ("png",),
        dpi=300,
        root_dir=network_root,
    )
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
