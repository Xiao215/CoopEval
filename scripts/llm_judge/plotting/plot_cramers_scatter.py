#!/usr/bin/env python3
"""Scatter plot of share gap vs. Cramér's V."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from coopeval.llm_judge.plotting_utils import (
    mechanism_differences_csv_path,
    prepare_figure_subdir,
    validate_input_name,
)
from coopeval.script_utils.figure_exports import save_matplotlib_figure


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Cramér's V scatter plot."""
    parser = argparse.ArgumentParser(
        description="Plot mechanism_label_differences.csv as a scatter."
    )
    parser.add_argument(
        "input_name",
        type=str,
        help=(
            "Judge run identifier (auto-discovers mechanism_label_differences.csv)."
        ),
    )
    args = parser.parse_args()
    args.input_name = validate_input_name(args.input_name)
    return args


def main() -> None:
    """CLI entry point for mechanism discriminativeness scatter plotting."""
    args = parse_args()
    input_name = validate_input_name(args.input_name)
    csv_path = mechanism_differences_csv_path(input_name)
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df["range_pp"],
        df["cramers_v"],
        c=range(len(df)),
        cmap="viridis",
        alpha=0.8,
    )
    ax.set_xlabel("Share gap (percentage points)")
    ax.set_ylabel("Cramér's V")
    ax.set_title("Label Discriminativeness by Mechanism")
    for _, row in df.iterrows():
        ax.annotate(
            row["label"],
            (row["range_pp"], row["cramers_v"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    fig.colorbar(scatter, ax=ax, label="Row index")
    fig.tight_layout()
    scatter_root = prepare_figure_subdir(input_name, "taxonomy_cramers_scatter")
    output_prefix = scatter_root / "taxonomy_cramers_scatter"
    saved_paths = save_matplotlib_figure(
        fig,
        output_prefix,
        ("png",),
        dpi=300,
        root_dir=scatter_root,
    )
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
