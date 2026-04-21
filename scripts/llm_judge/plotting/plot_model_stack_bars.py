#!/usr/bin/env python3
"""Stacked bar plots of taxonomy shares per model, faceted by mechanism/game."""

import argparse
import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

from coopeval.llm_judge.plotting_utils import (
    dataset_share_csv_path,
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
from coopeval.script_utils.figure_exports import save_matplotlib_figure


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for stacked taxonomy bar charts."""
    parser = argparse.ArgumentParser(
        description="Stacked bars of taxonomy shares per model using the canonical dataset."
    )
    parser.add_argument(
        "input_name",
        type=str,
        help="Judge run identifier (auto-discovers dataset share CSV).",
    )
    parser.add_argument(
        "--panel-dimension",
        choices=["mechanism", "game"],
        default="mechanism",
        help="Facet dimension for panels (default: mechanism).",
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=0,
        help=(
            "Number of top labels per mechanism to union together "
            "(0 = all labels, default)."
        ),
    )
    parser.add_argument(
        "--panels",
        nargs="*",
        default=None,
        help="Optional subset of panel names to include.",
    )
    args = parser.parse_args()
    args.input_name = validate_input_name(args.input_name)
    return args


def build_panel_tables(
    dataset: TaxonomyDataset,
    panel_dim: str,
    label_order: list[str],
    panels: list[str] | None,
) -> list[tuple[str, np.ndarray, list[str]]]:
    """Create stacked-bar matrices for each mechanism/game panel."""
    grouped = dataset.aggregate([panel_dim, "model"])
    value_col = "share_pct"
    if panels is not None:
        grouped = grouped[grouped[panel_dim].isin(panels)]

    unique_panels = grouped[panel_dim].unique().tolist()
    if panel_dim == "mechanism":
        unique_panels = sort_mechanisms(unique_panels)
    elif panel_dim == "game":
        unique_panels = sort_games(unique_panels)
    else:
        unique_panels = sorted(unique_panels)

    tables: list[tuple[str, np.ndarray, list[str]]] = []
    for panel in unique_panels:
        subset = grouped[grouped[panel_dim] == panel]
        matrix = (
            subset.pivot(index="model", columns="label", values=value_col)
            .reindex(columns=label_order, fill_value=0.0)
            .fillna(0.0)
        )

        if panel_dim == "mechanism":
            display_panel = format_mechanism_name(str(panel))
        else:
            display_panel = str(panel)

        sorted_models = sort_agents(matrix.index.tolist())
        matrix = matrix.loc[sorted_models]

        display_models = [
            format_model_name(str(m)) for m in matrix.index.tolist()
        ]
        tables.append((display_panel, matrix.to_numpy(), display_models))
    return tables


def main() -> None:
    """CLI entry point for taxonomy stacked bar plots."""
    args = parse_args()
    input_name = args.input_name
    share_csv = dataset_share_csv_path(input_name)
    dataset = TaxonomyDataset.from_share_csv(share_csv)
    if args.top_labels > 0:
        label_order = dataset.union_top_labels("mechanism", args.top_labels)
    else:
        label_order = dataset.top_labels()
    if not label_order:
        raise RuntimeError("No labels found in the share CSV.")

    panel_tables = build_panel_tables(
        dataset,
        args.panel_dimension,
        label_order,
        args.panels,
    )
    if not panel_tables:
        raise RuntimeError("No panels to plot; check --panels filters.")

    cmap = get_cmap("tab20", max(1, len(label_order)))
    ncols = min(2, len(panel_tables))
    nrows = math.ceil(len(panel_tables) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10 * ncols, 4.5 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    global_xlim = max(
        (data.sum(axis=1).max() if data.size else 0.0)
        for _, data, _ in panel_tables
    )
    global_xlim = max(global_xlim, 1.0)
    for ax, (panel, data, models) in zip(axes_flat, panel_tables):
        bottom = np.zeros(len(models))
        for idx, label in enumerate(label_order):
            values = data[:, idx] if data.size else np.zeros(len(models))
            ax.barh(
                models,
                values,
                left=bottom,
                color=cmap(idx),
                label=label,
            )
            bottom += values
        ax.set_title(panel)
        ax.set_xlabel("Share (%)")
        ax.set_xlim(0, global_xlim)
    for ax in axes_flat[len(panel_tables) :]:
        ax.axis("off")

    handles = [
        mpatches.Patch(color=cmap(i), label=label)
        for i, label in enumerate(label_order)
    ]
    fig.subplots_adjust(bottom=0.18)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(4, len(label_order)),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    stacks_root = prepare_figure_subdir(input_name, "taxonomy_model_stacks")
    output_prefix = (
        stacks_root / f"taxonomy_model_stacks_{args.panel_dimension}_frequency"
    )
    saved_paths = save_matplotlib_figure(
        fig,
        output_prefix,
        ("png",),
        dpi=300,
        root_dir=stacks_root,
    )
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
