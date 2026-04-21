#!/usr/bin/env python3
"""Generate taxonomy heatmaps from the canonical share dataset."""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

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
from coopeval.script_utils.colors import PALETTE_BASE
from coopeval.script_utils.figure_exports import save_matplotlib_figure

BASE_CMAP = LinearSegmentedColormap.from_list("taxonomy_heat", PALETTE_BASE)
HEATMAP_SUBDIR = "taxonomy_heatmaps"


@dataclass
class HeatmapEntry:
    """Data for a single heatmap."""

    slug: str
    title: str
    matrix: np.ndarray
    row_labels: list[str]
    col_labels: list[str]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for taxonomy heatmap generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualize taxonomy prevalence per mechanism, and per-player/game breakdowns, "
            "using the canonical share CSV."
        )
    )
    parser.add_argument(
        "input_name",
        type=str,
        help=(
            "Judge run identifier (same as other llm_judge scripts). "
            "The script auto-loads the canonical taxonomy share CSV from "
            "outputs/judge/<input_name>/dataset/."
        ),
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=0,
        help="Limit taxonomy labels (0 = show all, default).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of models for per-model heatmaps.",
        dest="models",
    )
    parser.add_argument(
        "--players",
        nargs="*",
        dest="models",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--games",
        nargs="*",
        default=None,
        help="Optional subset of games for per-game heatmaps.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI (default: 300).",
    )
    parser.add_argument(
        "--include-number",
        action="store_true",
        help="If set, overlay numeric values on each heatmap cell.",
    )
    args = parser.parse_args()
    args.input_name = validate_input_name(args.input_name)
    return args


def render_heatmap(
    entry: HeatmapEntry,
    output_prefix: Path,
    dpi: int,
    root_dir: Path,
    include_numbers: bool,
) -> None:
    """Render a single heatmap (labels × mechanisms)."""
    data = entry.matrix
    if data.size == 0:
        return
    vmin = 0.0
    vmax = 100.0
    fig_h = max(5.0, 0.4 * data.shape[0] + 1.5)
    fig_w = max(6.0, 0.45 * data.shape[1] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(data, aspect="auto", cmap=BASE_CMAP, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(entry.col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(entry.row_labels, fontsize=8)
    ax.set_title(entry.title, fontsize=12, pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Frequency of presence (%)")
    if include_numbers:
        norm = Normalize(vmin=vmin, vmax=vmax)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                r, g, b, _ = BASE_CMAP(norm(value))
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text_color = "black" if luminance > 0.6 else "white"
                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7,
                )
    saved_paths = save_matplotlib_figure(
        fig,
        output_prefix,
        ("png",),
        dpi=dpi,
        root_dir=root_dir,
    )
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


def render_grid(
    entries: list[HeatmapEntry],
    output_prefix: Path,
    dpi: int,
    root_dir: Path,
    include_numbers: bool,
    ncols: int | None = None,
) -> None:
    """Render a grid of heatmaps (shared colorbar)."""
    if not entries:
        return
    vmin = 0.0
    vmax = 100.0
    cols = min(ncols if ncols is not None else 3, len(entries))
    rows = math.ceil(len(entries) / cols)
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(cols * 4.0 + 2, rows * 3.5 + 1.5),
        constrained_layout=True,
    )
    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    images = []
    for ax, entry in zip(axes_list, entries):
        data = entry.matrix
        if data.size == 0:
            ax.axis("off")
            continue
        im = ax.imshow(
            data, aspect="auto", cmap=BASE_CMAP, vmin=vmin, vmax=vmax
        )
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(
            entry.col_labels, rotation=35, ha="right", fontsize=6
        )
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels(entry.row_labels, fontsize=6)
        ax.set_title(entry.title, fontsize=9, pad=6)
        images.append(im)
    for ax in axes_list[len(entries) :]:
        ax.axis("off")
    if images:
        cbar = fig.colorbar(images[0], ax=axes_list, fraction=0.025, pad=0.02)
        cbar.set_label("Frequency of presence (%)")
    if include_numbers:
        norm = Normalize(vmin=0.0, vmax=vmax)
        for ax, entry in zip(axes_list, entries):
            data = entry.matrix
            if data.size == 0:
                continue
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    value = data[i, j]
                    r, g, b, _ = BASE_CMAP(norm(value))
                    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    text_color = "black" if luminance > 0.6 else "white"
                    ax.text(
                        j,
                        i,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=5,
                    )
    saved_paths = save_matplotlib_figure(
        fig,
        output_prefix,
        ("png",),
        dpi=dpi,
        root_dir=root_dir,
    )
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


def build_entries_for_dimension(
    dataset: TaxonomyDataset,
    *,
    label_order: list[str],
    dimension: str,
    title_prefix: str,
    filters: Iterable[str] | None,
) -> list[HeatmapEntry]:
    """Create heatmap entries for each requested model or game group."""
    df = dataset.df
    unique_values = df[dimension].unique().tolist()
    if filters is not None:
        allowed = set(filters)
        unique_values = [value for value in unique_values if value in allowed]

    if dimension == "model":
        unique_values = sort_agents(unique_values)
    elif dimension == "game":
        unique_values = sort_games(unique_values)

    entries: list[HeatmapEntry] = []
    for value in unique_values:
        if dimension == "model":
            subset = dataset.filter(models=[value])
            display_value = format_model_name(str(value))
        elif dimension == "game":
            subset = dataset.filter(games=[value])
            display_value = str(value)
        else:
            continue
        matrix = subset.matrix("mechanism", labels=label_order)
        if matrix.empty:
            continue

        sorted_cols = sort_mechanisms(matrix.columns)
        matrix = matrix[sorted_cols]

        if dimension == "model":
            title = f"Justifications by {display_value}"
        elif dimension == "game":
            title = f"Justifications in {display_value}"
        else:
            title = f"{title_prefix}: {display_value}"

        entries.append(
            HeatmapEntry(
                slug=f"{dimension}_{value.replace('/', '_').replace(' ', '_')}",
                title=title,
                matrix=matrix.to_numpy(),
                row_labels=matrix.index.tolist(),
                col_labels=[format_mechanism_name(str(c)) for c in sorted_cols],
            )
        )
    return entries


def generate_latex_file(
    output_dir: Path,
    overall_prefix: Path,
    models_overview_prefix: Path,
    games_overview_prefix: Path,
) -> None:
    """Generate a LaTeX file with the three overview taxonomy heatmap figures."""

    def png_relpath(prefix: Path) -> str:
        rel = prefix.relative_to(output_dir)
        return f"{rel}.png"

    plots = [
        (
            "overall",
            overall_prefix,
            0.8,
            "Heatmap of how often, on average, each justification category (y-axis) is present in the reasoning behind an LLM model's decision under each mechanism (x-axis). Aggregated across all models and social dilemmas.",
        ),
        (
            "models_overview",
            models_overview_prefix,
            1,
            "Heatmap of how often, on average, each justification category (y-axis) is present in the reasoning behind an LLM model's decision under each mechanism (x-axis), broken down by LLM model.",
        ),
        (
            "games_overview",
            games_overview_prefix,
            1,
            "Heatmap of how often, on average, each justification category (y-axis) is present in the reasoning behind an LLM model's decision under each mechanism (x-axis), broken down by game.",
        ),
    ]

    latex_path = output_dir / "png" / "taxonomy_heatmaps.tex"
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    with latex_path.open("w", encoding="utf-8") as f:
        f.write("% Taxonomy Heatmap Visualizations\n")
        f.write("% Generated automatically\n\n")
        for label, prefix, width, caption in plots:
            f.write("\\begin{figure}[htbp]\n")
            f.write("    \\centering\n")
            f.write(
                f"    \\includegraphics[width={width}\\textwidth]{{judge/{png_relpath(prefix)}}}\n"
            )
            f.write(f"    \\caption{{{caption}}}\n")
            f.write(f"    \\label{{judge:heatmap_{label}}}\n")
            f.write("\\end{figure}\n\n")

    print(f"\nGenerated LaTeX file: {latex_path}")


def main() -> None:
    """CLI entry point for rendering taxonomy heatmaps."""
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

    output_dir = prepare_figure_subdir(input_name, HEATMAP_SUBDIR)
    heatmap_root = output_dir
    heatmap_root.mkdir(parents=True, exist_ok=True)

    mech_matrix = dataset.matrix("mechanism", labels=label_order)
    sorted_mech_cols = sort_mechanisms(mech_matrix.columns)
    mech_matrix = mech_matrix[sorted_mech_cols]
    overall_entry = HeatmapEntry(
        slug="mechanism_overall",
        title="Average Frequency of Justification",
        matrix=mech_matrix.to_numpy(),
        row_labels=mech_matrix.index.tolist(),
        col_labels=[format_mechanism_name(str(c)) for c in sorted_mech_cols],
    )
    render_heatmap(
        overall_entry,
        heatmap_root / f"taxonomy_heatmap_{overall_entry.slug}_frequency",
        args.dpi,
        heatmap_root,
        args.include_number,
    )

    model_dir = heatmap_root / "per_model_mechanism"
    game_dir = heatmap_root / "per_game_mechanism"

    model_entries = build_entries_for_dimension(
        dataset,
        label_order=label_order,
        dimension="model",
        title_prefix="Model",
        filters=args.models,
    )
    for entry in model_entries:
        render_heatmap(
            entry,
            model_dir / f"taxonomy_heatmap_{entry.slug}_frequency",
            args.dpi,
            heatmap_root,
            args.include_number,
        )
    render_grid(
        model_entries,
        heatmap_root / "taxonomy_heatmap_models_overview_frequency",
        args.dpi,
        heatmap_root,
        args.include_number,
    )

    game_entries = build_entries_for_dimension(
        dataset,
        label_order=label_order,
        dimension="game",
        title_prefix="Game",
        filters=args.games,
    )
    for entry in game_entries:
        render_heatmap(
            entry,
            game_dir / f"taxonomy_heatmap_{entry.slug}_frequency",
            args.dpi,
            heatmap_root,
            args.include_number,
        )
    render_grid(
        game_entries,
        heatmap_root / "taxonomy_heatmap_games_overview_frequency",
        args.dpi,
        heatmap_root,
        args.include_number,
        ncols=2,
    )

    generate_latex_file(
        heatmap_root,
        heatmap_root / f"taxonomy_heatmap_{overall_entry.slug}_frequency",
        heatmap_root / "taxonomy_heatmap_models_overview_frequency",
        heatmap_root / "taxonomy_heatmap_games_overview_frequency",
    )

    print("Heatmaps written to:", heatmap_root)


if __name__ == "__main__":
    main()
