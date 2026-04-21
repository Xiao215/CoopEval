#!/usr/bin/env python3
"""Radar charts for taxonomy category shares overall and per-model."""

import argparse
import math
import textwrap
from pathlib import Path
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

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
    sort_mechanisms,
)
from coopeval.script_utils.colors import MECHANISM_COLORS, MODEL_COLOR_PALETTE
from coopeval.script_utils.figure_exports import save_matplotlib_figure

_LABEL_OVERRIDES: dict[str, str] = {
    "risk aversion": "Risk\nAversion",
    "individual utility maximization": "Individual Utility\nMaximization",
}

_DEFAULT_RADAR_LINESTYLE = (0, (2.4, 2.0))
_REPUTATION_PLUS_LINESTYLE = (0, (1.0, 1.0))
_REPUTATION_MINUS_LINESTYLE = (0, (4.0, 4.0))


def _format_axis_label(label: str) -> str:
    """Apply title case and special-case line breaks to a taxonomy label."""
    override = _LABEL_OVERRIDES.get(label.lower())
    if override is not None:
        return override
    return "\n".join(textwrap.wrap(label.title(), width=16))


def _radar_linestyle(entry: str) -> tuple[int, tuple[float, ...]]:
    """Use dotted radar outlines, with distinct dot spacing for reputation variants."""
    base_entry = str(entry).split(" (", 1)[0]
    if base_entry == "Reputation":
        return _REPUTATION_PLUS_LINESTYLE
    if base_entry == "ReputationFirstOrder":
        return _REPUTATION_MINUS_LINESTYLE
    return _DEFAULT_RADAR_LINESTYLE


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for generating radar charts."""
    parser = argparse.ArgumentParser(
        description="Plot overall and per-model taxonomy radar charts."
    )
    parser.add_argument(
        "input_name",
        type=str,
        help="Judge run identifier (auto-discovers dataset share CSV).",
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=0,
        help="Max taxonomy labels to include (0 = all, default).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of models to plot.",
    )
    args = parser.parse_args()
    args.input_name = validate_input_name(args.input_name)
    return args


def build_mechanism_matrix(
    dataset: TaxonomyDataset, label_order: list[str]
) -> pd.DataFrame:
    """Aggregate label counts or shares at the mechanism level."""
    grouped = dataset.aggregate(["mechanism"])
    matrix = grouped.pivot(
        index="mechanism", columns="label", values="share_pct"
    ).fillna(0.0)
    matrix = matrix.reindex(columns=label_order, fill_value=0.0)

    sorted_mechs = sort_mechanisms(matrix.index)
    matrix = matrix.loc[sorted_mechs]
    return matrix


def build_model_matrix(
    dataset: TaxonomyDataset, label_order: list[str]
) -> pd.DataFrame:
    """Aggregate label shares at the model level."""
    grouped = dataset.aggregate(["model"])
    matrix = grouped.pivot(
        index="model", columns="label", values="share_pct"
    ).fillna(0.0)
    matrix = matrix.reindex(columns=label_order, fill_value=0.0)
    sorted_models = sort_agents(matrix.index.tolist())
    matrix = matrix.loc[sorted_models]
    return matrix


def plot_radar(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    root_dir: Path,
    subtitle: str | None = None,
    palette: dict[str, str] | None = None,
    label_formatter: Callable[[str], str] | None = None,
    palette_key_fn: Callable[[str], str] | None = None,
) -> None:
    """Draw a single radar chart and write it to disk."""
    labels = matrix.columns.tolist()
    if len(labels) < 3:
        raise RuntimeError(
            "Need at least three taxonomy labels for a radar chart."
        )
    num_vars = len(labels)
    angles = np.linspace(0, 2 * math.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        subplot_kw={"projection": "polar"}, figsize=(7.5, 7.5)
    )
    _draw_radar(
        ax,
        matrix,
        angles,
        labels,
        linewidth=2.0,
        labelsize=9,
        legend=True,
        palette=palette,
        label_formatter=label_formatter,
        palette_key_fn=palette_key_fn,
    )
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=12, pad=18)
    else:
        ax.set_title(title, pad=18)
    fig.tight_layout()
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


def _draw_radar(
    ax: Axes,
    matrix: pd.DataFrame,
    angles: list[float],
    labels: list[str],
    linewidth: float,
    labelsize: int,
    legend: bool,
    palette: dict[str, str] | None = None,
    label_formatter: Callable[[str], str] | None = None,
    palette_key_fn: Callable[[str], str] | None = None,
) -> None:
    """Render a radar polygon for each mechanism onto an existing axis."""
    colors = []
    default_cmap = matplotlib.colormaps["tab10"]
    for i, entry in enumerate(matrix.index.tolist()):
        palette_key = palette_key_fn(entry) if palette_key_fn else entry
        if palette and palette_key in palette:
            colors.append(palette[palette_key])
        else:
            colors.append(default_cmap(i / max(1, len(matrix))))
    y_max = 100.0
    ticks = np.linspace(y_max / 4.0, y_max, num=4)
    labels_y = [f"{tick:.0f}%" for tick in ticks]
    ax.set_theta_zero_location("N")  # type: ignore[attr-defined]
    ax.set_theta_direction(-1)  # type: ignore[attr-defined]
    ax.set_ylim(0, y_max)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels_y, fontsize=labelsize - 1)
    ax.set_rlabel_position(290)  # type: ignore[attr-defined]
    for color, entry in zip(colors, matrix.index.tolist()):
        values = matrix.loc[entry].tolist()
        values += values[:1]
        label_text = (
            label_formatter(entry)
            if label_formatter
            else format_mechanism_name(str(entry))
        )
        ax.plot(
            angles,
            values,
            linewidth=linewidth,
            linestyle=_radar_linestyle(str(entry)),
            label=label_text,
            color=color,
        )
        ax.fill(angles, values, color=color, alpha=0.12)
    wrapped = [_format_axis_label(label) for label in labels]
    ax.set_xticks(np.linspace(0, 2 * math.pi, len(labels), endpoint=False))
    ax.set_xticklabels(wrapped, fontsize=labelsize)
    ax.tick_params(axis="x", pad=15)
    if legend:
        ax.legend(bbox_to_anchor=(1.2, 1.0), loc="upper left", fontsize=8)


def plot_radar_grid(
    entries: list[tuple[str, pd.DataFrame]],
    label_order: list[str],
    output_path: Path,
    root_dir: Path,
    title_fn: Callable[[str], str],
    palette: dict[str, str] | None = None,
    label_formatter: Callable[[str], str] | None = None,
    palette_key_fn: Callable[[str], str] | None = None,
    legend_side: str = "bottom",
) -> None:
    """Render small-multiple per-model radar grid plus a shared legend."""
    if not entries:
        return
    n = len(entries)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        subplot_kw={"projection": "polar"},
        figsize=(ncols * 4.2, nrows * 4.2),
    )
    axes_flat = np.array(axes).reshape(-1)
    for ax, (model, matrix) in zip(axes_flat, entries):
        matrix = matrix.reindex(columns=label_order, fill_value=0.0)
        num_vars = len(label_order)
        angles = np.linspace(0, 2 * math.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        _draw_radar(
            ax,
            matrix,
            angles,
            label_order,
            linewidth=1.4,
            labelsize=7,
            legend=False,
            palette=palette,
            label_formatter=label_formatter,
            palette_key_fn=palette_key_fn,
        )
        ax.set_title(title_fn(model), fontsize=9, pad=12)
    for ax in axes_flat[len(entries) :]:
        ax.remove()
    mechanisms = entries[0][1].index.tolist()
    palette_for_handles = palette or MECHANISM_COLORS
    default_cmap = matplotlib.colormaps["tab10"]
    handles = []
    for i, mech in enumerate(mechanisms):
        palette_key = palette_key_fn(mech) if palette_key_fn else mech
        color = palette_for_handles.get(
            palette_key, default_cmap(i / max(1, len(mechanisms)))
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=_radar_linestyle(str(mech)),
                label=(
                    label_formatter(mech)
                    if label_formatter
                    else format_mechanism_name(str(mech))
                ),
                linewidth=1.8,
            )
        )
    if legend_side == "right":
        fig.tight_layout(h_pad=4.0, w_pad=3.0)
        fig.subplots_adjust(top=0.9, right=0.80)
        fig.legend(
            handles=handles,
            loc="center left",
            ncol=1,
            fontsize=8,
            bbox_to_anchor=(0.82, 0.5),
        )
    else:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(4, len(handles)),
            fontsize=8,
            bbox_to_anchor=(0.5, 0.02),
        )
        fig.tight_layout(h_pad=4.0, w_pad=3.0)
        fig.subplots_adjust(top=0.9, bottom=0.08)
    saved_paths = save_matplotlib_figure(
        fig, output_path, ("png",), dpi=300, root_dir=root_dir
    )
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved: {path}")


def main() -> None:
    """CLI entry point for taxonomy radar generation."""
    args = parse_args()
    input_name = args.input_name
    share_csv = dataset_share_csv_path(input_name)
    dataset = TaxonomyDataset.from_share_csv(share_csv)
    if args.top_labels > 0:
        label_order = dataset.union_top_labels("mechanism", args.top_labels)
    else:
        label_order = dataset.top_labels()
    if len(label_order) < 3:
        raise RuntimeError(
            "Need at least three labels to draw the radar charts."
        )

    radar_root = prepare_figure_subdir(input_name, "taxonomy_radar")

    overall_matrix = build_mechanism_matrix(dataset, label_order)
    plot_radar(
        overall_matrix,
        "Justification Profile per Mechanism",
        radar_root / "taxonomy_radar_mechanism_frequency",
        radar_root,
        palette=MECHANISM_COLORS,
        label_formatter=format_mechanism_name,
    )

    df = dataset.df
    models = df["model"].unique().tolist()
    if args.models:
        allowed = set(args.models)
        models = [m for m in models if m in allowed]
    models = sort_agents(models)
    per_model_dir = radar_root / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    model_entries: list[tuple[str, pd.DataFrame]] = []
    for model in models:
        subset = dataset.filter(models=[model])
        matrix = build_mechanism_matrix(subset, label_order)
        if matrix.replace(0.0, np.nan).isna().all().all():
            continue
        slug = model.replace("/", "_").replace(" ", "_")
        model_entries.append((model, matrix))
        model_display = format_model_name(str(model))
        plot_radar(
            matrix,
            f"Justification by {model_display}",
            per_model_dir / f"taxonomy_radar_{slug}_frequency",
            radar_root,
            subtitle=None,
            palette=MECHANISM_COLORS,
            label_formatter=format_mechanism_name,
            palette_key_fn=lambda mech: str(mech),
        )
    if model_entries:
        plot_radar_grid(
            model_entries,
            label_order,
            radar_root / "taxonomy_radar_models_overview_frequency",
            radar_root,
            title_fn=lambda m: f"Justification by {format_model_name(str(m))}",
            palette=MECHANISM_COLORS,
            label_formatter=format_mechanism_name,
            palette_key_fn=lambda mech: str(mech),
        )

    mech_entries: list[tuple[str, pd.DataFrame]] = []
    mechanisms = df["mechanism"].unique().tolist()
    mechanisms = sort_mechanisms(mechanisms)
    per_mechanism_dir = radar_root / "per_mechanism"
    per_mechanism_dir.mkdir(parents=True, exist_ok=True)
    for mechanism in mechanisms:
        subset = dataset.filter(mechanisms=[mechanism])
        matrix = build_model_matrix(subset, label_order)
        if matrix.empty or matrix.replace(0.0, np.nan).isna().all().all():
            continue
        mech_entries.append((mechanism, matrix))
        mech_display = format_mechanism_name(str(mechanism))
        slug = mechanism.replace("/", "_").replace(" ", "_")
        plot_radar(
            matrix,
            f"Justification Profile in {mech_display}",
            per_mechanism_dir / f"taxonomy_radar_{slug}_frequency",
            radar_root,
            palette=MODEL_COLOR_PALETTE,
            label_formatter=format_model_name,
            palette_key_fn=format_model_name,
        )
    if mech_entries:
        plot_radar_grid(
            mech_entries,
            label_order,
            radar_root / "taxonomy_radar_mechanisms_overview_frequency",
            radar_root,
            title_fn=lambda mech: (
                f"Justification Profile in {format_mechanism_name(str(mech))}"
            ),
            palette=MODEL_COLOR_PALETTE,
            label_formatter=format_model_name,
            palette_key_fn=format_model_name,
            legend_side="right",
        )


if __name__ == "__main__":
    main()
