#!/usr/bin/env python3
"""Per-game LaTeX table generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from coopeval.script_utils.display_helper import (
    format_mechanism_name,
    format_model_name,
)
from coopeval.visualization.analysis_utils import (
    NormalizeScore,
    is_reputation_mechanism,
)
from .colors import (
    colorize_cell,
    muted_inline,
    muted_text,
)
from .metrics import (
    LLM_AVERAGE,
    METRIC_LABELS,
    compute_per_game_cell_raw,
    raw_data,
)


def format_score_with_stderr(
    mean_val: float,
    stderr_val: float,
    precision: int,
    show_stderr: bool,
    num_samples: int,
    metric: str,
) -> str:
    """Format a score with optional standard error."""
    mean_str, stderr_str = format_mean_and_stderr_separate(
        mean_val,
        stderr_val,
        precision,
        show_stderr and num_samples > 1,
        metric,
    )
    if stderr_str is not None:
        return f"{mean_str}{muted_inline(stderr_str)}"
    return mean_str


def format_mean_and_stderr_separate(
    mean_val: float,
    stderr_val: float,
    precision: int,
    show_stderr: bool,
    metric: str,
) -> tuple[str, str | None]:
    if metric == "dr":
        mean_str = f"{mean_val:.1f}"
        stderr_str = f"$\\pm${stderr_val:.1f}" if show_stderr else None
    else:
        mean_str = f"{mean_val:.{precision}f}"
        stderr_str = (
            f"$\\pm${stderr_val:.{precision}f}" if show_stderr else None
        )
    return mean_str, stderr_str


def apply_ranking_format(
    model_values: Sequence[
        tuple[int, float | None, float | None, str, str | None]
    ],
    metric_type: str,
    tolerance: float = 0.05,
) -> list[str]:
    """Apply ranking-based formatting to model values."""
    valid_entries = [
        (idx, val, stderr, mean_s, stderr_s)
        for idx, val, stderr, mean_s, stderr_s in model_values
        if val is not None
    ]

    if not valid_entries:
        return [
            mean_s if stderr_s is None else f"{mean_s}{muted_inline(stderr_s)}"
            for _, _, _, mean_s, stderr_s in model_values
        ]

    reverse = metric_type == "maximize"
    sorted_entries = sorted(valid_entries, key=lambda x: x[1], reverse=reverse)

    ranks: dict[int, int] = {}
    current_rank = 1
    i = 0
    while i < len(sorted_entries):
        _, current_val, _, _, _ = sorted_entries[i]
        tied_indices = [sorted_entries[i][0]]
        j = i + 1
        while j < len(sorted_entries):
            _, next_val, _, _, _ = sorted_entries[j]
            if current_val != 0:
                rel_diff = abs(next_val - current_val) / abs(current_val)
            else:
                rel_diff = abs(next_val - current_val)
            if rel_diff <= tolerance:
                tied_indices.append(sorted_entries[j][0])
                j += 1
            else:
                break

        for idx in tied_indices:
            ranks[idx] = current_rank

        current_rank += len(tied_indices)
        i = j

    result: list[str] = [""] * len(model_values)
    for idx, val, _stderr, mean_str, stderr_str in model_values:
        if val is None:
            result[idx] = (
                mean_str
                if stderr_str is None
                else f"{mean_str}{muted_inline(stderr_str)}"
            )
            continue

        rank = ranks[idx]
        if rank <= 2:
            formatted_mean = f"\\textbf{{{mean_str}}}"
        elif rank >= len(valid_entries) - 1 and len(valid_entries) >= 5:
            formatted_mean = muted_text(mean_str)
        else:
            formatted_mean = mean_str

        if stderr_str is not None:
            result[idx] = f"{formatted_mean}{muted_inline(stderr_str)}"
        else:
            result[idx] = formatted_mean

    return result


def generate_game_table(
    game: str,
    mechanisms: Sequence[str],
    models: Sequence[str],
    precision: int = 3,
    metrics: Sequence[str] | None = None,
    game_config: dict | None = None,
    source_folders: Sequence[Path] | None = None,
    show_stderr: bool = False,
    colorize_cells: bool = False,
) -> str:
    """Generate a per-game LaTeX table."""
    if metrics is None:
        metrics = ("mean", "rd", "dr")

    lines: list[str] = []
    if source_folders:
        lines.append("% Source folders:")
        for folder in source_folders:
            lines.append(f"%   {folder}")

    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Results for {game}}}")
    game_slug = game.lower().replace(" ", "_")
    lines.append(f"\\label{{tab:{game_slug}}}")

    num_data_cols = len(models)
    col_spec = "ll||r|" + "r" * num_data_cols
    lines.append(r"\scalebox{0.78}{")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header_parts = [
        r"\textbf{Mechanism}",
        r"\textbf{Metric}",
        r"\textbf{LLM Average}",
    ]
    for model in models:
        header_parts.append(format_model_name(model))
    lines.append(" & ".join(header_parts) + r" \\")

    lines.append(r"\midrule")

    normalizer = (
        NormalizeScore(game, game_config) if game_config is not None else None
    )

    for mech_idx, mechanism in enumerate(mechanisms):
        display_name = format_mechanism_name(mechanism)
        is_reputation = is_reputation_mechanism(mechanism)
        metric_list = ["mean"] if is_reputation else metrics

        for metric_idx, metric in enumerate(metric_list):
            row_parts: list[str] = []
            if len(metric_list) == 3:
                if metric_idx == 0:
                    row_parts.append(
                        f"\\multirow{{3}}{{*}}{{\\textbf{{{display_name}}}}}"
                    )
                else:
                    row_parts.append("")
            else:
                if metric_idx == 0 and len(metric_list) > 1:
                    row_parts.append(
                        f"\\multirow{{{len(metric_list)}}}{{*}}{{\\textbf{{{display_name}}}}}"
                    )
                elif metric_idx == 0:
                    row_parts.append(f"\\textbf{{{display_name}}}")
                else:
                    row_parts.append("")

            row_parts.append(METRIC_LABELS[metric])

            mean_val, stderr_val = compute_per_game_cell_raw(
                game, mechanism, metric, LLM_AVERAGE
            )
            num_reps = len(raw_data[game][mechanism])
            average_cell = format_score_with_stderr(
                mean_val,
                stderr_val,
                precision,
                show_stderr,
                num_reps,
                metric,
            )
            color_value = mean_val
            if normalizer is not None and metric in ("mean", "rd"):
                color_value = normalizer.normalize(mean_val)
            row_parts.append(
                colorize_cell(average_cell, color_value, metric)
                if colorize_cells
                else average_cell
            )

            model_values: list[
                tuple[int, float | None, float | None, str, str | None]
            ] = []
            for idx, model in enumerate(models):
                mean_val, stderr_val = compute_per_game_cell_raw(
                    game, mechanism, metric, model
                )
                mean_str, stderr_str = format_mean_and_stderr_separate(
                    mean_val, stderr_val, precision, show_stderr, metric
                )
                model_values.append(
                    (
                        idx,
                        mean_val,
                        stderr_val if show_stderr else None,
                        mean_str,
                        stderr_str,
                    )
                )

            if metric in ("mean", "rd"):
                formatted_values = apply_ranking_format(
                    model_values, "maximize"
                )
            else:
                formatted_values = apply_ranking_format(
                    model_values, "minimize"
                )

            row_parts.extend(formatted_values)
            lines.append(" & ".join(row_parts) + r" \\")

        if mech_idx < len(mechanisms) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)
