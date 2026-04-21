#!/usr/bin/env python3
"""Aggregate table builder with pluggable formatters."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from coopeval.script_utils.display_helper import (
    format_mechanism_name,
    format_model_name,
)
from coopeval.visualization.analysis_utils import is_reputation_mechanism
from .formatters import AggregateFormatter, CellStats
from .metrics import (
    LLM_AVERAGE,
    METRIC_LABELS,
    compute_aggregate_cell_raw,
    compute_aggregate_color_value,
)


class AggregateTableBuilder:
    """Generate aggregate LaTeX tables using a formatter strategy."""

    def __init__(self, formatter: AggregateFormatter) -> None:
        self.formatter = formatter

    def build(
        self,
        mechanisms: Sequence[str],
        models: Sequence[str],
        game_configs: dict[str, dict],
        metrics: Sequence[str],
        show_stderr: bool,
        source_folders: Sequence[Path] | None = None,
        aggregate_games: Sequence[str] | None = None,
        normalize: bool = True,
    ) -> str:
        lines: list[str] = []

        if source_folders:
            lines.append("% Source folders:")
            for folder in source_folders:
                lines.append(f"%   {folder}")

        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(self.formatter.caption)
        lines.append(r"\label{tab:aggregate_results}")

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

        for mech_idx, mechanism in enumerate(mechanisms):
            metric_list = self._metrics_for_mechanism(mechanism, metrics)
            if not metric_list:
                continue

            display_name = format_mechanism_name(mechanism)
            for metric_idx, metric in enumerate(metric_list):
                row_parts: list[str] = []
                row_parts.append(
                    self._format_mechanism_cell(
                        display_name, metric_idx, len(metric_list)
                    )
                )
                row_parts.append(METRIC_LABELS[metric])

                avg_mean, avg_stderr = compute_aggregate_cell_raw(
                    mechanism,
                    metric,
                    LLM_AVERAGE,
                    game_configs,
                    aggregate_games,
                    normalize=normalize,
                )
                avg_stats = CellStats(
                    -1,
                    avg_mean,
                    avg_stderr,
                    compute_aggregate_color_value(
                        mechanism,
                        metric,
                        LLM_AVERAGE,
                        game_configs,
                        aggregate_games,
                    ),
                )
                row_parts.append(
                    self.formatter.format_average_cell(
                        mechanism, metric, avg_stats, show_stderr
                    )
                )

                model_cells: list[CellStats] = []
                for idx, model in enumerate(models):
                    mean_val, stderr_val = compute_aggregate_cell_raw(
                        mechanism,
                        metric,
                        model,
                        game_configs,
                        aggregate_games,
                        normalize=normalize,
                    )
                    model_cells.append(CellStats(idx, mean_val, stderr_val))

                row_parts.extend(
                    self.formatter.format_model_cells(
                        mechanism, metric, model_cells, show_stderr
                    )
                )
                lines.append(" & ".join(row_parts) + r" \\")

            if mech_idx < len(mechanisms) - 1:
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")
        lines.append(r"\end{table*}")
        return "\n".join(lines)

    @staticmethod
    def _metrics_for_mechanism(
        mechanism: str, requested: Sequence[str]
    ) -> list[str]:
        if is_reputation_mechanism(mechanism):
            return ["mean"] if "mean" in requested else []
        return list(requested)

    @staticmethod
    def _format_mechanism_cell(
        display_name: str, metric_idx: int, metric_count: int
    ) -> str:
        if metric_count <= 1:
            return f"\\textbf{{{display_name}}}"
        if metric_idx == 0:
            return (
                f"\\multirow{{{metric_count}}}{{*}}"
                f"{{\\textbf{{{display_name}}}}}"
            )
        return ""
