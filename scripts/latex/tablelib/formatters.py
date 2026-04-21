#!/usr/bin/env python3
"""Aggregate table formatter implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from coopeval.script_utils.colors import latex_html_color
from .colors import (
    HEATMAP_NEGATIVE_COLOR,
    HEATMAP_POSITIVE_COLOR,
    colorize_cell,
    muted_inline,
)
from .game_table import (
    apply_ranking_format,
    format_mean_and_stderr_separate,
)


@dataclass
class CellStats:
    """Container for a single aggregate cell's raw statistics."""

    index: int
    mean: float | None
    stderr: float | None
    color_value: float | None = None


class AggregateFormatter(ABC):
    """Strategy interface for formatting aggregate table cells."""

    def __init__(self, caption: str) -> None:
        self._caption = caption

    @property
    def caption(self) -> str:
        return self._caption

    @abstractmethod
    def format_average_cell(
        self,
        mechanism: str,
        metric: str,
        cell: CellStats,
        show_stderr: bool,
    ) -> str:
        """Return the formatted string for the LLM-average column."""

    @abstractmethod
    def format_model_cells(
        self,
        mechanism: str,
        metric: str,
        cells: Sequence[CellStats],
        show_stderr: bool,
    ) -> list[str]:
        """Return formatted strings for the per-model columns."""


class DefaultAggregateFormatter(AggregateFormatter):
    """Original normalized formatter (values in [0, 1])."""

    def __init__(
        self,
        precision: int = 3,
        use_color: bool = False,
        normalize: bool = True,
    ) -> None:
        if normalize:
            defect_color = latex_html_color(HEATMAP_NEGATIVE_COLOR)
            coop_color = latex_html_color(HEATMAP_POSITIVE_COLOR)
            caption = (
                r"\caption{Results aggregated from all four social dilemmas. "
                r"Before aggregation, payoffs have been shifted and rescaled "
                r"such that $0$ and $1$ reflect the payoff from everyone "
                rf"defecting ({{\color[HTML]{{{defect_color}}}\rule{{1ex}}{{1ex}}}}) and everyone playing their (most) cooperative action ({{\color[HTML]{{{coop_color}}}\rule{{1ex}}{{1ex}}}}) "
                r"respectively. Stronger and weaker LLM performances are bolded or greyed out. ``\mean{}'' and ``\rd{}'' $(\uparrow)$: Payoffs in "
                r"uniform population or after replicator dynamics. The LLM Average column is weighted by the respective population distributions. ``\dr{}'' "
                r"$(\downarrow)$: Rank obtained from deviation rankings. The "
                r"latter two are not compatible with \REPU{}, since we cannot "
                r"sensibly construct a metagame from \REPU{}.}"
            )
        else:
            caption = (
                r"\caption{Results aggregated from all selected games using raw scores. "
                r"Stronger and weaker LLM performances are bolded or greyed out. "
                r"`\mean{}' and `\rd{}' $(\uparrow)$: Payoffs in uniform population "
                r"or after replicator dynamics, `\dr{}' $(\downarrow)$: Rank "
                r"obtained from deviation rankings. The latter two are not "
                r"compatible with \REPU{}, since we cannot sensibly construct "
                r"a metagame from \REPU{}.}"
            )
        super().__init__(caption)
        self.precision = precision
        self.use_color = use_color

    def format_average_cell(
        self,
        mechanism: str,
        metric: str,
        cell: CellStats,
        show_stderr: bool,
    ) -> str:
        mean_str, stderr_str = format_mean_and_stderr_separate(
            cell.mean or 0.0,
            cell.stderr or 0.0,
            self.precision,
            show_stderr,
            metric,
        )
        if show_stderr and stderr_str:
            content = f"{mean_str}{muted_inline(stderr_str)}"
        else:
            content = mean_str
        if self.use_color:
            color_value = (
                cell.color_value if cell.color_value is not None else cell.mean
            )
            return colorize_cell(content, color_value, metric)
        return content

    def format_model_cells(
        self,
        mechanism: str,
        metric: str,
        cells: Sequence[CellStats],
        show_stderr: bool,
    ) -> list[str]:
        model_values: list[
            tuple[int, float | None, float | None, str, str | None]
        ] = []
        for cell in cells:
            mean_str, stderr_str = format_mean_and_stderr_separate(
                cell.mean or 0.0,
                cell.stderr or 0.0,
                self.precision,
                show_stderr,
                metric,
            )
            model_values.append(
                (
                    cell.index,
                    cell.mean,
                    cell.stderr if show_stderr else None,
                    mean_str,
                    stderr_str,
                )
            )

        metric_type = "maximize" if metric in ("mean", "rd") else "minimize"
        formatted = apply_ranking_format(model_values, metric_type)
        return formatted
