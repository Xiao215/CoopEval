#!/usr/bin/env python3
"""Shared LaTeX heatmap helpers for table cells."""

from __future__ import annotations

from dataclasses import dataclass

from coopeval.script_utils.colors import (
    HEATMAP_NEGATIVE_COLOR,
    HEATMAP_POSITIVE_COLOR,
    MUTED_TEXT_COLOR,
)


@dataclass(frozen=True)
class LatexColor:
    """A LaTeX color specification for xcolor/colortbl."""

    value: str
    model: str | None = None


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_value = hex_color.removeprefix("#")
    return tuple(int(hex_value[idx : idx + 2], 16) for idx in range(0, 6, 2))


def _blend_with_white(hex_color: str, intensity: int) -> str:
    """Blend a palette color toward white to mimic `color!<n>`."""
    mix = max(0, min(intensity, 100)) / 100.0
    base_rgb = _hex_to_rgb(hex_color)
    blended = tuple(
        round(255 * (1.0 - mix) + channel * mix) for channel in base_rgb
    )
    return "".join(f"{channel:02X}" for channel in blended)


def cell_color(value: float | None, metric: str) -> LatexColor | None:
    """Return the heatmap color for a metric cell."""
    if value is None:
        return None

    if metric == "dr":
        t = (3.5 - value) / 2.5
    else:
        t = (value - 0.5) * 2

    t = max(min(t, 1.0), -1.0)
    if abs(t) < 0.04:
        return None

    intensity = int(round(abs(t) * 50))
    if intensity < 3:
        return None

    base_color = HEATMAP_POSITIVE_COLOR if t > 0 else HEATMAP_NEGATIVE_COLOR
    return LatexColor(_blend_with_white(base_color, intensity), "HTML")


def prepend_cellcolor(cell_str: str, color: LatexColor | None) -> str:
    """Prefix a LaTeX cell with a background color when available."""
    if color is None:
        return cell_str
    if color.model is None:
        return f"\\cellcolor{{{color.value}}} {cell_str}"
    return f"\\cellcolor[{color.model}]{{{color.value}}} {cell_str}"


def colorize_cell(cell_str: str, value: float | None, metric: str) -> str:
    """Apply the shared heatmap background to a formatted cell string."""
    return prepend_cellcolor(cell_str, cell_color(value, metric))


def muted_text(text: str) -> str:
    """Format de-emphasized text consistently with other muted content."""
    return f"\\textcolor{{{MUTED_TEXT_COLOR}}}{{{text}}}"


def muted_inline(text: str) -> str:
    """Format inline secondary text without affecting surrounding content."""
    return f"{{\\scriptsize\\color{{{MUTED_TEXT_COLOR}}}{text}}}"
