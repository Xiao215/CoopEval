"""Shared color constants for CoopEval visualization and reporting."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from matplotlib.colors import LinearSegmentedColormap

PALETTE_BASE: tuple[str, ...] = (
    "#2E0854",
    "#B30000",
    "#FF5722",
    "#FFEB3B",
    "#00E5FF",
)
custom_cmap = LinearSegmentedColormap.from_list("custom", PALETTE_BASE)

# Colormap normalization range: normalized 0.0 = NE payoff, 1.0 = cooperative payoff.
# Extending below/above keeps the visual extremes from being pure dark/bright.
CMAP_VMIN = -1.0
CMAP_VMAX = 1.5

MECHANISM_COLORS: dict[str, str] = {
    "NoMechanism": "#888888",
    "Repetition": "#2196F3",
    "ReputationFirstOrder": "#FFEB3B",
    "Reputation": "#FFC107",
    "Mediation": "#F44336",
    "Contracting": "#4CAF50",
}

_MODEL_COLOR_PALETTE: dict[str, str] = {
    "Claude": "#228833",
    "Gemini-R": "#4477AA",
    "Gemini-B": "#66CCEE",
    "GPT-5.2": "#AA3377",
    "GPT-4o": "#EE6677",
    "Qwen-30b": "#CCBB44",
}
MODEL_COLOR_PALETTE: Mapping[str, str] = MappingProxyType(_MODEL_COLOR_PALETTE)

MUTED_TEXT_COLOR = "black!60"
HEATMAP_NEGATIVE_COLOR = "#0072B2"
HEATMAP_POSITIVE_COLOR = "#E69F00"
PLOT_SEPARATOR_COLOR = "#9E9E9E"


def latex_html_color(hex_color: str) -> str:
    """Return a hex color in the form LaTeX's HTML color model expects."""

    return hex_color.removeprefix("#").upper()


def hex_to_rgba(
    hex_color: str, alpha: float = 1.0
) -> tuple[float, float, float, float]:
    """Convert hex color to RGBA tuple for matplotlib."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return (r, g, b, alpha)
