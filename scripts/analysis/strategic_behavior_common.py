#!/usr/bin/env python3
"""Shared helpers for strategic behavior figure scripts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def stacked_barh_with_se(
    ax: Any,
    y: float,
    run_pcts: dict[str, list[float]],
    ordered_actions: Sequence[str],
    action_colors: dict[str, Any],
) -> None:
    """Draw a stacked horizontal bar with segment-level standard errors."""

    left = 0.0
    for action in ordered_actions:
        pcts = run_pcts.get(action, [])
        if not pcts:
            continue

        mean_pct = float(np.mean(pcts))
        se_pct = (
            float(np.std(pcts, ddof=1) / np.sqrt(len(pcts)))
            if len(pcts) > 1
            else 0.0
        )
        if mean_pct <= 0:
            continue

        ax.barh(
            y,
            mean_pct,
            height=0.8,
            left=left,
            color=action_colors[action],
            edgecolor="none",
        )
        if se_pct > 0:
            ax.errorbar(
                left + mean_pct,
                y,
                xerr=se_pct,
                fmt="none",
                color="black",
                capsize=2,
                linewidth=0.8,
                alpha=0.7,
            )
        left += mean_pct
