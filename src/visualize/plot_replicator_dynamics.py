from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.typing import ColorType

from config import FIGURE_DIR
from src.logger_manager import WandBLogger


def _ensure_date_dir() -> Path:
    """
    Create date-based directory under FIGURE_DIR/YYYY/MM/DD and return it.
    """
    now = datetime.now()
    date_path = (
        Path(FIGURE_DIR)
        / now.strftime("%Y")
        / now.strftime("%m")
        / now.strftime("%d")
    )
    date_path.mkdir(parents=True, exist_ok=True)
    return date_path


def _save_local(fig: Figure, filename: str) -> Path:
    """
    Save figure locally under FIGURE_DIR with timestamp prefix.
    Returns the full filepath.
    """
    date_path = _ensure_date_dir()
    timestamp = datetime.now().strftime("%H_%M")
    out_path = date_path / f"{timestamp}_{filename}"
    fig.savefig(out_path, bbox_inches="tight")
    return out_path


def _log_or_save(
    fig: Figure,
    filename: str,
    wb: WandBLogger | None,
    save_local: bool,
) -> Path | None:
    """
    Either log to WandB, save locally, or both.
    """
    local_path = None
    if wb:
        wb.log_figure(
            fig,
            filename,
        )
    if save_local:
        local_path = _save_local(fig, filename)
    plt.close(fig)
    return local_path


def plot_probability_evolution(
    trajectory: Sequence[np.ndarray],
    wb: WandBLogger | None,
    save_local: bool,
    colors: Sequence[ColorType] | None = None,
    labels: Sequence[str] | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> Path | None:
    """
    Stacked area chart of probability evolution.

    Returns local path if saved, otherwise None.
    """
    mat = np.array(trajectory)
    steps, items = mat.shape
    sums = mat.sum(axis=1)
    if not np.allclose(sums, 1, rtol=1e-3):
        print(
            f"Warning: distributions sum to [{sums.min():.4f}, {sums.max():.4f}]"
        )
    cumsum = np.cumsum(mat, axis=1)
    base = np.hstack([np.zeros((steps, 1)), cumsum])
    x = np.arange(steps)
    if colors is None:
        cmap = plt.get_cmap("Set3")
        colors = list(cmap(np.linspace(0, 1, items)))
    if labels is None:
        labels = [f"Item {i}" for i in range(items)]
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(items):
        ax.fill_between(
            x,
            base[:, i],
            base[:, i + 1],
            color=colors[i],
            label=labels[i],
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set(
        xlabel="Time Step",
        ylabel="Population Percentage",
        title="Population Evolution",
        ylim=(0, 1),
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return _log_or_save(fig, "population_evolution.png", wb, save_local)


def plot_share_progression(
    pop_payoff: Any,
    dynamics: Any,
    wb: WandBLogger | None = None,
    save_local: bool = False,
) -> Path | None:
    pop_traj, pay_traj, status = dynamics
    mat = np.array(pop_traj).T
    order = np.argsort(-mat[:, -1])
    mat = mat[order]
    labels = [pop_payoff.agent_types[i] for i in order]
    x = np.arange(mat.shape[1])
    cmap = plt.get_cmap("Set3")
    colors = cmap(np.linspace(0, 1, mat.shape[0]))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.stackplot(x, *mat, colors=colors, labels=labels, alpha=0.7)
    ax1.set(xlabel="Time Step", ylabel="Share", ylim=(0, 1), xlim=(0, x[-1]))
    ax1.axhline(1, linestyle="--", linewidth=1)
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x, pay_traj, color="red", linewidth=3, label="Payoff")
    ax2.set_ylabel("Payoff", color="red")
    ax2.tick_params(labelcolor="red")
    ax1.set_title(f"Share & Payoff (status={status})")
    lines, labs = ax1.get_legend_handles_labels()
    l2, l2_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + l2, labs + l2_labels, bbox_to_anchor=(1.15, 1))
    fig.tight_layout()
    return _log_or_save(fig, "share_progress.png", wb, save_local)


def plot_3simplex_trajectories(
    trajectories: Sequence[Sequence[np.ndarray]],
    wb: WandBLogger | None = None,
    save_local: bool = False,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(10, 8))
    from matplotlib.patches import Polygon

    triangle = Polygon(
        [(0, 0), (1, 0), (0.5, np.sqrt(3) / 2)],
        fill=False,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(triangle)

    def to_xy(p):
        return (p[1] + p[2] / 2, p[2] * np.sqrt(3) / 2)

    colors = plt.get_cmap("Set3")(np.linspace(0, 1, len(trajectories)))
    for i, traj in enumerate(trajectories):
        xs, ys = zip(*[to_xy(*state) for state in traj])
        ax.plot(xs, ys, color=colors[i], alpha=0.7, label=f"Traj {i+1}")
        ax.scatter(
            [xs[0], xs[-1]],
            [ys[0], ys[-1]],
            s=50,
            marker="o" if i == 0 else "s",
            color=colors[i],
        )
    ax.axis("off")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    ax.set_title("3-Simplex Dynamics")
    return _log_or_save(fig, "simplex.png", wb, save_local)
