from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple, List, Iterable
import textwrap
import re

import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns

# Soft palette tuned for reports
PALETTE_BASE = ["#355070", "#6D597A", "#B56576", "#E56B6F", "#EAAC8B", "#5C7AEA"]
COLOR_PALETTE = {
    "primary": "#355070",
    "secondary": "#6D597A",
    "accent1": "#E56B6F",
    "accent2": "#5C7AEA",
    "accent3": "#0EAD69",
    "muted": "#9AA5B1",
    "background": "#F6F7FB",
    "panel": "#FFFFFF",
    "grid": "#D8DEE9",
    "text": "#2F3437",
}

custom_cmap = LinearSegmentedColormap.from_list("custom", PALETTE_BASE)
coop_cmap = sns.light_palette(COLOR_PALETTE["accent3"], as_cmap=True)

sns.set_theme(style="whitegrid", context="talk", palette=PALETTE_BASE)

def setup_plot_style():
    """Configure global plot styling"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": COLOR_PALETTE["grid"],
        "axes.labelweight": "semibold",
        "axes.titleweight": "bold",
        "axes.facecolor": COLOR_PALETTE["panel"],
        "figure.facecolor": COLOR_PALETTE["background"],
        "axes.labelcolor": COLOR_PALETTE["text"],
        "axes.titlecolor": COLOR_PALETTE["text"],
        "xtick.color": COLOR_PALETTE["text"],
        "ytick.color": COLOR_PALETTE["text"],
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "grid.color": COLOR_PALETTE["grid"],
    })

setup_plot_style()


def _style_axes(ax, *, facecolor: str | None = None) -> None:
    if facecolor is None:
        facecolor = COLOR_PALETTE["panel"]
    ax.set_facecolor(facecolor)
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_PALETTE["grid"])


def _finalize(fig, ax) -> None:
    fig.patch.set_facecolor(COLOR_PALETTE["background"])
    _style_axes(ax)
    fig.tight_layout()


def _slugify(name: str) -> str:
    base = re.sub(r"[\\/:]+", "-", name)
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base


def _clean_agent_label(agent: str, *, wrap: bool = False, width: int = 12) -> str:
    """Shorten agent identifier for plotting axes."""
    base = agent.replace("(CoT)", "")
    base = base.split("/")[-1]
    base = base.replace("_", "-")
    if wrap and len(base) > width:
        return "\n".join(textwrap.wrap(base, width=width))
    return base


def _format_action_key(prefix: str, label: str) -> str:
    clean = label.lower().replace(" ", "_")
    return f"{prefix}_{clean}"

def add_watermark(ax, text="Agent Tournament Analysis"):
    """Add a subtle watermark to the plot"""
    ax.text(0.99, 0.01, text, transform=ax.transAxes, fontsize=8,
            color='gray', alpha=0.5, ha='right', va='bottom')

def improve_legend(ax, **kwargs):
    """Improve legend styling"""
    legend = ax.legend(frameon=True, fancybox=True, shadow=True,
                      framealpha=0.9, edgecolor='gray', **kwargs)
    if legend:
        legend.get_frame().set_facecolor('white')
    return legend

def set_title_with_subtitle(ax, title, subtitle=None, **kwargs):
    """Set a title with optional subtitle"""
    if subtitle:
        full_title = f"{title}\n{subtitle}"
    else:
        full_title = title
    ax.set_title(full_title, fontweight='bold', **kwargs)

def add_grid_style(ax, alpha=0.3):
    """Add consistent grid styling"""
    ax.grid(True, alpha=alpha, linewidth=0.5, color='gray')
    ax.set_axisbelow(True)


def plot_mechanism_effectiveness(csv_path: str | Path, out_dir: str | Path) -> None:
    # Read rows
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return

    # Aggregate mean cooperation by (game, mechanism)
    sums: Dict[Tuple[str, str], float] = {}
    counts: Dict[Tuple[str, str], int] = {}
    for r in rows:
        game = (r.get("game") or "").strip()
        mech = (r.get("mechanism") or "").strip()
        if not game or not mech:
            continue
        try:
            val = float(r.get("coop_average", 0.0))
        except Exception:
            continue
        key = (game, mech)
        sums[key] = sums.get(key, 0.0) + val
        counts[key] = counts.get(key, 0) + 1

    by_game: Dict[str, List[Tuple[str, float]]] = {}
    for (game, mech), s in sums.items():
        avg = s / max(counts[(game, mech)], 1)
        by_game.setdefault(game, []).append((mech, avg))

    for game, items in by_game.items():
        items_sorted = sorted(items, key=lambda kv: (-kv[1], kv[0]))
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [m for m, _ in items_sorted]
        values = [v for _, v in items_sorted]

        palette = sns.color_palette(PALETTE_BASE, len(values)) if values else []

        bars = ax.bar(
            labels,
            values,
            color=palette,
            edgecolor=COLOR_PALETTE["panel"],
            linewidth=1.5,
            alpha=0.9,
        )

        # Add value labels on bars while keeping them within [0, 1]
        y_max = 1.0
        offset = 0.02
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if height + offset <= y_max:
                y_pos = height + offset
                va = "bottom"
            else:
                y_pos = max(height - offset, offset)
                va = "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                f"{value:.3f}",
                ha="center",
                va=va,
                fontweight="semibold",
                fontsize=10,
                color=COLOR_PALETTE["text"],
            )

        ax.set_ylim(0, y_max)
        ax.set_title(
            f"Mechanism Effectiveness (Average Cooperation)\n{game}",
            pad=18,
        )
        ax.set_ylabel("Average Cooperation Rate")
        ax.set_xlabel("Mechanism")
        plt.xticks(rotation=30, ha="right")

        out_path = Path(out_dir) / "figures" / f"mechanism_effectiveness_{game}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _finalize(fig, ax)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_agent_performance(payoffs_csv: str | Path, coop_csv: str | Path, out_dir: str | Path) -> None:
    pay_rows: List[Dict[str, str]] = []
    coop_rows: List[Dict[str, str]] = []
    with open(payoffs_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pay_rows.append(row)
    with open(coop_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            coop_rows.append(row)
    if not pay_rows:
        return

    # Index coop rows for quick lookup
    coop_map: Dict[Tuple[str, str, str, str], float] = {}
    for row in coop_rows:
        key = (
            (row.get("run_dir") or "").strip(),
            (row.get("game") or "").strip(),
            (row.get("mechanism") or "").strip(),
            (row.get("agent") or "").strip(),
        )
        try:
            coop_map[key] = float(row.get("cooperation_rate", 0.0))
        except Exception:
            continue

    # Combine and aggregate per (game, mechanism, agent)
    sums_pay: Dict[Tuple[str, str, str], float] = {}
    sums_coop: Dict[Tuple[str, str, str], float] = {}
    counts: Dict[Tuple[str, str, str], int] = {}
    for row in pay_rows:
        key_run = (
            (row.get("run_dir") or "").strip(),
            (row.get("game") or "").strip(),
            (row.get("mechanism") or "").strip(),
            (row.get("agent") or "").strip(),
        )
        game, mech, agent = key_run[1], key_run[2], key_run[3]
        if not game or not mech or not agent:
            continue
        key = (game, mech, agent)
        try:
            pay = float(row.get("expected_payoff", 0.0))
        except Exception:
            continue
        coop = coop_map.get(key_run)
        if coop is None:
            # allow missing coop rows
            coop = math.nan
        sums_pay[key] = sums_pay.get(key, 0.0) + pay
        # Only accumulate coop if exists
        if not math.isnan(coop):
            sums_coop[key] = sums_coop.get(key, 0.0) + coop
        counts[key] = counts.get(key, 0) + 1

    # Organize by (game, mechanism)
    by_group: Dict[Tuple[str, str], List[Tuple[str, float, float]]] = {}
    for (game, mech, agent), c in counts.items():
        avg_pay = sums_pay[(game, mech, agent)] / c
        avg_coop = sums_coop.get((game, mech, agent), math.nan)
        if not math.isnan(avg_coop):
            avg_coop = avg_coop / c
        by_group.setdefault((game, mech), []).append((agent, avg_pay, avg_coop))

    for (game, mechanism), items in by_group.items():
        # Horizontal bar chart by expected payoff
        items_sorted = sorted(items, key=lambda t: (-t[1], t[0]))
        fig, ax = plt.subplots(figsize=(10, max(6, len(items_sorted) * 0.55)))
        labels = [a for a, _, _ in items_sorted]
        pays = [p for _, p, _ in items_sorted]

        palette = sns.color_palette(PALETTE_BASE, len(labels)) if labels else []

        bars = ax.barh(
            labels,
            pays,
            color=palette,
            edgecolor=COLOR_PALETTE["panel"],
            linewidth=1.2,
            alpha=0.92,
        )

        # Add value labels on bars
        for bar, pay in zip(bars, pays):
            width = bar.get_width()
            ax.text(
                width + max(pays) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{pay:.3f}",
                ha="left",
                va="center",
                fontweight="semibold",
                fontsize=10,
                color=COLOR_PALETTE["text"],
            )

        ax.set_title(f"Agent Expected Payoff\n{game} | {mechanism}", pad=18)
        ax.set_xlabel("Expected Payoff")
        ax.set_ylabel("Agent")
        ax.grid(axis="x")

        _finalize(fig, ax)
        out_path = Path(out_dir) / "figures" / f"agent_payoffs_{game}_{mechanism}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Scatter payoff vs cooperation where coop is available
        xs: List[float] = []
        ys: List[float] = []
        labs: List[str] = []
        for a, p, c in items:
            if c is None or math.isnan(c):
                continue
            xs.append(c)
            ys.append(p)
            labs.append(a)
        if xs:
            fig2, ax2 = plt.subplots(figsize=(8, 6))

            # Enhanced scatter plot with size based on payoff
            sizes = np.array(ys)
            if sizes.max() > sizes.min():
                norm_sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min())
                point_sizes = 80 + norm_sizes * 120  # Scale from 80 to 200
            else:
                point_sizes = [100] * len(sizes)

            scatter = ax2.scatter(
                xs,
                ys,
                c=ys,
                s=point_sizes,
                alpha=0.78,
                cmap=custom_cmap,
                edgecolors=COLOR_PALETTE["panel"],
                linewidth=1.2,
            )

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label("Expected Payoff")
            cbar.outline.set_color("none")

            # Better annotations with background boxes
            for x, y, lab in zip(xs, ys, labs):
                ax2.annotate(
                    lab,
                    (x, y),
                    fontsize=9,
                    fontweight="semibold",
                    xytext=(6, 6),
                    textcoords="offset points",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor=COLOR_PALETTE["panel"],
                        edgecolor=COLOR_PALETTE["grid"],
                        alpha=0.85,
                    ),
                )

            ax2.set_xlim(-0.05, 1.05)
            ax2.set_xlabel("Cooperation Rate")
            ax2.set_ylabel("Expected Payoff")
            ax2.set_title(f"Payoff vs Cooperation Analysis\n{game} | {mechanism}", pad=18)
            ax2.grid(True)

            _finalize(fig2, ax2)
            out_path2 = Path(out_dir) / "figures" / f"agent_payoff_vs_coop_{game}_{mechanism}.png"
            fig2.savefig(out_path2, dpi=300, bbox_inches="tight")
            plt.close(fig2)


def plot_pairwise_and_trajectories(
    pairwise_csv: str | Path,
    conditional_csv: str | Path,
    trajectory_csv: str | Path,
    out_dir: str | Path,
) -> None:
    # Pairwise heatmap-like bar plots per run (simple version: grouped bars per pair)
    def read_csv(path: Path) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _avg(values: Iterable[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and not (v != v)]
    return sum(vals) / len(vals) if vals else float("nan")


def plot_pairwise_heatmaps(pairwise_csv: str | Path, out_dir: str | Path) -> None:
    rows = _load_csv(Path(pairwise_csv))
    if not rows:
        return

    by_run: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        run_dir = row.get("run_dir")
        agent_i = row.get("agent_i")
        agent_j = row.get("agent_j")
        if not run_dir or not agent_i or not agent_j:
            continue
        bucket = by_run.setdefault(run_dir, {"rows": [], "mechanisms": set()})
        bucket["rows"].append(row)
        bucket["mechanisms"].add(row.get("mechanism") or "Unknown")

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for run_dir, payload in by_run.items():
        run_rows = payload["rows"]
        agents = sorted({r["agent_i"] for r in run_rows} | {r["agent_j"] for r in run_rows})
        if not agents:
            continue

        index = {agent: i for i, agent in enumerate(agents)}
        matrix = np.full((len(agents), len(agents)), np.nan, dtype=float)
        coop_matrix = np.full((len(agents), len(agents)), np.nan, dtype=float)
        values: Dict[Tuple[str, str], List[float]] = {}
        coop_values: Dict[Tuple[str, str], List[float]] = {}
        for r in run_rows:
            ai = r["agent_i"]
            aj = r["agent_j"]
            val = _safe_float(r.get("points_i"))
            coop_val = _safe_float(r.get("coop_i"))
            if math.isnan(val):
                continue
            values.setdefault((ai, aj), []).append(val)
            if not math.isnan(coop_val):
                coop_values.setdefault((ai, aj), []).append(coop_val)

        for (ai, aj), vals in values.items():
            i = index[ai]
            j = index[aj]
            matrix[i, j] = sum(vals) / len(vals)

        for (ai, aj), vals in coop_values.items():
            i = index[ai]
            j = index[aj]
            coop_matrix[i, j] = sum(vals) / len(vals)

        labels_x = [_clean_agent_label(agent, wrap=False, width=18) for agent in agents]
        labels_y = [_clean_agent_label(agent, wrap=True, width=14) for agent in agents]
        size = max(4.5, 0.9 * len(agents) + 2.0)
        fig, axes = plt.subplots(1, 2, figsize=(size * 1.9, size), sharey=True)
        ax_payoff, ax_coop = axes

        annot = np.empty_like(matrix, dtype=object)
        for i in range(len(agents)):
            for j in range(len(agents)):
                annot[i, j] = "" if math.isnan(matrix[i, j]) else f"{matrix[i, j]:.2f}"

        coop_annot = np.empty_like(coop_matrix, dtype=object)
        for i in range(len(agents)):
            for j in range(len(agents)):
                if math.isnan(coop_matrix[i, j]):
                    coop_annot[i, j] = ""
                else:
                    coop_annot[i, j] = f"{coop_matrix[i, j]:.2f}"

        mask = np.isnan(matrix)
        coop_mask = np.isnan(coop_matrix)
        heatmap_payoff = sns.heatmap(
            matrix,
            mask=mask,
            cmap=custom_cmap,
            annot=annot,
            fmt="",
            linewidths=0.5,
            linecolor=COLOR_PALETTE["background"],
            cbar_kws={"label": "Avg payoff"},
            square=True,
            vmin=0.0,
            vmax=3.0,
            cbar=False,
            ax=ax_payoff,
        )

        tick_positions = np.arange(len(agents)) + 0.5
        ax_payoff.set_xticks(tick_positions)
        ax_payoff.set_xticklabels(labels_x, rotation=35, ha="right")
        ax_payoff.set_yticks(tick_positions)
        ax_payoff.set_yticklabels(labels_y, rotation=0, ha="right")
        ax_payoff.tick_params(axis="y", pad=8)

        mech_title = ", ".join(sorted(payload["mechanisms"]))
        ax_payoff.set_title(f"Pairwise Payoffs – {mech_title}")
        ax_payoff.set_xlabel("Column agent")
        ax_payoff.set_ylabel("Row agent")

        heatmap_coop = sns.heatmap(
            coop_matrix,
            mask=coop_mask,
            cmap=coop_cmap,
            annot=coop_annot,
            fmt="",
            linewidths=0.5,
            linecolor=COLOR_PALETTE["background"],
            cbar=False,
            square=True,
            vmin=0.0,
            vmax=1.0,
            ax=ax_coop,
        )

        ax_coop.set_xticks(tick_positions)
        ax_coop.set_xticklabels(labels_x, rotation=35, ha="right")
        ax_coop.set_yticks(tick_positions)
        ax_coop.set_yticklabels(["" for _ in labels_y])
        ax_coop.tick_params(axis="y", length=0)
        ax_coop.set_title(f"Pairwise Cooperation – {mech_title}")
        ax_coop.set_xlabel("Column agent")
        ax_coop.set_ylabel("")

        # Add dedicated colorbars for readability
        if ax_payoff.collections:
            cbar1 = fig.colorbar(ax_payoff.collections[0], ax=ax_payoff, fraction=0.046, pad=0.04)
            cbar1.set_label("Avg payoff")
        if ax_coop.collections:
            cbar2 = fig.colorbar(ax_coop.collections[0], ax=ax_coop, fraction=0.046, pad=0.04)
            cbar2.set_label("Cooperation rate")

        _style_axes(ax_payoff)
        _style_axes(ax_coop)
        fig.patch.set_facecolor(COLOR_PALETTE["background"])
        fig.tight_layout()

        slug = _slugify(Path(run_dir).name)
        out_path = fig_dir / f"pairwise_payoffs_{slug}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Scatter plot of cooperation vs payoff per ordered pair
        points: List[Tuple[float, float, str, str]] = []
        for i, agent_i in enumerate(agents):
            for j, agent_j in enumerate(agents):
                val = matrix[i, j]
                coop_val = coop_matrix[i, j]
                if math.isnan(val) or math.isnan(coop_val):
                    continue
                points.append((coop_val, val, agent_i, agent_j))

        if points:
            fig_scatter, ax_scatter = plt.subplots(figsize=(12, 7.5))
            palette = sns.color_palette("Set2", n_colors=max(len(agents), 2))
            color_map = {agent: palette[idx] for idx, agent in enumerate(agents)}
            marker_cycle = ["o", "s", "D", "^", "v", "P", "X", "<", ">", "*", "H"]
            marker_map = {
                agent: marker_cycle[idx % len(marker_cycle)] for idx, agent in enumerate(agents)
            }

            handles_opponent: Dict[str, Line2D] = {}
            label_offsets: Dict[Tuple[float, float], int] = defaultdict(int)
            x_vals: List[float] = []
            y_vals: List[float] = []

            for coop_val, payoff_val, row_agent, col_agent in points:
                color = color_map[row_agent]
                marker = marker_map[col_agent]
                x_pct = coop_val * 100.0
                ax_scatter.scatter(
                    x_pct,
                    payoff_val,
                    s=110,
                    color=color,
                    marker=marker,
                    edgecolor=COLOR_PALETTE["background"],
                    linewidth=0.9,
                    alpha=0.9,
                )

                key = (round(x_pct, 1), round(payoff_val, 3))
                idx = label_offsets[key]
                label_offsets[key] += 1
                dx = 4.5 * ((idx % 3) - 1)
                dy = 0.08 * (idx // 3)
                text_x = max(min(x_pct + dx, 103.0), -3.0)
                text_y = max(min(payoff_val + dy, 3.3), -0.3)
                ax_scatter.annotate(
                    _clean_agent_label(row_agent, wrap=False, width=18),
                    xy=(x_pct, payoff_val),
                    xytext=(text_x, text_y),
                    textcoords="data",
                    fontsize=9,
                    ha="center",
                    color=COLOR_PALETTE["text"],
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor=COLOR_PALETTE["panel"],
                        edgecolor=COLOR_PALETTE["grid"],
                        alpha=0.85,
                    ),
                )

                if col_agent not in handles_opponent:
                    handles_opponent[col_agent] = Line2D(
                        [],
                        [],
                        marker=marker,
                        linestyle="",
                        color=COLOR_PALETTE["muted"],
                        markersize=8,
                        markerfacecolor=COLOR_PALETTE["muted"],
                        label=_clean_agent_label(col_agent),
                    )

                x_vals.append(x_pct)
                y_vals.append(payoff_val)

            ax_scatter.set_xlim(-5, 105)
            ax_scatter.set_ylim(-0.2, 3.3)
            ax_scatter.set_xlabel("Cooperation rate (%) – row agent")
            ax_scatter.set_ylabel("Avg payoff (row -> column)")
            ax_scatter.set_title(f"Pairwise Payoff vs Cooperation – {mech_title}")

            if handles_opponent:
                legend_cols = ax_scatter.legend(
                    handles=list(handles_opponent.values()),
                    bbox_to_anchor=(1.02, 0.95),
                    loc="upper left",
                    title="Opponent (marker)",
                    frameon=True,
                    framealpha=0.9,
                )
                legend_cols.get_frame().set_facecolor(COLOR_PALETTE["panel"])

            stats_text = None
            if x_vals and y_vals:
                avg_coop = float(np.mean(x_vals))
                std_coop = float(np.std(x_vals))
                avg_payoff = float(np.mean(y_vals))
                std_payoff = float(np.std(y_vals))
                stats_text = (
                    f"n={len(points)} pairs\n"
                    f"Avg coop: {avg_coop:.1f}% ± {std_coop:.1f}%\n"
                    f"Avg payoff: {avg_payoff:.2f} ± {std_payoff:.2f}"
                )

                filtered = [(x, y) for x, y in zip(x_vals, y_vals) if y > 0.05]
                if len(filtered) >= 3:
                    xs, ys = map(np.array, zip(*filtered))
                    if np.ptp(xs) > 1e-6:
                        z = np.polyfit(xs, ys, 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(0, 100, 200)
                        ax_scatter.plot(
                            x_line,
                            p(x_line),
                            linestyle="--",
                            color=COLOR_PALETTE["accent1"],
                            linewidth=1.8,
                            alpha=0.6,
                            label=f"Trend: payoff = {z[0]:.3f}·coop + {z[1]:.2f}",
                        )
                        corr = float(np.corrcoef(xs, ys)[0, 1]) if np.std(xs) > 1e-6 else float("nan")
                        if not math.isnan(corr):
                            ax_scatter.text(
                                0.02,
                                0.95,
                                f"Corr≈{corr:.2f}",
                                transform=ax_scatter.transAxes,
                                fontsize=10,
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    facecolor=COLOR_PALETTE["panel"],
                                    edgecolor=COLOR_PALETTE["grid"],
                                    alpha=0.75,
                                ),
                            )

            if stats_text:
                ax_scatter.text(
                    1.02,
                    0.4,
                    stats_text,
                    transform=ax_scatter.transAxes,
                    fontsize=9,
                    va="center",
                    ha="left",
                    bbox=dict(
                        boxstyle="round,pad=0.35",
                        facecolor=COLOR_PALETTE["panel"],
                        edgecolor=COLOR_PALETTE["grid"],
                        alpha=0.85,
                    ),
                )

            _style_axes(ax_scatter)
            fig_scatter.patch.set_facecolor(COLOR_PALETTE["background"])
            fig_scatter.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])

            scatter_path = fig_dir / f"pairwise_payoff_coop_{slug}.png"
            fig_scatter.savefig(scatter_path, dpi=300, bbox_inches="tight")
            plt.close(fig_scatter)


def plot_mechanism_summary(summary_csv: str | Path, out_dir: str | Path) -> None:
    rows = _load_csv(Path(summary_csv))
    if not rows:
        return

    by_game: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        game = row.get("game") or "Unknown"
        by_game.setdefault(game, []).append(row)

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for game, items in by_game.items():
        labels = [item.get("mechanism", "Unknown") for item in items]
        payoffs = [_safe_float(item.get("avg_expected_payoff")) for item in items]
        coop = [_safe_float(item.get("avg_cooperation_rate")) for item in items]
        delta = [_safe_float(item.get("avg_delta_expected_payoff")) for item in items]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(10, 6))
        bars = ax1.bar(
            x - width / 2,
            payoffs,
            width,
            label="Avg Payoff",
            color=COLOR_PALETTE["primary"],
            alpha=0.9,
        )
        ax1.bar(
            x + width / 2,
            coop,
            width,
            label="Avg Cooperation",
            color=COLOR_PALETTE["accent3"],
            alpha=0.85,
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=30, ha="right")
        ax1.set_ylabel("Value")
        ax1.set_title(f"Mechanism Summary – {game}")
        ax1.legend(frameon=False, ncols=2)
        ax1.grid(axis="y")

        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="semibold",
                color=COLOR_PALETTE["text"],
            )

        ax2 = ax1.twinx()
        ax2.plot(
            x,
            delta,
            color=COLOR_PALETTE["accent1"],
            marker="o",
            linewidth=2.2,
            label="Δ Payoff vs Base",
        )
        ax2.set_ylabel("Δ Payoff", color=COLOR_PALETTE["accent1"])
        ax2.tick_params(axis="y", colors=COLOR_PALETTE["accent1"])
        _style_axes(ax1)
        _style_axes(ax2, facecolor="none")

        lines, labels_leg = ax1.get_legend_handles_labels()
        l2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + l2, labels_leg + labels2, loc='upper left')

        _finalize(fig, ax1)
        fig.savefig(fig_dir / f"mechanism_summary_{game}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_agent_metrics(agent_csv: str | Path, out_dir: str | Path) -> None:
    rows = _load_csv(Path(agent_csv))
    if not rows:
        return

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Scatter: expected payoff vs cooperation rate colored by mechanism
    mechanisms = sorted(set(row.get("mechanism", "Unknown") for row in rows))
    mech_colors = {mech: plt.get_cmap("tab10")(i % 10) for i, mech in enumerate(mechanisms)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for row in rows:
        mech = row.get("mechanism", "Unknown")
        x_val = _safe_float(row.get("expected_payoff"))
        y_val = _safe_float(row.get("coop_rate"))
        ax.scatter(
            x_val,
            y_val,
            color=mech_colors.get(mech, COLOR_PALETTE["primary"]),
            alpha=0.8,
            s=110,
            edgecolor=COLOR_PALETTE["panel"],
            linewidth=1.0,
            label=mech,
        )
        ax.annotate(
            row.get("agent", ""),
            (x_val, y_val),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            fontweight="semibold",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=COLOR_PALETTE["panel"],
                edgecolor=COLOR_PALETTE["grid"],
                alpha=0.85,
            ),
        )
    ax.set_xlabel("Expected Payoff")
    ax.set_ylabel("Cooperation Rate")
    ax.set_title("Agent Payoff vs Cooperation")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="best", frameon=False)
    ax.grid(True)
    _finalize(fig, ax)
    fig.savefig(fig_dir / "agent_payoff_vs_coop.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Horizontal bars of expected payoff per mechanism
    clusters: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        mech = row.get("mechanism", "Unknown")
        clusters[mech].append(_safe_float(row.get("expected_payoff")))

    fig, ax = plt.subplots(figsize=(10, max(4, len(clusters) * 0.8)))
    mechanisms = sorted(clusters)
    means = [_avg(clusters[m]) for m in mechanisms]
    palette = sns.color_palette(PALETTE_BASE, len(mechanisms)) if mechanisms else []
    bars = ax.barh(
        mechanisms,
        means,
        color=palette,
        alpha=0.88,
    )
    for bar, val in zip(bars, means):
        ax.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=10,
            fontweight="semibold",
            color=COLOR_PALETTE["text"],
        )
    ax.set_xlabel("Average Expected Payoff")
    ax.set_title("Average Agent Payoff by Mechanism")
    ax.grid(axis="x")
    _finalize(fig, ax)
    fig.savefig(fig_dir / "agent_payoff_by_mechanism.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_disarm_cap_trajectories(rows: List[Dict[str, Any]], out_dir: str | Path) -> None:
    if not rows:
        return
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run[row["run_dir"]].append(row)

    for run_dir, run_rows in by_run.items():
        actions = sorted({r.get("action_label", r.get("action", "Action")) for r in run_rows})
        agents = sorted({r["agent"] for r in run_rows})
        palette = sns.color_palette(PALETTE_BASE, len(agents))
        agent_colors = {agent: palette[idx % len(palette)] for idx, agent in enumerate(agents)}

        session_groups = defaultdict(list)
        for row in run_rows:
            session_groups[(row["agent"], row.get("session_id"))].append(row)

        session_display: Dict[tuple[str, str | None], str] = {}
        pair_counts: Dict[tuple[str, str], int] = defaultdict(int)
        for (agent, session_id), rows_in_session in session_groups.items():
            opponents = rows_in_session[0].get("opponents", "")
            pair_key = (agent, opponents)
            pair_counts[pair_key] += 1
            agent_label = _clean_agent_label(agent, wrap=True, width=14)
            opp_label = ", ".join(
                _clean_agent_label(o, wrap=False)
                for o in opponents.split("|")
                if o
            )
            base_label = f"{agent_label} vs {opp_label}" if opp_label else agent_label
            count = pair_counts[pair_key]
            display_label = base_label if count == 1 else f"{base_label} (#{count})"
            session_display[(agent, session_id)] = display_label

        fig, axes = plt.subplots(
            len(actions),
            1,
            sharex=True,
            figsize=(10, max(4.5, 3.0 * len(actions))),
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for action_idx, (ax, action_label) in enumerate(zip(axes, actions)):
            series = defaultdict(list)
            for row in run_rows:
                if row.get("action_label") != action_label:
                    continue
                series[(row["agent"], row.get("session_id"))].append(row)

            for (agent, session_id), seq in series.items():
                seq = sorted(
                    seq,
                    key=lambda r: float(r.get("session_round") or r.get("round") or 0),
                )
                xs = [float(r.get("session_round") or r.get("round") or 0) for r in seq]
                ys = [float(r.get("cap", 0.0) or 0.0) for r in seq]
                if not xs:
                    continue
                color = agent_colors.get(agent, COLOR_PALETTE["primary"])
                label = session_display.get((agent, session_id)) if action_idx == 0 else None
                ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=label)

            ax.set_ylabel(f"{action_label}\nCap (%)")
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)
            _style_axes(ax)

        axes[-1].set_xlabel("Negotiation round")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="upper right")

        fig.patch.set_facecolor(COLOR_PALETTE["background"])
        fig.tight_layout()
        slug = _slugify(Path(run_dir).name)
        fig.savefig(fig_dir / f"disarm_caps_{slug}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_disarm_reduction_bars(rows: List[Dict[str, Any]], out_dir: str | Path) -> None:
    if not rows:
        return

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run[row["run_dir"]].append(row)

    for run_dir, run_rows in by_run.items():
        if not run_rows:
            continue
        action_labels = run_rows[0].get("action_labels", "")
        actions = [label for label in action_labels.split("|") if label]
        if not actions:
            actions = ["Action 0", "Action 1"]

        agents = [r["agent"] for r in run_rows]
        positions = np.arange(len(agents))
        fig, ax = plt.subplots(figsize=(10, max(4.5, 0.6 * len(agents))))

        bottoms = np.zeros(len(agents))
        colors = sns.color_palette(PALETTE_BASE, len(actions))

        for idx, action_label in enumerate(actions):
            key = _format_action_key("reduction", action_label)
            vals = [float(r.get(key, 0.0) or 0.0) for r in run_rows]
            ax.barh(
                positions,
                vals,
                left=bottoms,
                color=colors[idx % len(colors)],
                alpha=0.85,
                label=action_label,
            )
            bottoms += np.array(vals)

        ax.set_yticks(positions)
        ax.set_yticklabels([_clean_agent_label(agent, wrap=True, width=16) for agent in agents])
        ax.set_xlabel("Total cap reduction")
        ax.set_title("Disarmament Cap Reductions")
        ax.legend(loc="lower right", frameon=False)
        _style_axes(ax)
        fig.patch.set_facecolor(COLOR_PALETTE["background"])
        fig.tight_layout()
        slug = _slugify(Path(run_dir).name)
        fig.savefig(fig_dir / f"disarm_reduction_{slug}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_disarm_binding_heatmap(rows: List[Dict[str, Any]], out_dir: str | Path) -> None:
    if not rows:
        return

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run[row["run_dir"]].append(row)

    for run_dir, run_rows in by_run.items():
        action_labels = run_rows[0].get("action_labels", "")
        actions = [label for label in action_labels.split("|") if label]
        if not actions:
            actions = ["Action 0", "Action 1"]

        agents = sorted({r["agent"] for r in run_rows})
        matrix = np.zeros((len(agents), len(actions)))
        matrix[:] = np.nan

        for i, agent in enumerate(agents):
            row = next((r for r in run_rows if r["agent"] == agent), None)
            if not row:
                continue
            for j, action_label in enumerate(actions):
                key = _format_action_key("binding_rate", action_label)
                matrix[i, j] = float(row.get(key, float("nan")) or float("nan"))

        fig, ax = plt.subplots(figsize=(7, max(3.5, 0.5 * len(agents))))
        annot = np.empty_like(matrix, dtype=object)
        for i in range(len(agents)):
            for j in range(len(actions)):
                val = matrix[i, j]
                annot[i, j] = "" if math.isnan(val) else f"{val:.2f}"

        sns.heatmap(
            matrix,
            mask=np.isnan(matrix),
            cmap=custom_cmap,
            annot=annot,
            fmt="",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor=COLOR_PALETTE["background"],
            cbar_kws={"label": "Binding rate"},
            ax=ax,
        )
        ax.set_xticklabels(actions, rotation=35, ha="right")
        ax.set_yticklabels([_clean_agent_label(a, wrap=True, width=16) for a in agents], rotation=0)
        ax.set_title("Cap Binding Frequency")
        _style_axes(ax)
        fig.patch.set_facecolor(COLOR_PALETTE["background"])
        fig.tight_layout()
        slug = _slugify(Path(run_dir).name)
        fig.savefig(fig_dir / f"disarm_binding_{slug}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_disarm_cooperation_trajectory(rows: List[Dict[str, Any]], out_dir: str | Path) -> None:
    if not rows:
        return

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run[row["run_dir"]].append(row)

    for run_dir, run_rows in by_run.items():
        sessions = defaultdict(list)
        for row in run_rows:
            sessions[(row["agent"], row.get("session_id"))].append(row)

        agents = sorted({row["agent"] for row in run_rows})
        palette = sns.color_palette(PALETTE_BASE, len(agents))
        agent_colors = {agent: palette[idx % len(palette)] for idx, agent in enumerate(agents)}

        session_display: Dict[tuple[str, str | None], str] = {}
        pair_counts: Dict[tuple[str, str], int] = defaultdict(int)
        for (agent, session_id), session_rows in sessions.items():
            opponents = session_rows[0].get("opponents", "")
            pair_key = (agent, opponents)
            pair_counts[pair_key] += 1
            agent_label = _clean_agent_label(agent, wrap=True, width=14)
            opp_label = ", ".join(
                _clean_agent_label(o, wrap=False)
                for o in opponents.split("|")
                if o
            )
            base_label = f"{agent_label} vs {opp_label}" if opp_label else agent_label
            count = pair_counts[pair_key]
            display_label = base_label if count == 1 else f"{base_label} (#{count})"
            session_display[(agent, session_id)] = display_label

        fig, ax = plt.subplots(figsize=(9, 4))
        for (agent, session_id), session_rows in sessions.items():
            seq = sorted(
                session_rows,
                key=lambda r: float(r.get("session_round") or r.get("round") or 0),
            )
            xs = [float(r.get("session_round") or r.get("round") or 0) for r in seq]
            ys = [float(r.get("avg_cooperation", 0.0) or 0.0) for r in seq]
            if not xs:
                continue
            color = agent_colors.get(agent, COLOR_PALETTE["primary"])
            label = session_display.get((agent, session_id))
            ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=label)

        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Negotiation round")
        ax.set_ylabel("Average cooperation")
        ax.set_title("Cooperation Through Disarmament Rounds")
        ax.grid(True, alpha=0.3)
        _style_axes(ax)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best")
        fig.patch.set_facecolor(COLOR_PALETTE["background"])
        fig.tight_layout()
        slug = _slugify(Path(run_dir).name)
        fig.savefig(fig_dir / f"disarm_cooperation_{slug}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_disarm_payoff_vs_caps(rows: List[Dict[str, Any]], out_dir: str | Path) -> None:
    if not rows:
        return

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run[row["run_dir"]].append(row)

    for run_dir, run_rows in by_run.items():
        action_labels = run_rows[0].get("action_labels", "").split("|")
        if len(action_labels) < 2:
            continue
        defect_label = action_labels[1]
        key_final = _format_action_key("final_cap", defect_label)
        key_reduce = _format_action_key("reduction", defect_label)

        fig, ax = plt.subplots(figsize=(8, 5))
        for row in run_rows:
            x = float(row.get(key_final, 0.0) or 0.0)
            y = float(row.get("expected_payoff", 0.0) or 0.0)
            size = float(row.get(key_reduce, 0.0) or 0.0)
            size = 80 + min(max(size, 0.0), 100.0)
            label = _clean_agent_label(row["agent"], wrap=False)
            ax.scatter(x, y, s=size, color=COLOR_PALETTE["accent1"], alpha=0.75)
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

        ax.set_xlabel(f"Final cap on {defect_label}")
        ax.set_ylabel("Expected payoff")
        ax.set_title("Payoff vs Final Defect Cap")
        ax.grid(True, alpha=0.3)
        _style_axes(ax)
        fig.patch.set_facecolor(COLOR_PALETTE["background"])
        fig.tight_layout()
        slug = _slugify(Path(run_dir).name)
        fig.savefig(fig_dir / f"disarm_payoff_cap_{slug}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_disarm_negotiation_hist(rows: List[Dict[str, Any]], out_dir: str | Path) -> None:
    if not rows:
        return

    durations = [float(row.get("negotiation_rounds", 0) or 0) for row in rows]
    if not durations:
        return

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    max_duration = max(durations)
    bins = range(1, int(math.ceil(max_duration)) + 2)
    ax.hist(durations, bins=bins, color=COLOR_PALETTE["secondary"], alpha=0.85, rwidth=0.85)
    ax.set_xlabel("Negotiation rounds until termination")
    ax.set_ylabel("Run count")
    ax.set_title("Disarmament Negotiation Length")
    ax.grid(True, axis="y", alpha=0.3)
    _style_axes(ax)
    fig.patch.set_facecolor(COLOR_PALETTE["background"])
    fig.tight_layout()
    fig.savefig(fig_dir / "disarm_negotiation_lengths.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
