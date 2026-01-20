"""Generate payoff tensor visualizations from experiment results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.ranking_evaluations.matchup_payoffs import MatchupPayoffs

# Styling constants
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


# Helper functions
def load_config(folder_path: Path) -> dict[str, Any]:
    """Load config.json from experiment folder."""
    with open(folder_path / "config.json") as f:
        return json.load(f)


def load_and_build_tensor(folder_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load matchup payoffs and build tensor."""
    with open(folder_path / "matchup_payoffs.json") as f:
        json_data = json.load(f)

    payoffs = MatchupPayoffs.from_json(json_data)
    payoffs.build_payoff_tensor()

    return payoffs._payoff_tensor, payoffs._tensor_agent_types


def get_num_players(game_type: str) -> int:
    """Return 3 for PublicGoods, 2 for all others."""
    return 3 if game_type == "PublicGoods" else 2


def clean_agent_label(label: str) -> str:
    """Shorten agent label for plotting."""
    # Remove provider prefix and shorten
    parts = label.split("/")
    if len(parts) > 1:
        model_and_type = parts[-1]
        # Shorten model name if too long
        if len(model_and_type) > 25:
            model_and_type = model_and_type[:22] + "..."
        return model_and_type
    return label


def to_snake_case(text: str) -> str:
    """Convert text to snake_case."""
    # Handle camelCase to snake_case
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def get_output_path(output_dir: Path, mechanism: str, game: str) -> Path:
    """Create output path: {output_dir}/{mechanism}/{game}_payoff_tensor.png"""
    mechanism_folder = output_dir / to_snake_case(mechanism)
    mechanism_folder.mkdir(parents=True, exist_ok=True)

    game_filename = to_snake_case(game) + "_payoff_tensor.png"
    return mechanism_folder / game_filename


def discover_experiment_folders(experiment_dir: Path) -> list[Path]:
    """Find all subdirectories in experiment folder."""
    return [d for d in experiment_dir.iterdir() if d.is_dir()]


# Visualization functions
def plot_2player_payoff_tensor(
    tensor: np.ndarray,
    agent_labels: list[str],
    game_name: str,
    mechanism_name: str,
    output_path: Path,
) -> None:
    """Create heatmap for 2-player game payoff tensor."""
    # Extract both players' payoffs
    p1_payoffs = tensor
    p2_payoffs = tensor.T

    # Create annotation matrix with both payoffs
    n = len(agent_labels)
    annotations = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            annotations[i, j] = f"{p1_payoffs[i, j]:.2f} / {p2_payoffs[i, j]:.2f}"

    # Clean labels for display
    cleaned_labels = [clean_agent_label(label) for label in agent_labels]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        data=p1_payoffs,
        annot=annotations,
        fmt='s',
        cmap=custom_cmap,
        square=True,
        linewidths=0.5,
        linecolor='white',
        xticklabels=cleaned_labels,
        yticklabels=cleaned_labels,
        cbar_kws={'label': 'Player 1 Payoff'},
        ax=ax,
    )

    ax.set_xlabel("Player 2 Model", fontsize=12, fontweight='semibold')
    ax.set_ylabel("Player 1 Model", fontsize=12, fontweight='semibold')
    ax.set_title(f"{game_name} - {mechanism_name}", fontsize=16, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_3player_payoff_tensor(
    tensor: np.ndarray,
    agent_labels: list[str],
    game_name: str,
    mechanism_name: str,
    output_path: Path,
) -> None:
    """Create multiple heatmaps for 3-player game, one per player 3's choice."""
    n = len(agent_labels)

    # Create 2x3 subplot grid for 6 agents
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Clean labels for display
    cleaned_labels = [clean_agent_label(label) for label in agent_labels]

    # Find global min/max for consistent colorbar
    vmin = tensor.min()
    vmax = tensor.max()

    # Create a heatmap for each player 3 choice
    for k in range(n):
        ax = axes[k]

        # Extract payoffs for all players when player 3 chooses agent k
        # P1 payoffs: tensor[i, j, k]
        # P2 payoffs: tensor[j, i, k] (swap positions 0 and 1)
        # P3 payoffs: tensor[k, j, i] (swap positions 0 and 2)

        # Create annotation matrix with all three payoffs
        annotations = np.empty((n, n), dtype=object)
        p1_payoffs = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                p1 = tensor[i, j, k]
                p2 = tensor[j, i, k]
                p3 = tensor[k, j, i]
                p1_payoffs[i, j] = p1
                annotations[i, j] = f"{p1:.2f}/{p2:.2f}/{p3:.2f}"

        # Create heatmap
        sns.heatmap(
            data=p1_payoffs,
            annot=annotations,
            fmt='s',
            cmap=custom_cmap,
            square=True,
            linewidths=0.5,
            linecolor='white',
            xticklabels=cleaned_labels,
            yticklabels=cleaned_labels,
            cbar=False,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            annot_kws={'fontsize': 8},
        )

        ax.set_xlabel("Player 2 Model", fontsize=10)
        ax.set_ylabel("Player 1 Model", fontsize=10)
        ax.set_title(f"Player 3: {cleaned_labels[k]}", fontsize=11, fontweight='bold')

    # Add a single colorbar for all subplots
    fig.colorbar(
        axes[0].collections[0],
        ax=axes,
        label='Player 1 Payoff',
        shrink=0.6,
    )

    fig.suptitle(f"{game_name} - {mechanism_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Main function
def plot_payoff_tensors(experiment_dir: str | Path, output_dir: str | Path) -> None:
    """Generate payoff tensor visualizations for all game-mechanism combinations."""
    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir)

    # Discover all experiment folders
    folders = discover_experiment_folders(experiment_dir)

    for folder in folders:
        # Load config to get game type and mechanism
        config = load_config(folder)
        game_type = config["game"]["type"]
        mechanism_type = config["mechanism"]["type"]

        # Load and build payoff tensor
        tensor, agent_labels = load_and_build_tensor(folder)

        # Determine number of players
        num_players = get_num_players(game_type)

        # Get output path
        output_path = get_output_path(output_dir, mechanism_type, game_type)

        # Create appropriate visualization
        if num_players == 2:
            plot_2player_payoff_tensor(
                tensor, agent_labels, game_type, mechanism_type, output_path
            )
        else:  # num_players == 3
            plot_3player_payoff_tensor(
                tensor, agent_labels, game_type, mechanism_type, output_path
            )

        print(f"Created: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate payoff tensor visualizations from experiment results"
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment folder containing game-mechanism subfolders",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path where visualization subfolders will be created",
    )

    args = parser.parse_args()
    plot_payoff_tensors(args.experiment_dir, args.output_dir)
