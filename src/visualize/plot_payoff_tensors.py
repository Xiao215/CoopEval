"""Generate payoff tensor visualizations from experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.ranking_evaluations.matchup_payoffs import MatchupPayoffs
from src.visualize.analysis_utils import (NormalizeScore,
                                            discover_experiment_subfolders,
                                            get_num_players_from_matchup,
                                            load_json,
                                            simplify_model_name,
                                            sort_games,
                                            sort_mechanisms,
                                            )

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
def load_and_build_tensor(folder_path: Path) -> tuple[np.ndarray, list[str], int]:
    """
    Load matchup payoffs and build full payoff tensor for all players.

    Returns:
        Tuple of (full_tensor, agent_labels, num_players)
        where full_tensor has shape (num_players, n_strategies^num_players)
    """
    json_data = load_json(folder_path / "matchup_payoffs.json")

    payoffs = MatchupPayoffs.from_json(json_data)
    payoffs.build_payoff_tensor()

    num_players = get_num_players_from_matchup(json_data)

    # Build full payoff tensor for all players using existing symmetry method
    full_tensor = payoffs.build_full_payoff_tensor()

    return full_tensor, payoffs._tensor_agent_types, num_players


def to_snake_case(text: str) -> str:
    """Convert text to snake_case."""
    # Handle camelCase to snake_case
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def get_output_path(output_dir: Path, mechanism: str, game: str) -> Path:
    """Create output path: {output_dir}/{mechanism}_{game}_payoff_tensor.png"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{to_snake_case(mechanism)}_{to_snake_case(game)}_payoff_tensor.png"
    return output_dir / filename


def generate_latex_file(output_dir: Path, created_plots: list[tuple[str, str, Path]]) -> None:
    """Generate LaTeX file with all plots for easy inclusion in papers.
    
    Args:
        output_dir: Directory where LaTeX file will be saved
        created_plots: List of (mechanism, game, filepath) tuples
    """
    latex_path = output_dir / "payoff_tensors.tex"
    
    with open(latex_path, 'w') as f:
        f.write("% Payoff Tensor Visualizations\n")
        f.write("% Generated automatically\n\n")
        
        # Group by mechanism
        mechanisms = {}
        for mechanism, game, filepath in created_plots:
            if mechanism not in mechanisms:
                mechanisms[mechanism] = []
            mechanisms[mechanism].append((game, filepath))
        
        # Write organized by mechanism
        for mechanism in sort_mechanisms(list(mechanisms.keys())):
            f.write(f"\n% {mechanism.replace('_', ' ').title()}\n")
            # Sort games within each mechanism
            game_list = [game for game, _ in mechanisms[mechanism]]
            sorted_games = sort_games(game_list)
            game_filepath_map = {game: filepath for game, filepath in mechanisms[mechanism]}
            
            for game in sorted_games:
                filepath = game_filepath_map[game]
                filename = filepath.name
                game_title = game.replace('_', ' ').title()
                f.write(f"\n% {game_title}\n")
                f.write("\\begin{figure}[htbp]\n")
                f.write("    \\centering\n")
                f.write(f"    \\includegraphics[width=0.8\\textwidth]{{{filename}}}\n")
                f.write(f"    \\caption{{{game_title} - {mechanism.replace('_', ' ').title()}}}\n")
                f.write(f"    \\label{{fig:{to_snake_case(mechanism)}_{to_snake_case(game)}}}\n")
                f.write("\\end{figure}\n")
    
    print(f"\nGenerated LaTeX file: {latex_path}")


# Visualization functions
def plot_2player_payoff_tensor(
    full_tensor: np.ndarray,
    agent_labels: list[str],
    game_name: str,
    mechanism_name: str,
    output_path: Path,
    normalizer: NormalizeScore,
) -> None:
    """Create heatmap for 2-player game payoff tensor.

    Args:
        full_tensor: Shape (2, n^2) from build_full_payoff_tensor()
        normalizer: Score normalizer for the game
    """
    n = len(agent_labels)

    # Reshape full_tensor to get payoff matrices for each player
    # Joint strategies are ordered as (i,j) -> i*n + j
    p1_payoffs = full_tensor[0, :].reshape(n, n)
    p2_payoffs = full_tensor[1, :].reshape(n, n)

    # Create normalized payoff matrix for coloring
    p1_normalized = np.zeros_like(p1_payoffs)
    for i in range(n):
        for j in range(n):
            p1_normalized[i, j] = normalizer.normalize(p1_payoffs[i, j])

    # Clip normalized scores to [-1, 1.5] range for consistent color mapping
    p1_normalized = np.clip(p1_normalized, -1.0, 1.5)

    # Create annotation matrix with both players' payoffs (raw scores)
    annotations = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            annotations[i, j] = f"{p1_payoffs[i, j]:.1f}/{p2_payoffs[i, j]:.1f}"

    # Clean labels for display
    cleaned_labels = [simplify_model_name(label) for label in agent_labels]

    # Create heatmap - color by normalized player 1's payoff, annotate with raw payoffs
    # Use fixed vmin/vmax for consistent colors across all plots
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        data=p1_normalized,
        annot=annotations,
        fmt='s',
        cmap=custom_cmap,
        square=True,
        linewidths=0.5,
        linecolor='white',
        xticklabels=cleaned_labels,
        yticklabels=cleaned_labels,
        cbar_kws={'label': 'Normalized Score (0=NE, 1=Cooperative)'},
        vmin=-1.0,
        vmax=1.5,
        ax=ax,
    )

    ax.set_xlabel("Player 2 Model", fontsize=12, fontweight='semibold')
    ax.set_ylabel("Player 1 Model", fontsize=12, fontweight='semibold')
    ax.set_title(f"{game_name} - {mechanism_name}", fontsize=16, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_3player_payoff_tensor(
    full_tensor: np.ndarray,
    agent_labels: list[str],
    game_name: str,
    mechanism_name: str,
    output_path: Path,
    normalizer: NormalizeScore,
) -> None:
    """Create multiple heatmaps for 3-player game, one per player 3's choice.

    Args:
        full_tensor: Shape (3, n^3) from build_full_payoff_tensor()
        normalizer: Score normalizer for the game
    """
    n = len(agent_labels)

    # Create 2x3 subplot grid for 6 agents with larger figure and adjusted spacing
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    # Clean labels for display
    cleaned_labels = [simplify_model_name(label) for label in agent_labels]

    # Use fixed vmin/vmax for consistent colors across all plots
    vmin = -1.0
    vmax = 1.5

    # Create a heatmap for each player 3 choice
    for k in range(n):
        ax = axes[k]

        # Extract all players' payoffs when player 3 chooses strategy k
        # Joint strategies are ordered as (i,j,k) -> i*n*n + j*n + k
        p1_payoffs = np.zeros((n, n))
        p1_normalized = np.zeros((n, n))
        p2_payoffs = np.zeros((n, n))
        p3_payoffs = np.zeros((n, n))
        annotations = np.empty((n, n), dtype=object)

        for i in range(n):
            for j in range(n):
                joint_idx = i * n * n + j * n + k
                p1 = full_tensor[0, joint_idx]
                p2 = full_tensor[1, joint_idx]
                p3 = full_tensor[2, joint_idx]

                p1_payoffs[i, j] = p1
                p1_normalized[i, j] = normalizer.normalize(p1)
                p2_payoffs[i, j] = p2
                p3_payoffs[i, j] = p3
                annotations[i, j] = f"{p1:.1f}/{p2:.1f}/{p3:.1f}"

        # Clip normalized scores to [-1, 1.5] range for consistent color mapping
        p1_normalized = np.clip(p1_normalized, -1.0, 1.5)

        # Create heatmap - color by normalized player 1's payoff, annotate with raw payoffs
        sns.heatmap(
            data=p1_normalized,
            annot=annotations,
            fmt='s',
            annot_kws={'fontsize': 8},
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
        )

        ax.set_xlabel("Player 2 Model", fontsize=11)
        ax.set_ylabel("Player 1 Model", fontsize=11)
        ax.set_title(f"Player 3: {cleaned_labels[k]}", fontsize=12, fontweight='bold')

    # Add a single colorbar for all subplots on the right side
    fig.subplots_adjust(right=0.92, hspace=0.3, wspace=0.3)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(
        axes[0].collections[0],
        cax=cbar_ax,
        label='Normalized Score (0=NE, 1=Cooperative)',
    )

    fig.suptitle(f"{game_name} - {mechanism_name}", fontsize=18, fontweight='bold', y=0.98)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Main function
def plot_payoff_tensors(experiment_dirs: list[str | Path], output_dir: str | Path) -> None:
    """Generate payoff tensor visualizations for all game-mechanism combinations from multiple experiment folders."""
    output_dir = Path(output_dir)

    # Collect all experiment folders from all input directories
    all_folders = []
    seen_outputs = {}  # Track which output files we've created
    created_plots = []  # Track all created plots for LaTeX file
    failed_folders = []  # Track folders that failed to process

    for experiment_dir in experiment_dirs:
        experiment_dir = Path(experiment_dir)
        folders = discover_experiment_subfolders(experiment_dir)

        for folder in folders:
            # Load config
            config = load_json(folder / "config.json")
            game_type = config["game"]["type"]
            mechanism_type = config["mechanism"]["type"]

            # Skip reputation mechanisms
            if mechanism_type.lower() == "reputation":
                print(f"Skipping reputation mechanism: {mechanism_type}_{game_type}")
                continue

            all_folders.append((folder, game_type, mechanism_type))

        print(f"Discovered {len(folders)} experiment folders from {experiment_dir}")

    print(f"\nProcessing {len(all_folders)} game-mechanism combinations\n")

    # Process all folders
    for folder, game_type, mechanism_type in all_folders:
        try:
            # Get output path
            output_path = get_output_path(output_dir, mechanism_type, game_type)

            # Check if we're overwriting a previously created plot
            if output_path in seen_outputs:
                print(f"WARNING: Duplicate found for {mechanism_type}_{game_type}")
                print(f"  Previous: {seen_outputs[output_path]}")
                print(f"  Current:  {folder}")
                print(f"  Replacing plot at {output_path}")

            seen_outputs[output_path] = folder

            # Load config to create normalizer
            config = load_json(folder / "config.json")
            game_config = config["game"]
            normalizer = NormalizeScore(game_type, game_config)

            # Load and build payoff tensor, get num_players from matchup data
            full_tensor, agent_labels, num_players = load_and_build_tensor(folder)

            # Create appropriate visualization
            if num_players == 2:
                plot_2player_payoff_tensor(
                    full_tensor, agent_labels, game_type, mechanism_type, output_path, normalizer
                )
            else:  # num_players == 3
                plot_3player_payoff_tensor(
                    full_tensor, agent_labels, game_type, mechanism_type, output_path, normalizer
                )

            print(f"Created: {output_path}")
            created_plots.append((mechanism_type, game_type, output_path))

        except Exception as e:
            print(f"WARNING: Failed to process {mechanism_type}_{game_type} from {folder}")
            print(f"  Error: {type(e).__name__}: {e}")
            failed_folders.append((folder, game_type, mechanism_type, e))

    # Generate LaTeX file
    if created_plots:
        generate_latex_file(output_dir, created_plots)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Successfully created: {len(created_plots)} plots")
    print(f"  Failed: {len(failed_folders)} folders")

    if failed_folders:
        print(f"\n{'='*80}")
        print(f"Failed folders ({len(failed_folders)}):")
        for folder, game_type, mechanism_type, error in failed_folders:
            print(f"  - {mechanism_type}_{game_type} ({folder})")
            print(f"    Error: {type(error).__name__}: {error}")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate payoff tensor visualizations from experiment results"
    )
    parser.add_argument(
        "experiment_dirs",
        type=str,
        nargs="+",
        help="Path(s) to experiment folder(s) containing game-mechanism subfolders",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path where visualization subfolders will be created",
    )

    args = parser.parse_args()
    plot_payoff_tensors(args.experiment_dirs, args.output_dir)
