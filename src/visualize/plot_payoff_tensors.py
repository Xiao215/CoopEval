"""Generate payoff tensor visualizations from experiment results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.ranking_evaluations.matchup_payoffs import MatchupPayoffs
from src.visualize.analysis_utils import (NormalizeScore,
                                            discover_experiment_subfolders,
                                            display_mechanism_name,
                                            get_num_players_from_matchup,
                                            load_json,
                                            simplify_model_name,
                                            sort_games,
                                            sort_mechanisms,
                                            validate_dict_consistency,
                                            validate_folder_count_consistency,
                                            validate_list_consistency,
                                            )

# Styling constants (monotone from "bad" purple to "good" cyan for payoff shifts)
PALETTE_BASE = ["#2E0854", "#B30000", "#FF5722", "#FFEB3B", "#00E5FF", "#00BCD4"]
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

    # Leverage the symmetry-aware helper so we do not duplicate indexing logic here
    full_tensor = payoffs.build_full_payoff_tensor()

    return full_tensor, payoffs._tensor_agent_types, num_players


def to_snake_case(text: str) -> str:
    """Convert CamelCase/mixed identifiers to snake_case."""
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def get_output_path(output_dir: Path, mechanism: str, game: str) -> Path:
    """Create output path: {output_dir}/{mechanism}_{game}_payoff_tensor.pdf"""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{to_snake_case(mechanism)}_{to_snake_case(game)}_payoff_tensor.pdf"
    return output_dir / filename


def average_tensors(tensors: list[np.ndarray], group_key: tuple[str, str]) -> np.ndarray:
    """Average multiple payoff tensors for the same game-mechanism combination.

    Args:
        tensors: List of tensor arrays to average (all must have same shape)
        group_key: (game_type, mechanism_type) for error messages

    Returns:
        Averaged tensor with same shape as inputs

    Raises:
        ValueError: If tensors have inconsistent shapes
    """
    if not tensors:
        raise ValueError(f"No tensors to average for {group_key}")

    # Validate all tensors have same shape
    expected_shape = tensors[0].shape
    for i, tensor in enumerate(tensors[1:], start=1):
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch in {group_key}: "
                f"tensor 0 has shape {expected_shape}, "
                f"tensor {i} has shape {tensor.shape}"
            )

    # Stack and compute mean along new axis
    stacked = np.stack(tensors, axis=0)
    averaged = np.mean(stacked, axis=0)

    return averaged


def validate_group_consistency(
    agent_labels_list: list[list[str]],
    configs: list[dict],
    folders: list[Path],
    group_key: tuple[str, str]
) -> tuple[list[str], dict, dict]:
    """Validate all folders in a group have identical experimental setup.

    Args:
        agent_labels_list: List of agent_labels from each folder
        configs: List of full config dicts from each folder
        folders: List of folder paths (for error messages)
        group_key: (game_type, mechanism_type) for error messages

    Returns:
        Tuple of (validated_agent_labels, validated_game_config, validated_mechanism_config)

    Raises:
        AssertionError: If any inconsistencies are found
    """
    if not agent_labels_list or not configs or not folders:
        raise ValueError(f"Empty inputs for validation of {group_key}")

    # Convert folder paths to strings for identifiers
    folder_identifiers = [f.name for f in folders]

    # Validate agent labels (same models in same order)
    validated_labels = validate_list_consistency(
        agent_labels_list,
        folder_identifiers,
        group_key,
        "agent labels"
    )

    # Validate game configs (identical game parameters)
    game_configs = [config["game"] for config in configs]
    validated_game_config = validate_dict_consistency(
        game_configs,
        folder_identifiers,
        group_key,
        "game config"
    )

    # Validate mechanism configs (identical mechanism parameters)
    mechanism_configs = [config["mechanism"] for config in configs]
    validated_mechanism_config = validate_dict_consistency(
        mechanism_configs,
        folder_identifiers,
        group_key,
        "mechanism config"
    )

    return validated_labels, validated_game_config, validated_mechanism_config


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
        
        # Keep mechanisms grouped so LaTeX consumers can import exactly the sections they need
        mechanisms = {}
        for mechanism, game, filepath in created_plots:
            if mechanism not in mechanisms:
                mechanisms[mechanism] = []
            mechanisms[mechanism].append((game, filepath))
        
        # Deterministic ordering within each mechanism keeps git diffs readable
        for mechanism in sort_mechanisms(list(mechanisms.keys())):
            display_mech = display_mechanism_name(mechanism)
            f.write(f"\n% {display_mech.replace('_', ' ').title()}\n")
            game_list = [game for game, _ in mechanisms[mechanism]]
            sorted_games = sort_games(game_list)
            game_filepath_map = {game: filepath for game, filepath in mechanisms[mechanism]}

            for game in sorted_games:
                filepath = game_filepath_map[game]
                filename = filepath.name
                game_title = game.replace('_', ' ').title()
                f.write(f"\n% {game_title}\n")
                f.write("\\begin{figure}[t]\n")
                f.write("    \\centering\n")
                f.write(f"    \\includegraphics[width=0.8\\textwidth]{{payoff_tensors/{filename}}}\n")
                f.write(f"    \\caption{{{game_title} - {display_mech}}}\n")
                f.write(f"    \\label{{payoff:{to_snake_case(mechanism)}_{to_snake_case(game)}}}\n")
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

    # Reshape using the same joint-strategy ordering (i,j) -> i*n + j used during serialization
    p1_payoffs = full_tensor[0, :].reshape(n, n)
    p2_payoffs = full_tensor[1, :].reshape(n, n)

    # Color encodes Player 1's normalized payoff so plots share a consistent legend
    p1_normalized = np.zeros_like(p1_payoffs)
    for i in range(n):
        for j in range(n):
            p1_normalized[i, j] = normalizer.normalize(p1_payoffs[i, j])

    # Clamp extremes so qualitatively similar plots use the same palette range
    p1_normalized = np.clip(p1_normalized, -1.0, 1.5)

    # Keep raw payoffs visible so readers can recover both players' utilities
    annotations = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            annotations[i, j] = f"{p1_payoffs[i, j]:.1f}/{p2_payoffs[i, j]:.1f}"

    # Strip provider prefixes to prevent axis labels from overflowing
    cleaned_labels = [simplify_model_name(label) for label in agent_labels]

    # Fixed vmin/vmax keeps visual comparisons meaningful across mechanisms
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
    ax.set_title(f"{game_name} - {display_mechanism_name(mechanism_name)}", fontsize=16, fontweight='bold')

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

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    cleaned_labels = [simplify_model_name(label) for label in agent_labels]

    vmin = -1.0
    vmax = 1.5

    for k in range(n):
        ax = axes[k]

        # Joint strategies follow (i,j,k) -> i*n*n + j*n + k
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

        p1_normalized = np.clip(p1_normalized, -1.0, 1.5)

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

    fig.subplots_adjust(right=0.92, hspace=0.3, wspace=0.3)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(
        axes[0].collections[0],
        cax=cbar_ax,
        label='Normalized Score (0=NE, 1=Cooperative)',
    )

    fig.suptitle(f"{game_name} - {display_mechanism_name(mechanism_name)}", fontsize=18, fontweight='bold', y=0.98)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Main function
def plot_payoff_tensors(experiment_dirs: list[str | Path], output_dir: str | Path) -> None:
    """Generate payoff tensor visualizations for all game-mechanism combinations from multiple experiment folders."""
    output_dir = Path(output_dir)
    created_plots = []

    print("Phase 1: Discovering and grouping experiment folders...")

    grouped_folders: dict[tuple[str, str], dict] = defaultdict(lambda: {
        'folders': [],
        'configs': [],
        'tensors': [],
        'agent_labels_list': [],
    })

    for experiment_dir in experiment_dirs:
        experiment_dir = Path(experiment_dir)
        folders = discover_experiment_subfolders(experiment_dir)

        for folder in folders:
            config = load_json(folder / "config.json")
            game_type = config["game"]["type"]
            mechanism_type = config["mechanism"]["type"]

            if mechanism_type.lower() in ["reputation", "reputationfirstorder"]:
                print(f"Skipping reputation mechanism: {mechanism_type}_{game_type}")
                continue

            group_key = (game_type, mechanism_type)
            grouped_folders[group_key]['folders'].append(folder)
            grouped_folders[group_key]['configs'].append(config)

        print(f"Discovered {len(folders)} experiment folders from {experiment_dir}")

    print(f"Grouped into {len(grouped_folders)} game-mechanism combinations\n")

    print("Phase 2: Validating folder counts and loading tensors...")

    expected_folder_count = validate_folder_count_consistency(grouped_folders)
    print(f"All groups have {expected_folder_count} folder(s) - validation passed\n")

    for group_key, group_data in grouped_folders.items():
        game_type, mechanism_type = group_key
        print(f"Loading {mechanism_type}_{game_type}...")

        for folder in group_data['folders']:
            full_tensor, agent_labels, num_players = load_and_build_tensor(folder)
            group_data['tensors'].append(full_tensor)
            group_data['agent_labels_list'].append(agent_labels)

        agent_labels, game_config, mechanism_config = validate_group_consistency(
            group_data['agent_labels_list'],
            group_data['configs'],
            group_data['folders'],
            group_key
        )

        group_data['agent_labels'] = agent_labels
        group_data['game_config'] = game_config
        group_data['mechanism_config'] = mechanism_config

        print(f"  Loaded and validated {len(group_data['tensors'])} tensor(s)")

    print("\nPhase 3: Averaging tensors and creating plots...")

    for group_key, group_data in grouped_folders.items():
        game_type, mechanism_type = group_key
        print(f"\nPlotting {mechanism_type}_{game_type}...")

        averaged_tensor = average_tensors(group_data['tensors'], group_key)
        print(f"  Averaged {len(group_data['tensors'])} tensor(s)")

        agent_labels = group_data['agent_labels']
        game_config = group_data['game_config']

        normalizer = NormalizeScore(game_type, game_config)

        num_players = averaged_tensor.shape[0]

        output_path = get_output_path(output_dir, mechanism_type, game_type)

        if num_players == 2:
            plot_2player_payoff_tensor(
                averaged_tensor, agent_labels, game_type,
                mechanism_type, output_path, normalizer
            )
        else:  # num_players == 3
            plot_3player_payoff_tensor(
                averaged_tensor, agent_labels, game_type,
                mechanism_type, output_path, normalizer
            )

        created_plots.append((mechanism_type, game_type, output_path))
        print(f"  Created: {output_path}")

    # Generate LaTeX file
    generate_latex_file(output_dir, created_plots)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Created {len(created_plots)} plots")
    print(f"  Each plot averaged {expected_folder_count} tensor(s)")
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
