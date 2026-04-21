"""Generate payoff tensor visualizations from experiment results."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

from coopeval.config import FIGURE_DIR
from coopeval.ranking_evaluations.matchup_payoffs import MatchupPayoffs
from coopeval.script_utils.display_helper import (
    format_identifier_as_title,
    format_mechanism_name,
    format_model_name,
    make_mechanism_suffix,
    sort_games,
    sort_mechanisms,
    to_snake_case,
)
from coopeval.script_utils.result_loader import (
    DEFAULT_SKIP_GAMES,
    iter_experiments,
    load_json,
)
from coopeval.utils.json_io import clean_path
from coopeval.visualization.analysis_utils import (
    NormalizeScore,
    is_reputation_mechanism,
    validate_dict_consistency,
    validate_folder_count_consistency,
    validate_list_consistency,
)
from coopeval.script_utils.colors import CMAP_VMAX, CMAP_VMIN, custom_cmap
from coopeval.script_utils.figure_exports import save_matplotlib_figure


def relabel_colorbar_ticks(
    cbar: Colorbar,
    normalizer: NormalizeScore,
) -> None:
    """Show normalized colorbar ticks as raw game payoffs."""
    normalized_ticks = [float(tick) for tick in cbar.get_ticks()]
    raw_ticks = [normalizer.denormalize(tick) for tick in normalized_ticks]
    cbar.set_ticks(normalized_ticks)
    cbar.set_ticklabels([f"{val:.1f}" for val in raw_ticks])


def load_full_payoff_tensor(
    folder_path: Path,
) -> tuple[np.ndarray, list[str]]:
    """Load matchup payoffs from one experiment and expand them for plotting."""
    json_data = load_json(folder_path / "matchup_payoffs.json")

    payoffs = MatchupPayoffs.from_json(json_data)

    # Leverage the symmetry-aware helper so we do not duplicate indexing logic here
    full_tensor = payoffs.build_full_payoff_tensor()

    agent_types = list(payoffs.agent_types)

    return full_tensor, agent_types


def get_output_path(output_dir: Path, mechanism: str, game: str) -> Path:
    """Return the PDF path for a mechanism-game payoff tensor figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{to_snake_case(mechanism)}_{to_snake_case(game)}_payoff_tensor.pdf"
    )
    return output_dir / filename


def average_tensors(
    tensors: list[np.ndarray], group_key: tuple[str, str]
) -> np.ndarray:
    """Average compatible tensor arrays for one game-mechanism group."""
    if not tensors:
        raise ValueError(f"No tensors to average for {group_key}")

    expected_shape = tensors[0].shape
    for i, tensor in enumerate(tensors[1:], start=1):
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch in {group_key}: "
                f"tensor 0 has shape {expected_shape}, "
                f"tensor {i} has shape {tensor.shape}"
            )

    stacked = np.stack(tensors, axis=0)
    averaged = np.mean(stacked, axis=0)

    return averaged


def validate_group_consistency(
    agent_labels_list: list[list[str]],
    configs: list[dict],
    folders: list[Path],
    group_key: tuple[str, str],
) -> tuple[list[str], dict, dict]:
    """Validate that repeated runs in a group can be averaged together."""
    if not agent_labels_list or not configs or not folders:
        raise ValueError(f"Empty inputs for validation of {group_key}")

    folder_identifiers = [f.name for f in folders]

    validated_labels = validate_list_consistency(
        agent_labels_list, folder_identifiers, group_key, "agent labels"
    )

    game_configs = [config["game"] for config in configs]
    validated_game_config = validate_dict_consistency(
        game_configs, folder_identifiers, group_key, "game config"
    )

    mechanism_configs = [config["mechanism"] for config in configs]
    validated_mechanism_config = validate_dict_consistency(
        mechanism_configs, folder_identifiers, group_key, "mechanism config"
    )

    return validated_labels, validated_game_config, validated_mechanism_config


def generate_latex_file(
    output_dir: Path, created_plots: list[tuple[str, str, Path]]
) -> None:
    """Write a LaTeX include file referencing the generated tensor figures."""
    latex_path = output_dir / "payoff_tensors.tex"

    with latex_path.open("w", encoding="utf-8") as f:
        f.write("% Payoff Tensor Visualizations\n")
        f.write("% Generated automatically\n\n")

        mechanisms = {}
        for mechanism, game, filepath in created_plots:
            if mechanism not in mechanisms:
                mechanisms[mechanism] = []
            mechanisms[mechanism].append((game, filepath))

        for mechanism in sort_mechanisms(mechanisms):
            display_mech = format_mechanism_name(mechanism)
            f.write(f"\n% {display_mech}\n")
            game_list = [game for game, _ in mechanisms[mechanism]]
            sorted_games = sort_games(game_list)
            game_filepath_map = {
                game: filepath for game, filepath in mechanisms[mechanism]
            }

            for game in sorted_games:
                filepath = game_filepath_map[game]
                filename = filepath.name
                game_title = format_identifier_as_title(game)
                f.write(f"\n% {game_title}\n")
                f.write("\\begin{figure}[htbp]\n")
                f.write("    \\centering\n")
                f.write(
                    f"    \\includegraphics[width=0.8\\textwidth]{{payoff_tensors/{filename}}}\n"
                )
                f.write(
                    "    \\caption{The cells display the payoff vectors in the metagame where each player can select an LLM model to play the game with. The cell color indicates player 1's payoff specifically. Light red (resp.\\ green) represents the payoff player 1 would receive under the Nash equilibrium (resp.\\ the cooperative action profile) of the base game.}\n"
                )
                f.write(
                    f"    \\label{{payoff:{to_snake_case(mechanism)}_{to_snake_case(game)}}}\n"
                )
                f.write("\\end{figure}\n")

    print(f"\nGenerated LaTeX file: {latex_path}")


def plot_2player_payoff_tensor(
    full_tensor: np.ndarray,
    agent_labels: list[str],
    game_name: str,
    mechanism_name: str,
    normalizer: NormalizeScore,
) -> Figure:
    """Render a two-player payoff tensor as a single annotated heatmap."""
    n = len(agent_labels)

    # Reshape using the same joint-strategy ordering (i,j) -> i*n + j used during serialization
    p1_payoffs = full_tensor[0, :].reshape(n, n)
    p2_payoffs = full_tensor[1, :].reshape(n, n)

    p1_normalized = np.zeros_like(p1_payoffs)
    for i in range(n):
        for j in range(n):
            p1_normalized[i, j] = normalizer.normalize(p1_payoffs[i, j])

    p1_normalized = np.clip(p1_normalized, CMAP_VMIN, CMAP_VMAX)

    annotations = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            annotations[i, j] = f"{p1_payoffs[i, j]:.1f}/{p2_payoffs[i, j]:.1f}"

    # Strip provider prefixes to prevent axis labels from overflowing
    cleaned_labels = [format_model_name(label) for label in agent_labels]

    fig, ax = plt.subplots(figsize=(10, 8))

    heatmap = sns.heatmap(
        data=p1_normalized,
        annot=annotations,
        fmt="s",
        cmap=custom_cmap,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=cleaned_labels,
        yticklabels=cleaned_labels,
        cbar_kws={
            "label": f"Player 1 Payoff ({normalizer.denormalize(0.0)} = NE payoff, {normalizer.denormalize(1.0)} = Cooperative payoff)"
        },
        vmin=CMAP_VMIN,
        vmax=CMAP_VMAX,
        ax=ax,
    )

    cbar = heatmap.collections[0].colorbar
    if cbar is None:
        raise RuntimeError("Expected seaborn heatmap to create a colorbar.")
    relabel_colorbar_ticks(cbar, normalizer)

    ax.set_xlabel("Player 2 Model", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Player 1 Model", fontsize=12, fontweight="semibold")
    ax.set_title(
        f"{game_name} - {format_mechanism_name(mechanism_name)}",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def plot_3player_payoff_tensor(
    full_tensor: np.ndarray,
    agent_labels: list[str],
    game_name: str,
    mechanism_name: str,
    normalizer: NormalizeScore,
) -> Figure:
    """Render a three-player payoff tensor as one heatmap per player-3 model."""
    n = len(agent_labels)

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    cleaned_labels = [format_model_name(label) for label in agent_labels]

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

        p1_normalized = np.clip(p1_normalized, CMAP_VMIN, CMAP_VMAX)

        sns.heatmap(
            data=p1_normalized,
            annot=annotations,
            fmt="s",
            annot_kws={"fontsize": 8},
            cmap=custom_cmap,
            square=True,
            linewidths=0.5,
            linecolor="white",
            xticklabels=cleaned_labels,
            yticklabels=cleaned_labels,
            cbar=False,
            vmin=CMAP_VMIN,
            vmax=CMAP_VMAX,
            ax=ax,
        )

        ax.set_xlabel("Player 2 Model", fontsize=11)
        ax.set_ylabel("Player 1 Model", fontsize=11)
        ax.set_title(
            f"Player 3: {cleaned_labels[k]}", fontsize=12, fontweight="bold"
        )

    fig.subplots_adjust(right=0.92, hspace=0.3, wspace=0.3)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(
        axes[0].collections[0],
        cax=cbar_ax,
        label=f"Player 1 Payoff ({normalizer.denormalize(0.0)} = NE payoff, {normalizer.denormalize(1.0)} = Cooperative payoff)",
    )

    relabel_colorbar_ticks(cbar, normalizer)

    fig.suptitle(
        f"{game_name} - {format_mechanism_name(mechanism_name)}",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    return fig


def plot_payoff_tensor(
    full_tensor: np.ndarray,
    agent_labels: list[str],
    game_name: str,
    mechanism_name: str,
    normalizer: NormalizeScore,
) -> Figure:
    """Render a payoff tensor figure for the supported player count."""
    num_players = full_tensor.shape[0]
    if num_players == 2:
        return plot_2player_payoff_tensor(
            full_tensor,
            agent_labels,
            game_name,
            mechanism_name,
            normalizer,
        )
    if num_players == 3:
        return plot_3player_payoff_tensor(
            full_tensor,
            agent_labels,
            game_name,
            mechanism_name,
            normalizer,
        )
    raise ValueError(
        f"Unsupported number of players for {game_name}/{mechanism_name}: "
        f"{num_players}"
    )


def generate_payoff_tensor_outputs(
    experiment_dirs: list[str | Path],
    output_dir: str | Path,
    skip_games: list[str] | None = None,
) -> None:
    """Build, average, and render payoff tensor plots from experiment results."""
    output_dir = Path(output_dir)
    created_plots = []

    print("Phase 1: Discovering and grouping experiment folders...")

    raw_entries = []

    for experiment_dir in experiment_dirs:
        experiment_dir = Path(experiment_dir)
        discovered_count = 0

        for experiment in iter_experiments(
            experiment_dir, skip_games=skip_games
        ):
            folder = experiment.path
            config = load_json(folder / "config.json")
            game_type = experiment.game
            mechanism_type = experiment.mechanism
            mechanism_kwargs = config["mechanism"]["kwargs"]
            discovered_count += 1

            if is_reputation_mechanism(mechanism_type):
                print(
                    f"Skipping reputation mechanism: {mechanism_type}_{game_type}"
                )
                continue

            raw_entries.append(
                (mechanism_type, mechanism_kwargs, game_type, folder, config)
            )

        print(
            f"Discovered {discovered_count} experiment folders from {experiment_dir}"
        )

    type_to_kwargs_list: dict[str, list[dict]] = defaultdict(list)
    for (
        mechanism_type,
        mechanism_kwargs,
        _game,
        _folder,
        _config,
    ) in raw_entries:
        type_to_kwargs_list[mechanism_type].append(mechanism_kwargs)

    varying_keys_per_type: dict[str, frozenset] = {}
    for mtype, kwargs_list in type_to_kwargs_list.items():
        all_keys = {k for kw in kwargs_list for k in kw}
        varying = frozenset(
            k for k in all_keys if len({str(kw[k]) for kw in kwargs_list}) > 1
        )
        if varying:
            varying_keys_per_type[mtype] = varying

    grouped_folders: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "folders": [],
            "configs": [],
            "tensors": [],
            "agent_labels_list": [],
        }
    )

    for (
        mechanism_type,
        mechanism_kwargs,
        game_type,
        folder,
        config,
    ) in raw_entries:
        if mechanism_type in varying_keys_per_type:
            varying_keys = varying_keys_per_type[mechanism_type]
            suffix = make_mechanism_suffix(mechanism_kwargs, varying_keys)
            mechanism_key = f"{mechanism_type} ({suffix})"
        else:
            mechanism_key = mechanism_type

        group_key = (game_type, mechanism_key)
        grouped_folders[group_key]["folders"].append(folder)
        grouped_folders[group_key]["configs"].append(config)

    print(f"Grouped into {len(grouped_folders)} game-mechanism combinations\n")

    print("Phase 2: Validating folder counts and loading tensors...")

    expected_folder_count = validate_folder_count_consistency(grouped_folders)
    print(
        f"All groups have {expected_folder_count} folder(s) - validation passed\n"
    )

    for group_key, group_data in grouped_folders.items():
        game_type, mechanism_type = group_key
        print(f"Loading {mechanism_type}_{game_type}...")

        for folder in group_data["folders"]:
            full_tensor, agent_labels = load_full_payoff_tensor(folder)
            group_data["tensors"].append(full_tensor)
            group_data["agent_labels_list"].append(agent_labels)

        agent_labels, game_config, mechanism_config = (
            validate_group_consistency(
                group_data["agent_labels_list"],
                group_data["configs"],
                group_data["folders"],
                group_key,
            )
        )

        group_data["agent_labels"] = agent_labels
        group_data["game_config"] = game_config
        group_data["mechanism_config"] = mechanism_config

        print(f"  Loaded and validated {len(group_data['tensors'])} tensor(s)")

    print("\nPhase 3: Averaging tensors and creating plots...")

    for group_key, group_data in grouped_folders.items():
        game_type, mechanism_type = group_key
        print(f"\nPlotting {mechanism_type}_{game_type}...")

        averaged_tensor = average_tensors(group_data["tensors"], group_key)
        print(f"  Averaged {len(group_data['tensors'])} tensor(s)")

        agent_labels = group_data["agent_labels"]
        game_config = group_data["game_config"]

        normalizer = NormalizeScore(game_type, game_config)

        output_path = get_output_path(output_dir, mechanism_type, game_type)
        fig = plot_payoff_tensor(
            averaged_tensor,
            agent_labels,
            game_type,
            mechanism_type,
            normalizer,
        )
        save_matplotlib_figure(
            fig,
            output_path.with_suffix(""),
            [output_path.suffix.lstrip(".")],
            dpi=300,
            bbox_inches="tight",
            format_subdirs=False,
        )
        plt.close(fig)

        created_plots.append((mechanism_type, game_type, output_path))
        print(f"  Created: {output_path}")

    generate_latex_file(output_dir, created_plots)

    print(f"\n{'=' * 80}")
    print("Summary:")
    print(f"  Created {len(created_plots)} plots")
    print(f"  Each plot averaged {expected_folder_count} tensor(s)")
    print(f"{'=' * 80}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for payoff tensor plotting."""

    parser = argparse.ArgumentParser(
        description="Generate payoff tensor visualizations from experiment results"
    )
    parser.add_argument(
        "--tournament_result_dirs",
        nargs="+",
        type=clean_path,
        required=True,
        help="Tournament result batch to scan.",
    )
    parser.add_argument(
        "--output-dir",
        type=clean_path,
        default=FIGURE_DIR,
        help="Directory where figures and the LaTeX include file are written.",
    )
    parser.add_argument(
        "--skip-games",
        dest="skip_games",
        nargs="*",
        default=DEFAULT_SKIP_GAMES,
        help=(
            "Games to skip before aggregation "
            "(default: %(default)s; pass with no values to include all)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Parse CLI arguments and generate payoff tensor plots."""
    args = parse_args()
    generate_payoff_tensor_outputs(
        args.tournament_result_dirs,
        args.output_dir,
        skip_games=args.skip_games,
    )


if __name__ == "__main__":
    main()
