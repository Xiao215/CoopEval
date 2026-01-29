#!/usr/bin/env python3
"""
Re-run Evaluation Methods on Existing Batch Experiment Folders

This script loads saved matchup payoffs from existing experiment folders,
reconstructs necessary objects, and re-runs evaluation methods (replicator
dynamics and deviation ratings) with new configuration parameters.

Usage:
    python script/rerun_evaluations.py \
        --batch-folders data/clean/run2_20260124_025842 \
        --evaluation-config evaluation/default_evaluation.yaml \
        --seed 42
"""

import argparse
import copy
import json
import random
import sys
import yaml
from datetime import datetime
from pathlib import Path
import traceback

import numpy as np
import torch

from src.agents.agent_manager import Agent
from src.config_loader import ConfigLoader
from src.logger_manager import LOGGER
from src.ranking_evaluations.deviation_rating import DeviationRating
from src.ranking_evaluations.matchup_payoffs import MatchupPayoffs
from src.ranking_evaluations.replicator_dynamics import DiscreteReplicatorDynamics
from src.registry.agent_registry import create_players_with_player_id


def set_seed(seed: int = 42):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def discover_experiment_folders(batch_dir: Path) -> list[Path]:
    """
    Discover experiment subfolders in batch directory, excluding reputation experiments.

    Args:
        batch_dir: Path to batch root directory

    Returns:
        List of experiment directory paths (non-reputation only)
    """
    experiment_dirs = []

    for item in batch_dir.iterdir():
        # Only consider real experiment directories; skip metadata and stray files
        if not item.is_dir():
            continue
        if item.name in {"configs", "__pycache__", "slurm"}:
            continue

        # Treat directories with serialized configs as runnable experiments
        config_file = item / "config.json"
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        # Reputation experiments have bespoke evaluation logic; leave them untouched here
        mechanism_type = config["mechanism"]["type"].lower()
        if mechanism_type in {"reputation", "reputationfirstorder"}:
            print(f"  Skipping reputation experiment: {item.name}")
            continue

        matchup_file = item / "matchup_payoffs.json"
        assert matchup_file.exists()
        experiment_dirs.append(item)

    return sorted(experiment_dirs)


def load_matchup_payoffs(experiment_dir: Path) -> MatchupPayoffs:
    """
    Load MatchupPayoffs from matchup_payoffs.json.

    Args:
        experiment_dir: Path to experiment folder

    Returns:
        MatchupPayoffs instance with reconstructed data
    """
    matchup_file = experiment_dir / "matchup_payoffs.json"

    with open(matchup_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # MatchupPayoffs.from_json() rehydrates players/metadata needed downstream
    return MatchupPayoffs.from_json(payload)


def load_agents_from_config(config: dict) -> list[Agent]:
    """
    Load and reconstruct Agent instances from experiment config.

    Args:
        config: Experiment config dict

    Returns:
        List of Agent instances
    """
    agents_config = config["agents"]
    game_config = config["game"]

    # Non-public-goods games are 2-player; public goods encodes its arity explicitly
    game_type = game_config["type"]
    if game_type == "PublicGoods":
        num_players = game_config["kwargs"]["num_players"]
    else:
        num_players = 2  # Default for most games

    # Use the registry helper so player IDs match the serialized payoff tensor
    players = create_players_with_player_id(agents_config, num_players)

    return players


def validate_evaluation_config(evaluation_config: dict) -> None:
    """
    Validate the entire evaluation config once at startup.

    Args:
        evaluation_config: Complete evaluation config dict with 'methods' key

    Raises:
        ValueError: If required kwargs are missing for any evaluation method
    """
    required_kwargs = {
        "evolutionary_dynamics": ["initial_population", "steps", "lr_method", "lr_nu"],
        "deviation_rating": ["tolerance"],
    }

    for eval_method in evaluation_config["methods"]:
        eval_type = eval_method["type"]
        eval_kwargs = eval_method.get("kwargs", {})

        missing = [k for k in required_kwargs[eval_type] if k not in eval_kwargs]
        if missing:
            raise ValueError(
                f"Missing required kwargs for '{eval_type}': {missing}"
            )


def run_evolutionary_dynamics_rerun(
    payoffs: MatchupPayoffs,
    players: list[Agent],
    eval_kwargs: dict
) -> None:
    """
    Run evolutionary dynamics evaluation.

    Args:
        payoffs: MatchupPayoffs instance
        players: List of Agent instances
        eval_kwargs: Evaluation-specific kwargs from config
    """
    print("\n" + "="*60)
    print("RUNNING EVOLUTIONARY DYNAMICS")
    print("="*60 + "\n")

    replicator_dynamics = DiscreteReplicatorDynamics(
        players=players,
        matchup_payoffs=payoffs,
    )

    population_history = replicator_dynamics.run_dynamics(
        initial_population=eval_kwargs["initial_population"],
        steps=int(eval_kwargs["steps"]),
        lr_method=eval_kwargs["lr_method"],
        lr_nu=float(eval_kwargs["lr_nu"]),
    )

    LOGGER.log_record(population_history, "population_history.json")

    print("\n" + "="*60 + "\n")


def run_deviation_rating_rerun(
    payoffs: MatchupPayoffs,
    eval_kwargs: dict
) -> None:
    """
    Run deviation rating evaluation.

    Args:
        payoffs: MatchupPayoffs instance (with payoff_tensor already built)
        eval_kwargs: Evaluation-specific kwargs from config
    """
    print("\n" + "="*60)
    print("RUNNING DEVIATION RATING")
    print("="*60 + "\n")

    deviation_rating = DeviationRating(
        matchup_payoffs=payoffs,
        tolerance=float(eval_kwargs["tolerance"]),
    )

    ratings = deviation_rating.compute_ratings()

    print("\nDeviation Ratings:")
    for model, rating in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {rating:.6f}")

    LOGGER.log_record(ratings, "deviation_ratings.json")

    print("\n" + "="*60 + "\n")


def update_config_evaluation(config_file: Path, new_evaluation_config: dict) -> None:
    """
    Update the evaluation.methods field in experiment config.json.

    Args:
        config_file: Path to config.json
        new_evaluation_config: New evaluation config dict (with 'methods' key)
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Mirror the rerun configuration so downstream tooling sees the updated metadata
    config["evaluation"] = new_evaluation_config
    temp_file = config_file.with_suffix(".json.tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    temp_file.replace(config_file)

    print(f"    Updated config.json with new evaluation methods")


def append_to_stdout(experiment_dir: Path, evaluation_methods: list[str]) -> None:
    """
    Append re-evaluation log to stdout.txt with datestamp.

    Args:
        experiment_dir: Path to experiment folder
        evaluation_methods: List of evaluation method types that were run
    """
    stdout_file = experiment_dir / "stdout.txt"

    timestamp = datetime.now().isoformat()
    separator = "\n" + "="*70 + "\n"

    log_message = (
        f"{separator}"
        f"RE-EVALUATION SCRIPT EXECUTED\n"
        f"Timestamp: {timestamp}\n"
        f"Evaluation methods applied: {', '.join(evaluation_methods)}\n"
        f"Previous evaluation results have been overwritten.\n"
        f"{separator}"
    )

    with open(stdout_file, "a", encoding="utf-8") as f:
        f.write(log_message)

    print(f"    Appended re-evaluation log to stdout.txt")


def run_evaluations_on_experiment(
    experiment_dir: Path,
    evaluation_config: dict,
    seed: int
) -> None:
    """
    Re-run evaluation methods on a single experiment folder.

    Args:
        experiment_dir: Path to experiment subfolder
        evaluation_config: Loaded evaluation config dict
        seed: Random seed for reproducibility
    """
    print(f"\nProcessing experiment: {experiment_dir.name}")

    # 1. Load config for agent/game metadata
    config_file = experiment_dir / "config.json"
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 2. Load existing matchup payoffs to avoid re-running the expensive tournament
    print("  Loading matchup payoffs...")
    matchup_payoffs = load_matchup_payoffs(experiment_dir)

    # 3. Build payoff tensor (required for evaluations)
    print("  Building payoff tensor...")
    matchup_payoffs.build_payoff_tensor()

    # 4. Load agents from config so rankings use readable agent labels
    print("  Reconstructing agents...")
    players = load_agents_from_config(config)

    # 5. Set random seed for deterministic replicator dynamics
    set_seed(seed)
    print(f"  Random seed set to: {seed}")

    # 6. Redirect logger output to the experiment directory for easy auditing
    original_log_dir = LOGGER._log_dir
    LOGGER.set_log_dir(experiment_dir)

    evaluation_methods_run = []

    try:
        # 7. Execute every evaluation method declared in the new config
        methods = evaluation_config["methods"]
        print(f"  Running {len(methods)} evaluation method(s)...")

        for eval_method in methods:
            eval_type = eval_method["type"]
            eval_kwargs = eval_method.get("kwargs", {})

            if eval_type == "evolutionary_dynamics":
                print(f"    - Running evolutionary dynamics...")
                run_evolutionary_dynamics_rerun(matchup_payoffs, players, eval_kwargs)
                evaluation_methods_run.append("evolutionary_dynamics")
            elif eval_type == "deviation_rating":
                print(f"    - Running deviation rating...")
                run_deviation_rating_rerun(matchup_payoffs, eval_kwargs)
                evaluation_methods_run.append("deviation_rating")

        # 8. Persist evaluation metadata alongside the experiment
        print("  Updating metadata...")
        update_config_evaluation(config_file, evaluation_config)

        # 9. Leave a visible breadcrumb for future investigators
        append_to_stdout(experiment_dir, evaluation_methods_run)

        print(f"  ✓ Completed {experiment_dir.name}")

    finally:
        # 10. Restore the original logging destination so global state is unchanged
        if original_log_dir:
            LOGGER._log_dir = original_log_dir


def update_batch_config(batch_dir: Path, evaluation_config_path: str) -> None:
    """
    Update batch_config.json to reflect new evaluation_config.

    Args:
        batch_dir: Path to batch root directory
        evaluation_config_path: Path to new evaluation config
    """
    batch_config_file = batch_dir / "batch_config.json"

    with open(batch_config_file, "r", encoding="utf-8") as f:
        batch_config = json.load(f)

    # Record which evaluation config produced the refreshed artifacts
    batch_config["evaluation_config"] = evaluation_config_path
    temp_file = batch_config_file.with_suffix(".json.tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(batch_config, f, indent=2)
    temp_file.replace(batch_config_file)

    print(f"✓ Updated batch_config.json")


def update_yaml_configs(
    batch_dir: Path,
    evaluation_config_path: str,
    processed_experiments: list[str]
) -> None:
    """
    Update configs/*.yaml files to reflect new evaluation_config.

    Args:
        batch_dir: Path to batch root directory
        evaluation_config_path: Path to new evaluation config
        processed_experiments: List of experiment names that were processed
    """
    configs_dir = batch_dir / "configs"

    for exp_name in processed_experiments:
        yaml_file = configs_dir / f"{exp_name}.yaml"

        # Load the saved CLI arguments for this experiment
        with open(yaml_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Point each config at the evaluation spec used during the rerun
        config["evaluation_config"] = evaluation_config_path

        temp_file = yaml_file.with_suffix(".yaml.tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        temp_file.replace(yaml_file)

    print(f"✓ Updated {len(processed_experiments)} YAML config file(s)")


def update_experiments_json(
    batch_dir: Path,
    evaluation_config_path: str,
    processed_experiments: list[str]
) -> None:
    """
    Update experiments.json to reflect new evaluation_config.

    Args:
        batch_dir: Path to batch root directory
        evaluation_config_path: Path to new evaluation config
        processed_experiments: List of experiment names that were processed
    """
    experiments_file = batch_dir / "experiments.json"

    # Keep the experiments index in sync so dashboards show the new evaluation spec
    with open(experiments_file, "r", encoding="utf-8") as f:
        experiments = json.load(f)

    # Only touch experiments we actually processed in this run
    processed_set = set(processed_experiments)
    updated_count = 0

    for experiment in experiments:
        if experiment["experiment_name"] in processed_set:
            experiment["evaluation_config"] = evaluation_config_path
            updated_count += 1

    temp_file = experiments_file.with_suffix(".json.tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2)
    temp_file.replace(experiments_file)

    print(f"✓ Updated experiments.json ({updated_count} experiment(s))")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Re-run evaluation methods on existing batch experiment folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--batch-folders",
        type=str,
        nargs="+",
        required=True,
        help="List of batch experiment folder paths (e.g., data/clean/run2_20260124_025842)"
    )
    parser.add_argument(
        "--evaluation-config",
        type=str,
        required=True,
        help="Path to evaluation config YAML file (relative to config dir or absolute)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for replicator dynamics (default: 42)"
    )

    return parser.parse_args()


def main():
    """Main entry point for re-running evaluations."""
    args = parse_arguments()

    print("="*70)
    print("Re-run Evaluation Methods Script")
    print("="*70)
    print(f"Evaluation config: {args.evaluation_config}")
    print(f"Random seed: {args.seed}")
    print(f"Batch folders: {len(args.batch_folders)}")

    # Use ConfigLoader so relative paths can reference configs/ without extra plumbing
    loader = ConfigLoader()
    evaluation_config = loader.load_component(
        args.evaluation_config,
    )

    print(f"Evaluation methods: {[m['type'] for m in evaluation_config['methods']]}")

    # Fail fast if the supplied evaluation config is incomplete
    validate_evaluation_config(evaluation_config)

    # Iterate through each batch independently so one failure does not abort the rest
    all_successful = []
    all_failed = []

    for batch_folder_str in args.batch_folders:
        batch_dir = Path(batch_folder_str)

        print(f"\n{'='*70}")
        print(f"Processing batch: {batch_dir}")
        print(f"{'='*70}")

        # Skip reputation batches—they require different evaluators than the matchup ones
        experiment_dirs = discover_experiment_folders(batch_dir)
        processed_experiment_names = []

        print(f"\nFound {len(experiment_dirs)} non-reputation experiment(s) to process")

        # Keep going even if one experiment fails; others might still succeed
        for exp_dir in experiment_dirs:
            try:
                run_evaluations_on_experiment(exp_dir, evaluation_config, args.seed)
                all_successful.append(exp_dir.name)
                processed_experiment_names.append(exp_dir.name)
            except Exception as e:
                print(f"✗ ERROR processing {exp_dir.name}: {e}")
                traceback.print_exc()
                all_failed.append(exp_dir.name)

        # Update batch-level metadata
        if processed_experiment_names:
            print(f"\nUpdating batch metadata...")
            update_batch_config(batch_dir, args.evaluation_config)
            update_yaml_configs(batch_dir, args.evaluation_config, processed_experiment_names)
            update_experiments_json(batch_dir, args.evaluation_config, processed_experiment_names)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total experiments processed: {len(all_successful) + len(all_failed)}")
    print(f"  Successful: {len(all_successful)}")
    print(f"  Failed: {len(all_failed)}")

    if all_failed:
        print("\nFailed experiments:")
        for name in all_failed:
            print(f"  - {name}")

    print()

    return 0 if not all_failed else 1


if __name__ == "__main__":
    sys.exit(main())
