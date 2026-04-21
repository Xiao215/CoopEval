#!/usr/bin/env python3
"""Experiment parsing helpers for LaTeX table generation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from coopeval.script_utils.display_helper import (
    make_mechanism_suffix,
    sort_agents,
    sort_games,
    sort_mechanisms,
)
from coopeval.script_utils.result_loader import discover_experiment_subfolders
from coopeval.utils.json_io import load_json
from coopeval.visualization.analysis_utils import (
    is_reputation_mechanism,
    validate_dict_consistency,
    validate_folder_count_consistency,
    validate_list_consistency,
)


@dataclass
class ExperimentData:
    """Represents a single experiment's data."""

    mechanism: str
    mechanism_kwargs: dict
    game: str
    model_scores: dict[str, float]
    folder_path: Path
    game_config: dict
    eval_config: dict
    rd_fitness: dict[str, float] | None = None
    rd_populations: dict[str, float] | None = None
    deviation_ranks: dict[str, str] | None = None


def compute_deviation_ranks(ratings: dict[str, float]) -> dict[str, str]:
    """Convert deviation ratings to rank strings."""
    sorted_models = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    ranks: dict[str, str] = {}
    i = 0
    while i < len(sorted_models):
        rating = sorted_models[i][1]
        tied_models = [sorted_models[i][0]]
        j = i + 1
        while j < len(sorted_models) and sorted_models[j][1] == rating:
            tied_models.append(sorted_models[j][0])
            j += 1

        avg_rank = sum(range(i + 1, j + 1)) / len(tied_models)
        rank_str = (
            str(int(avg_rank)) if avg_rank.is_integer() else f"{avg_rank:.1f}"
        )
        for model in tied_models:
            ranks[model] = rank_str
        i = j

    return ranks


def parse_tournament_result_dir(
    tournament_result_dir: Path, metrics: Sequence[str]
) -> tuple[
    dict[tuple[str, str], list[ExperimentData]], list[tuple[Path, Exception]]
]:
    """Parse a tournament result directory by (game, mechanism)."""
    if not tournament_result_dir.exists():
        raise FileNotFoundError(
            f"Tournament result directory not found: {tournament_result_dir}"
        )
    if not tournament_result_dir.is_dir():
        raise NotADirectoryError(
            f"Path is not a directory: {tournament_result_dir}"
        )

    failed_experiments: list[tuple[Path, Exception]] = []
    subdirs = discover_experiment_subfolders(tournament_result_dir)
    raw_experiments: list[tuple[str, dict, str, ExperimentData]] = []

    for subdir in subdirs:
        try:
            config_path = subdir / "config.json"
            payoff_path = subdir / "agent_average_payoff.json"
            if not config_path.exists() or not payoff_path.exists():
                raise FileNotFoundError(f"Missing required files in {subdir}")

            config = load_json(config_path)
            payoffs = load_json(payoff_path)

            mechanism_type = config["mechanism"]["type"]
            mechanism_kwargs = config["mechanism"]["kwargs"]
            game = config["game"]["type"]
            game_config = config["game"]
            eval_config = config["evaluation"]

            if is_reputation_mechanism(mechanism_type):
                rd_fitness = None
                rd_populations = None
                deviation_ranks = None
            else:
                rd_fitness = None
                rd_populations = None
                deviation_ranks = None

                if "rd" in metrics:
                    rd_fitness_path = (
                        subdir / "replicator_dynamics_fitness.json"
                    )
                    if not rd_fitness_path.exists():
                        raise FileNotFoundError(
                            f"Missing replicator_dynamics_fitness.json in {subdir} for mechanism {mechanism_type}"
                        )
                    rd_fitness_data = load_json(rd_fitness_path)
                    rd_fitness = {
                        model: data["fitness"]
                        for model, data in rd_fitness_data.items()
                    }
                    rd_populations = {
                        model: data["final_population"]
                        for model, data in rd_fitness_data.items()
                    }

                if "dr" in metrics:
                    ratings_path = subdir / "deviation_ratings.json"
                    if not ratings_path.exists():
                        raise FileNotFoundError(
                            f"Missing deviation_ratings.json in {subdir} for mechanism {mechanism_type}"
                        )
                    deviation_ranks = compute_deviation_ranks(
                        load_json(ratings_path)
                    )

            exp_data = ExperimentData(
                mechanism=mechanism_type,
                mechanism_kwargs=mechanism_kwargs,
                game=game,
                model_scores=payoffs,
                folder_path=subdir,
                game_config=game_config,
                eval_config=eval_config,
                rd_fitness=rd_fitness,
                rd_populations=rd_populations,
                deviation_ranks=deviation_ranks,
            )
            raw_experiments.append(
                (mechanism_type, mechanism_kwargs, game, exp_data)
            )
        except Exception as err:  # noqa: BLE001
            # Isolate one malformed experiment so the report can continue.
            print(f"WARNING: Failed to parse experiment in {subdir}")
            print(f"  Error: {type(err).__name__}: {err}")
            failed_experiments.append((subdir, err))

    type_to_kwargs: dict[str, list[dict]] = defaultdict(list)
    for mechanism_type, mechanism_kwargs, _game, _exp in raw_experiments:
        type_to_kwargs[mechanism_type].append(mechanism_kwargs)

    varying_keys_per_type: dict[str, frozenset[str]] = {}
    for mtype, kwargs_list in type_to_kwargs.items():
        all_keys = {k for kw in kwargs_list for k in kw}
        varying = frozenset(
            k
            for k in all_keys
            if len({str(kw.get(k)) for kw in kwargs_list}) > 1
        )
        if varying:
            varying_keys_per_type[mtype] = varying

    grouped: dict[tuple[str, str], list[ExperimentData]] = defaultdict(list)
    for mechanism_type, mechanism_kwargs, game, exp_data in raw_experiments:
        if mechanism_type in varying_keys_per_type:
            suffix = make_mechanism_suffix(
                mechanism_kwargs, varying_keys_per_type[mechanism_type]
            )
            mechanism_key = f"{mechanism_type} ({suffix})"
            exp_data.mechanism = mechanism_key
        else:
            mechanism_key = mechanism_type

        grouped[(game, mechanism_key)].append(exp_data)

    return grouped, failed_experiments


def _relabel_default_mechanism_variants(
    grouped_experiments: dict[tuple[str, str], list[ExperimentData]],
) -> dict[tuple[str, str], list[ExperimentData]]:
    """Add explicit default kwarg suffixes when variant families are mixed in."""
    kwargs_by_base: dict[str, list[dict]] = defaultdict(list)
    for (_game, mechanism), experiments in grouped_experiments.items():
        base_name = mechanism.split(" (", 1)[0]
        kwargs_by_base[base_name].extend(
            exp.mechanism_kwargs for exp in experiments
        )

    varying_keys_by_base: dict[str, frozenset[str]] = {}
    for base_name, kwargs_list in kwargs_by_base.items():
        all_keys = {key for kwargs in kwargs_list for key in kwargs}
        varying = frozenset(
            key
            for key in all_keys
            if len({str(kwargs.get(key)) for kwargs in kwargs_list}) > 1
        )
        if varying:
            varying_keys_by_base[base_name] = varying

    relabeled: dict[tuple[str, str], list[ExperimentData]] = defaultdict(list)
    for (game, mechanism), experiments in grouped_experiments.items():
        base_name = mechanism.split(" (", 1)[0]
        mechanism_key = mechanism
        if (
            " (" not in mechanism
            and base_name in varying_keys_by_base
            and experiments
        ):
            suffix = make_mechanism_suffix(
                experiments[0].mechanism_kwargs,
                varying_keys_by_base[base_name],
            )
            mechanism_key = f"{base_name} ({suffix})"
            for exp in experiments:
                exp.mechanism = mechanism_key

        relabeled[(game, mechanism_key)].extend(experiments)

    return relabeled


def collect_grouped_experiments(
    tournament_result_dirs: Sequence[Path], metrics: Sequence[str]
) -> tuple[
    dict[tuple[str, str], list[ExperimentData]], list[tuple[Path, Exception]]
]:
    """Parse multiple tournament result dirs and merge experiment groups."""
    all_grouped: dict[tuple[str, str], list[ExperimentData]] = defaultdict(list)
    failures: list[tuple[Path, Exception]] = []

    for tournament_result_dir in tournament_result_dirs:
        grouped, failed = parse_tournament_result_dir(
            tournament_result_dir, metrics
        )
        failures.extend(failed)
        for group_key, experiments in grouped.items():
            all_grouped[group_key].extend(experiments)
        print(
            f"Parsed {sum(len(exps) for exps in grouped.values())} experiments from {tournament_result_dir}"
        )

    return _relabel_default_mechanism_variants(all_grouped), failures


def resolve_tournament_result_dirs(paths: Sequence[Path]) -> list[Path]:
    """Expand parent directories so each returned path points to result root."""

    def looks_like_tournament_result_dir(path: Path) -> bool:
        return (path / "batch_config.json").exists()

    resolved: list[Path] = []
    for path in paths:
        if looks_like_tournament_result_dir(path):
            resolved.append(path)
            continue

        child_result_dirs: list[Path] = []
        if path.is_dir():
            for child in path.iterdir():
                if child.is_dir() and looks_like_tournament_result_dir(child):
                    child_result_dirs.append(child)

        if child_result_dirs:
            resolved.extend(child_result_dirs)
        else:
            resolved.append(path)

    return resolved


def validate_metric_consistency(
    experiments: list[ExperimentData],
    metric_name: str,
    group_key: tuple[str, str],
) -> bool:
    """Ensure metrics are either present for all reps or absent for all."""
    values = (
        [exp.rd_fitness for exp in experiments]
        if metric_name == "rd_fitness"
        else [exp.deviation_ranks for exp in experiments]
    )

    none_count = sum(1 for value in values if value is None)
    if none_count == len(values):
        _, mechanism = group_key
        if not is_reputation_mechanism(mechanism):
            raise ValueError(
                f"Metric {metric_name} missing for non-reputation mechanism "
                f"{group_key}"
            )
        return False
    if none_count == 0:
        return True

    folder_paths = [str(exp.folder_path.name) for exp in experiments]
    none_indices = [i for i, value in enumerate(values) if value is None]
    non_none_indices = [
        i for i, value in enumerate(values) if value is not None
    ]
    raise ValueError(
        f"Inconsistent {metric_name} data in {group_key}:\n"
        f"  Folders with None: {[folder_paths[i] for i in none_indices]}\n"
        f"  Folders with data: {[folder_paths[i] for i in non_none_indices]}\n"
        "  All experiments in a group must consistently have or lack this metric."
    )


def validate_experiment_groups(
    grouped_experiments: dict[tuple[str, str], list[ExperimentData]],
) -> tuple[list[str], list[str], list[str], dict]:
    """Validate experiment groups and extract canonical metadata."""
    expected_folder_count = validate_folder_count_consistency(
        grouped_experiments
    )
    print(
        f"All groups have {expected_folder_count} folder(s) - validation passed"
    )

    canonical_mechanisms = set()
    canonical_games = set()
    canonical_models: list[str] | None = None
    canonical_eval_config = None

    for group_key, experiments in grouped_experiments.items():
        game, mechanism = group_key
        canonical_mechanisms.add(mechanism)
        canonical_games.add(game)

        folder_paths = [str(exp.folder_path) for exp in experiments]
        model_lists = [sorted(exp.model_scores) for exp in experiments]
        eval_configs = [exp.eval_config for exp in experiments]
        game_configs = [exp.game_config for exp in experiments]

        validated_models = validate_list_consistency(
            model_lists, folder_paths, group_key, "model list"
        )
        validate_dict_consistency(
            eval_configs, folder_paths, group_key, "evaluation config"
        )
        validate_dict_consistency(
            game_configs, folder_paths, group_key, "game config"
        )

        if canonical_models is None:
            canonical_models = validated_models
            canonical_eval_config = eval_configs[0]
        elif validated_models != canonical_models:
            raise ValueError(
                f"Model list mismatch across groups:\n"
                f"  Expected: {canonical_models}\n"
                f"  Got in {group_key}: {validated_models}"
            )

    return (
        sort_mechanisms(canonical_mechanisms),
        sort_games(canonical_games),
        sort_agents(canonical_models or []),
        canonical_eval_config or {},
    )
