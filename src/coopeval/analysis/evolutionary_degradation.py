"""Evolutionary degradation analysis for CoopEval tournament outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from coopeval.ranking_evaluations.matchup_payoffs import MatchupPayoffs
from coopeval.script_utils.display_helper import format_model_name
from coopeval.script_utils.result_loader import TournamentData


@dataclass
class ExperimentData:
    """Data for a single experiment run."""

    exp_path: Path
    game: str
    mechanism: str
    game_config: dict[str, Any]
    initial_payoffs: dict[str, float]
    final_fitness: dict[str, float]
    final_populations: dict[str, float]
    population_history: list[dict[str, float]]
    matchup_payoffs_data: dict[str, Any]


@dataclass
class DegradedAgent:
    """Information about an agent that degraded during evolution."""

    agent_name: str
    short_name: str
    initial_payoff: float
    initial_rank: int
    final_fitness: float
    final_population: float
    collapse_index: float
    fitness_trajectory: list[float]


@dataclass
class CandidateExperiment:
    """An experiment with one or more degraded agents."""

    exp_path: Path
    game: str
    mechanism: str
    degraded_agents: list[DegradedAgent]
    max_collapse_index: float
    avg_collapse_index: float

    @property
    def num_degraded(self) -> int:
        """Return the number of degraded agents in this experiment."""
        return len(self.degraded_agents)


@dataclass
class AnalyzedExperiment:
    """Loaded experiment data plus computed degradation signals."""

    exp_data: ExperimentData
    degraded_agents: list[DegradedAgent]
    fitness_trajectories: dict[str, list[float]]


def load_experiment_data(data: TournamentData) -> ExperimentData:
    """Load the JSON artifacts needed for degradation analysis."""

    initial_payoffs = data.load_json("agent_average_payoff.json")
    rd_fitness_data = data.load_json("replicator_dynamics_fitness.json")
    population_history = data.load_json("population_history.json")
    matchup_payoffs_data = data.load_json("matchup_payoffs.json")

    final_fitness = {
        agent: data["fitness"] for agent, data in rd_fitness_data.items()
    }
    final_populations = {
        agent: data["final_population"]
        for agent, data in rd_fitness_data.items()
    }

    return ExperimentData(
        exp_path=data.path,
        game=data.game,
        mechanism=data.mechanism,
        game_config=data.config["game"],
        initial_payoffs=initial_payoffs,
        final_fitness=final_fitness,
        final_populations=final_populations,
        population_history=population_history,
        matchup_payoffs_data=matchup_payoffs_data,
    )


def compute_fitness_trajectory(
    population_history: list[dict[str, float]],
    matchup_payoffs_data: dict[str, Any],
) -> dict[str, list[float]]:
    """Compute each agent's fitness at every recorded population step."""

    matchup_obj = MatchupPayoffs.from_json(matchup_payoffs_data)
    matchup_obj.build_payoff_tensor()
    agent_types = list(matchup_obj.agent_types)
    fitness_trajectories = {agent: [] for agent in agent_types}

    for pop_dict in population_history:
        total = sum(pop_dict[agent] for agent in agent_types)
        if not np.isclose(total, 1.0, atol=1e-2):
            raise ValueError(
                "Population shares sum to "
                f"{total:.3f}, which is not close to 1."
            )
        normalized_pop = {
            agent: pop_dict[agent] / total for agent in agent_types
        }
        fitness_dict = matchup_obj.fitness(normalized_pop)

        for agent in agent_types:
            fitness_trajectories[agent].append(fitness_dict[agent])

    return fitness_trajectories


def compute_initial_ranks(payoffs: dict[str, float]) -> dict[str, int]:
    """Rank agents by initial payoff, where 1 is highest."""

    sorted_agents = sorted(
        payoffs.items(), key=lambda item: item[1], reverse=True
    )
    return {
        agent: rank
        for rank, (agent, _payoff) in enumerate(sorted_agents, start=1)
    }


def compute_collapse_index(
    agent: str,
    initial_payoffs: dict[str, float],
    final_populations: dict[str, float],
) -> float:
    """Score how starkly a high-payoff agent loses population mass."""

    ranks = compute_initial_ranks(initial_payoffs)
    n_agents = len(initial_payoffs)
    rank = ranks[agent]
    rank_percentile = 1.0 - (rank - 1) / max(n_agents - 1, 1)
    uniform_share = 1.0 / n_agents
    final_pop = final_populations[agent]
    pop_collapse_fraction = max(
        0.0, (uniform_share - final_pop) / uniform_share
    )
    return rank_percentile * pop_collapse_fraction


def identify_degraded_agents(
    exp_data: ExperimentData,
    fitness_trajectories: dict[str, list[float]],
    *,
    min_initial_rank: float = 0.5,
    max_final_pop: float = 0.10,
) -> list[DegradedAgent]:
    """Find agents that start in the top payoff percentile and end rare."""

    ranks = compute_initial_ranks(exp_data.initial_payoffs)
    n_agents = len(exp_data.initial_payoffs)
    degraded = []

    for agent, initial_payoff in exp_data.initial_payoffs.items():
        rank = ranks[agent]
        rank_percentile = 1.0 - (rank - 1) / max(n_agents - 1, 1)
        final_pop = exp_data.final_populations[agent]

        if rank_percentile >= min_initial_rank and final_pop < max_final_pop:
            degraded.append(
                DegradedAgent(
                    agent_name=agent,
                    short_name=format_model_name(agent),
                    initial_payoff=initial_payoff,
                    initial_rank=rank,
                    final_fitness=exp_data.final_fitness[agent],
                    final_population=final_pop,
                    collapse_index=compute_collapse_index(
                        agent,
                        exp_data.initial_payoffs,
                        exp_data.final_populations,
                    ),
                    fitness_trajectory=fitness_trajectories[agent],
                )
            )

    degraded.sort(key=lambda agent: agent.collapse_index, reverse=True)
    return degraded


def analyze_tournament_data(
    data: TournamentData,
    *,
    min_initial_rank: float,
    max_final_pop: float,
) -> AnalyzedExperiment | None:
    """Load one tournament and compute degradation signals."""

    exp_data = load_experiment_data(data)
    fitness_trajectories = compute_fitness_trajectory(
        exp_data.population_history, exp_data.matchup_payoffs_data
    )
    degraded_agents = identify_degraded_agents(
        exp_data,
        fitness_trajectories,
        min_initial_rank=min_initial_rank,
        max_final_pop=max_final_pop,
    )
    if not degraded_agents:
        return None
    return AnalyzedExperiment(
        exp_data=exp_data,
        degraded_agents=degraded_agents,
        fitness_trajectories=fitness_trajectories,
    )


def candidate_from_analysis(
    analyzed: AnalyzedExperiment,
) -> CandidateExperiment:
    """Build a sortable discovery candidate from analyzed experiment data."""

    degraded_agents = analyzed.degraded_agents
    max_collapse = max(agent.collapse_index for agent in degraded_agents)
    avg_collapse = sum(agent.collapse_index for agent in degraded_agents) / len(
        degraded_agents
    )
    exp_data = analyzed.exp_data
    return CandidateExperiment(
        exp_path=exp_data.exp_path,
        game=exp_data.game,
        mechanism=exp_data.mechanism,
        degraded_agents=degraded_agents,
        max_collapse_index=max_collapse,
        avg_collapse_index=avg_collapse,
    )


def degradation_summary_rows(
    analyzed: AnalyzedExperiment,
) -> list[dict[str, Any]]:
    """Build CSV rows for the degraded agents in one analyzed experiment."""

    exp_data = analyzed.exp_data
    return [
        {
            "experiment_path": str(exp_data.exp_path),
            "game": exp_data.game,
            "mechanism": exp_data.mechanism,
            "agent_name": agent.agent_name,
            "short_name": agent.short_name,
            "initial_payoff": agent.initial_payoff,
            "initial_rank": agent.initial_rank,
            "final_fitness": agent.final_fitness,
            "final_population": agent.final_population,
            "collapse_index": agent.collapse_index,
        }
        for agent in analyzed.degraded_agents
    ]
