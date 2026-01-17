"""Utilities for running discrete-time replicator dynamics tournaments."""

from typing import Literal

import numpy as np
from tqdm import tqdm

from src.logger_manager import LOGGER
from src.ranking_evaluations.matchup_payoffs import MatchupPayoffs


class DiscreteReplicatorDynamics:
    """
    Discrete-time replicator dynamics using exponential weight updates.

    This implements the update rule:
    x_i(t+1) = x_i(t) * exp(η * (f_i - f_avg)) / Z(t)

    where η is the learning rate and Z(t) is the normalization constant.
    For learning rate going to zero, this approaches the continuous-time replicator dynamics.
    """

    def __init__(
        self,
        agent_cfgs: list[dict],
        *,
        matchup_payoffs: MatchupPayoffs | None = None,
    ) -> None:
        """Bind a population of ``agents`` to the tournament payoffs."""
        self.agent_cfgs = agent_cfgs
        self.matchup_payoffs = matchup_payoffs

    def population_update(
        self,
        current_pop: np.ndarray,
        fitness: np.ndarray,
        ave_population_fitness: float,
        lr: float,
    ) -> np.ndarray:
        """
        Args:
            current_dist: numpy array, current probability distribution over agent types
            fitness: numpy array, fitness of each agent type against current distribution
            avg_fitness: float, current average performance
            t: int, current time step (must be > 0)

        Returns:
            numpy array, next step's probability distribution over agent types
        """
        weights = current_pop * np.exp(lr * (fitness - ave_population_fitness))
        next_pop = weights / np.sum(weights)
        return next_pop

    def _log_final_fitness(
        self,
        population: np.ndarray,
        model_types: list[str],
        matchup_payoffs: MatchupPayoffs,
    ) -> None:
        """
        Compute and log final fitness values and population shares.

        Args:
            population: Final population distribution
            model_types: List of model type names
            matchup_payoffs: MatchupPayoffs instance for computing fitness
        """
        # Compute final fitness values using final population
        final_population_dict = {
            model: float(prob)
            for model, prob in zip(model_types, population)
        }
        final_fitness_dict = matchup_payoffs.fitness(final_population_dict)

        # Create combined record with both fitness and population for human readability
        rd_fitness_record = {
            model: {
                "fitness": final_fitness_dict[model],
                "final_population": final_population_dict[model]
            }
            for model in model_types
        }

        # Log fitness values and population to new file
        LOGGER.log_record(
            record=rd_fitness_record,
            file_name="replicator_dynamics_fitness.json"
        )

    def run_dynamics(
        self,
        initial_population: np.ndarray | str = "uniform",
        steps: int = 1000,
        tol: float = 1e-6,
        *,
        lr_method: Literal["constant", "sqrt"] = "constant",
        lr_nu: float = 0.1,
        matchup_payoffs: MatchupPayoffs | None = None,
    ) -> list[dict[str, float]]:
        """
        Run the multiplicative weights dynamics for a specified number of steps.
        """

        if lr_method == "constant":
            lr_fct = lambda t: lr_nu
        elif lr_method == "sqrt":
            lr_fct = lambda t: lr_nu / np.sqrt(t)
        else:
            raise ValueError(
                "learning_rate method must be 'constant' or 'sqrt'"
            )

        # Initialize population distribution
        if isinstance(initial_population, np.ndarray):
            assert len(initial_population) == len(
                self.agent_cfgs
            ), "Initial population distribution must match number of agent types"
            assert np.all(
                initial_population > 0
            ), "Initial population distribution must be positive everywhere; zero probabilities stay zero in Replicator Dynamics!"
            population = initial_population
        elif initial_population == "random":
            population = np.random.exponential(
                scale=1.0, size=len(self.agent_cfgs)
            )
        elif initial_population == "uniform":
            population = np.ones(len(self.agent_cfgs))
        else:
            raise ValueError(
                "initial_population must be a numpy array or 'uniform'"
            )

        # Normalize to ensure it is a probability distribution
        population /= population.sum()
        model_types = [
            str(agent_config["llm"]["model"])
            for agent_config in self.agent_cfgs
        ]
        population_history = [
            {model: float(prob) for model, prob in zip(model_types, population)}
        ]

        matchup_payoffs = matchup_payoffs or self.matchup_payoffs
        if matchup_payoffs is None:
            raise ValueError(
                "Matchup payoffs must be provided before running dynamics."
            )

        if matchup_payoffs._payoff_tensor is None:
            matchup_payoffs.build_payoff_tensor()
        model_average_payoff = matchup_payoffs.model_average_payoff()

        LOGGER.log_record(
            record=model_average_payoff, file_name="model_average_payoff.json"
        )

        for step in tqdm(range(1, steps + 1), desc="Evolution Steps"):
            population_dict = {
                model: float(prob)
                for model, prob in zip(model_types, population)
            }
            fitness_dict = matchup_payoffs.fitness(population_dict)
            if step % 10 == 0 or step == 1:
                print(f"Step {step}: Population fitness is {fitness_dict}")
            fitness = np.array([fitness_dict[model] for model in model_types])
            ave_population_fitness = float(np.dot(population, fitness))

            if np.max(np.abs(fitness - ave_population_fitness)) < tol:
                print("Converged: approximate equilibrium reached")
                self._log_final_fitness(population, model_types, matchup_payoffs)
                return population_history

            population = self.population_update(
                current_pop=population,
                fitness=fitness,
                ave_population_fitness=ave_population_fitness,
                lr=lr_fct(step),
            )
            population_history.append(
                {
                    model: float(prob)
                    for model, prob in zip(model_types, population)
                }
            )

        # Log final fitness values after reaching max steps
        self._log_final_fitness(population, model_types, matchup_payoffs)
        return population_history
