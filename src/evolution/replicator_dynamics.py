"""Utilities for running discrete-time replicator dynamics tournaments."""

from typing import Literal, Sequence

import numpy as np
from tqdm import tqdm

from src.logger_manager import LOGGER
from src.mechanisms.base import Mechanism


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
        agent_cfgs: Sequence[dict],
        mechanism: Mechanism,
    ) -> None:
        """Bind a population of ``agents`` to a tournament ``mechanism``."""
        self.mechanism = mechanism
        self.agent_cfgs = agent_cfgs

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

    def run_dynamics(
        self,
        initial_population: np.ndarray | str = "uniform",
        steps: int = 1000,
        tol: float = 1e-6,
        *,
        lr_method: Literal["constant", "sqrt"] = "constant",
        lr_nu: float = 0.1,
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
                initial_population >= 0
            ), "Initial population distribution must be non-negative"
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

        population_payoffs = self.mechanism.run_tournament(
            agent_cfgs=self.agent_cfgs
        )
        model_average_payoff = population_payoffs.model_average_payoff()

        print(f"Model average payoff: {model_average_payoff}")

        LOGGER.log_record(
            record=model_average_payoff, file_name="model_average_payoff.json"
        )

        for step in tqdm(range(1, steps + 1), desc="Evolution Steps"):
            population_dict = {
                model: float(prob)
                for model, prob in zip(model_types, population)
            }
            fitness_dict = population_payoffs.fitness(population_dict)
            print(f"Step {step}: Population fitness is {fitness_dict}")
            fitness = np.array([fitness_dict[model] for model in model_types])
            ave_population_fitness = float(np.dot(population, fitness))

            if np.max(np.abs(fitness - ave_population_fitness)) < tol:
                print("Converged: approximate equilibrium reached")
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

        return population_history
