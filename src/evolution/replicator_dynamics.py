import random
from typing import Literal

import numpy as np
from tqdm import tqdm

from src.logger_manager import log_record
from src.agents.agent_manager import Agent
from src.mechanisms.base import Mechanism

random.seed(42)


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
        agents: list[Agent],
        mechanism: Mechanism,
    ) -> None:
        self.mechanism = mechanism
        self.agents = agents

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
    ) -> list[np.ndarray]:
        """
        Run the multiplicative weights dynamics for a specified number of steps.
        """

        if lr_method == "constant":
            lr_fct = lambda t: lr_nu
        elif lr_method == "sqrt":
            lr_fct = lambda t: lr_nu / np.sqrt(t)
        else:
            raise ValueError("learning_rate method must be 'constant' or 'sqrt'")

        # Initialize population distribution
        if isinstance(initial_population, np.ndarray):
            assert len(initial_population) == len(
                self.agents
            ), "Initial population distribution must match number of agent types"
            assert np.all(
                initial_population >= 0
            ), "Initial population distribution must be non-negative"
            population = initial_population
        elif initial_population == "random":
            population = np.random.exponential(scale=1.0, size=len(self.agents))
        elif initial_population == "uniform":
            population = np.ones(len(self.agents))
        else:
            raise ValueError("initial_population must be a numpy array or 'uniform'")

        # Normalize to ensure it is a probability distribution
        population /= population.sum()
        population_history = [population.copy()]

        population_payoffs = self.mechanism.run_tournament(agents=self.agents)
        fitness = population_payoffs.fitness(population)

        log_record(record=population_payoffs.to_record(), file_name="payoffs.json")

        for step in tqdm(range(1, steps + 1), desc="Evolution Steps"):

            # average population fitness is the society's average performance
            ave_population_fitness = np.dot(population, fitness)

            if np.max(np.abs(fitness - ave_population_fitness)) < tol:
                print("Converged: approximate equilibrium reached")
                return population_history

            population = self.population_update(
                current_pop=population,
                fitness=fitness,
                ave_population_fitness=ave_population_fitness,
                lr=lr_fct(len(population_history) + 1),
            )
            population_history.append(population.copy())

        print("Steps limit reached")
        return population_history
