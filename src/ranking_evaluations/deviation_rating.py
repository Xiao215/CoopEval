"""Deviation Rating: A clone-invariant rating method for N-player general-sum games.

Reimplementation of "Deviation Ratings: A General, Clone Invariant Rating Method"
(arXiv:2502.11645) by Marris et al., 2025.
"""

from itertools import product

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.ranking_evaluations.population_payoffs import PopulationPayoffs


class DeviationRating:
    """
    Compute deviation ratings for strategies in symmetric N-player games.

    Deviation ratings are based on coarse correlated equilibria (CCE) and select
    for the strictest equilibrium by iteratively minimizing maximum deviation gains.
    """

    def __init__(
        self, population_payoffs: PopulationPayoffs, tolerance: float = 1e-14
    ) -> None:
        """
        Initialize deviation rating computation.

        Args:
            population_payoffs: PopulationPayoffs instance with built payoff tensor.
            tolerance: Base tolerance for numerical comparisons (will be scaled by payoff range).
        """
        # Validate that tensor has been built
        if population_payoffs._payoff_tensor is None:
            raise ValueError(
                "Must call build_payoff_tensor() before creating DeviationRating"
            )
        if population_payoffs._tensor_model_types is None:
            raise ValueError(
                "PopulationPayoffs must have _tensor_model_types populated"
            )

        # Extract game parameters
        self.n_players = population_payoffs._payoff_tensor.ndim
        self.n_strategies = population_payoffs._payoff_tensor.shape[0]
        self.model_types = population_payoffs._tensor_model_types

        if len(self.model_types) != self.n_strategies:
            raise ValueError(
            "Number of model types must match number of strategies in payoff tensor; got "
            f"{len(self.model_types)} model types and {self.n_strategies} strategies"
        )

        # Build full payoff tensor for all players
        self.G = population_payoffs.build_full_payoff_tensor()

        # Store tolerance
        self.base_tolerance = tolerance

        # Generate joint strategy space for reference
        self.joint_strategies = list(
            product(range(self.n_strategies), repeat=self.n_players)
        )

    def _build_deviation_matrix(self, G: np.ndarray) -> np.ndarray:
        """
        Construct the deviation matrix M.

        For each player p and deviation strategy s, computes the deviation gain
        when deviating to s from each joint strategy.

        Args:
            G: Full payoff tensor of shape (N, S^N).

        Returns:
            Deviation matrix M of shape (N×S, S^N) where:
            M[(p×S + s), a] = G_p(s, a_{-p}) - G_p(a)
        """
        N = self.n_players
        S = self.n_strategies
        n_joint_strategies = len(self.joint_strategies)

        # Initialize deviation matrix
        M = np.zeros((N * S, n_joint_strategies), dtype=float)

        # For each player and deviation strategy
        for player_idx in range(N):
            for deviation_strat in range(S):
                row_idx = player_idx * S + deviation_strat

                # For each joint strategy
                for joint_strat_idx, joint_strat in enumerate(self.joint_strategies):
                    # Construct deviation: replace player's strategy with deviation_strat
                    deviation_joint_strat = list(joint_strat)
                    deviation_joint_strat[player_idx] = deviation_strat

                    # Find index of deviation joint strategy
                    deviation_joint_strat_idx = self.joint_strategies.index(
                        tuple(deviation_joint_strat)
                    )

                    # M[(p,s), a] = G_p(s, a_{-p}) - G_p(a)
                    M[row_idx, joint_strat_idx] = (
                        G[player_idx, deviation_joint_strat_idx]
                        - G[player_idx, joint_strat_idx]
                    )

        return M

    def _compute_relative_tolerance(self, G: np.ndarray) -> float:
        """
        Compute tolerance relative to the payoff range.

        Args:
            G: Full payoff tensor of shape (N, S^N).

        Returns:
            Relative tolerance scaled by payoff range.
        """
        payoff_range = float(np.max(G) - np.min(G))
        return self.base_tolerance * payoff_range

    def _run_iterative_lp(self, M: np.ndarray, rel_tol: float) -> np.ndarray:
        """
        Run the iterative LP algorithm (Algorithm 1) to compute raw ratings.

        Args:
            M: Deviation matrix of shape (N×S, S^N).
            rel_tol: Relative tolerance for identifying active constraints.

        Returns:
            Flattened ratings array of shape (N×S,).
        """
        N = self.n_players
        S = self.n_strategies
        n_joint_strategies = len(self.joint_strategies)

        # Initialize ratings and tracking
        ratings = np.zeros(N * S, dtype=float)
        is_rated = np.zeros(N * S, dtype=bool)
        active_set_count = 0
        iteration = 0
        max_iterations = N * S

        # Iterative LP algorithm (Algorithm 1)
        while active_set_count < N * S:
            iteration += 1

            # Create Gurobi model
            model = gp.Model("deviation_rating")
            # model.setParam("OutputFlag", 0)  # Suppress output
            model.setParam("Method", 2)  # Dual simplex

            # Variables
            sigma = model.addMVar(
                n_joint_strategies, lb=0.0, name="sigma"
            )  # Joint distribution
            epsilon = model.addVar(
                lb=-GRB.INFINITY, name="epsilon"
            )  # Max deviation gain

            # Constraint: probability simplex
            model.addConstr(sigma.sum() == 1.0, name="simplex")

            # Compute M @ sigma
            M_sigma = M @ sigma

            # Constraints for rated strategies (equality) and unrated strategies (inequality)
            for i in range(N * S):
                if is_rated[i]:
                    model.addConstr(
                        M_sigma[i] == ratings[i],
                        name=f"rated_{i}",
                    )
                else:
                    model.addConstr(
                        M_sigma[i] <= epsilon,
                        name=f"unrated_{i}",
                    )

            # Objective: minimize epsilon
            model.setObjective(epsilon, GRB.MINIMIZE)

            # Solve
            model.optimize()

            if model.status != GRB.OPTIMAL:
                raise RuntimeError(
                    f"LP solver failed with status {model.status}. "
                    "This should not happen for CCE problems."
                )

            # Extract solution
            sigma_star = sigma.X
            epsilon_star = epsilon.X

            # Identify active constraints
            delta = M @ sigma_star

            newly_active = 0
            for i in range(N * S):
                if not is_rated[i]:
                    if abs(delta[i] - epsilon_star) <= rel_tol:
                        is_rated[i] = True
                        ratings[i] = epsilon_star
                        active_set_count += 1
                        newly_active += 1

            if newly_active == 0:
                raise RuntimeError(
                    "No new constraints became active in LP iteration. "
                    "This indicates a numerical issue."
                )

        print(f"Iterative LP converged in {iteration} iterations (max: {max_iterations})")
        return ratings

    def _verify_and_extract_ratings(
        self, ratings: np.ndarray, rel_tol: float
    ) -> dict[str, float]:
        """
        Verify symmetry and extract final ratings as a dictionary.

        Args:
            ratings: Flattened ratings array of shape (N×S,).
            rel_tol: Relative tolerance for symmetry verification.

        Returns:
            Dictionary mapping model type names to their deviation ratings.
        """
        N = self.n_players
        S = self.n_strategies

        # Reshape to (N, S) matrix
        ratings_matrix = ratings.reshape((N, S))

        # Verify symmetry across players
        for s in range(S):
            player_ratings = ratings_matrix[:, s]
            max_deviation = float(
                np.max(np.abs(player_ratings - player_ratings[0]))
            )
            if max_deviation > rel_tol:
                raise ValueError(
                    f"Symmetry verification failed for strategy {s}. "
                    f"Max deviation across players: {max_deviation} (tolerance: {rel_tol}). "
                    f"Ratings: {player_ratings}"
                )

        # Extract ratings for Player 1 (all players should have same ratings)
        final_ratings = ratings_matrix[0, :]

        # Return as dictionary mapping model types to ratings
        return {
            self.model_types[s]: float(final_ratings[s]) for s in range(S)
        }

    def compute_ratings(self) -> dict[str, float]:
        """
        Compute deviation ratings using Algorithm 1 from the paper.

        Returns:
            Dictionary mapping model type names to their deviation ratings.
        """
        # Build deviation matrix
        M = self._build_deviation_matrix(self.G)

        # Compute relative tolerance
        rel_tol = self._compute_relative_tolerance(self.G)

        # Run iterative LP to get raw ratings
        ratings = self._run_iterative_lp(M, rel_tol)

        # Verify symmetry and extract final ratings
        return self._verify_and_extract_ratings(ratings, rel_tol)
