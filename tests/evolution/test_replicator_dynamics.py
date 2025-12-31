"""Tests for discrete replicator dynamics."""

import unittest
import numpy as np
from typing import Sequence

from src.evolution.replicator_dynamics import DiscreteReplicatorDynamics
from src.evolution.population_payoffs import PopulationPayoffs
from src.mechanisms.base import Mechanism
from tests.fakes.general_fakes import MockAction, make_move


class DummyMechanism(Mechanism):
    """Dummy mechanism with hardcoded 3x3 bimatrix payoffs."""

    def __init__(self):
        """Initialize without a base game."""
        pass

    def _play_matchup(self, players, payoffs):
        """Not used in this dummy mechanism."""
        pass

    def run_tournament(self, agent_cfgs: Sequence[dict]) -> PopulationPayoffs:
        """Return PopulationPayoffs with the hardcoded bimatrix:

              LLM1      LLM2      LLM3
        LLM1: (1,1)   (0.5,2)   (0,0)
        LLM2: (2,0.5) (0,0)     (0,0)
        LLM3: (0,0)   (0,0)     (2,2)
        """
        # Create 6 agents (2 per model type) with UIDs 0-5
        uids_to_model_types = {
            0: "LLM1", 1: "LLM1",
            2: "LLM2", 3: "LLM2",
            4: "LLM3", 5: "LLM3",
        }

        payoffs = PopulationPayoffs(uids_to_model_types=uids_to_model_types)

        # Hardcoded symmetric bimatrix
        bimatrix = [
            [(1, 1), (0.5, 2), (0, 0)],  # LLM1 vs LLM1, LLM2, LLM3
            [(2, 0.5), (0, 0), (0, 0)],  # LLM2 vs LLM1, LLM2, LLM3
            [(0, 0), (0, 0), (2, 2)],    # LLM3 vs LLM1, LLM2, LLM3
        ]

        # Add all 9 matchups
        uid_map = {"LLM1": (0, 1), "LLM2": (2, 3), "LLM3": (4, 5)}
        models = ["LLM1", "LLM2", "LLM3"]

        for i, model_i in enumerate(models):
            for j, model_j in enumerate(models):
                row_payoff, col_payoff = bimatrix[i][j]
                uid_i = uid_map[model_i][0]
                uid_j = uid_map[model_j][1] if i == j else uid_map[model_j][0]

                moves = [
                    make_move(uid_i, row_payoff, MockAction.HOLD),
                    make_move(uid_j, col_payoff, MockAction.PASS),
                ]
                payoffs.add_profile([moves])

        return payoffs


class TestReplicatorDynamics(unittest.TestCase):
    """Test discrete replicator dynamics with different initial distributions."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent_cfgs = [
            {"llm": {"model": "LLM1"}},
            {"llm": {"model": "LLM2"}},
            {"llm": {"model": "LLM3"}},
        ]
        self.mechanism = DummyMechanism()

    def test_uniform_start(self):
        """Test starting from uniform distribution."""
        print("\n=== Uniform Start ===")
        dynamics = DiscreteReplicatorDynamics(
            agent_cfgs=self.agent_cfgs, mechanism=self.mechanism
        )
        history = dynamics.run_dynamics(
            initial_population="uniform", steps=100, lr_nu=0.1
        )

        # Check initial distribution is uniform
        self.assertTrue(
            np.allclose(
                [history[0]["LLM1"], history[0]["LLM2"], history[0]["LLM3"]],
                [1 / 3, 1 / 3, 1 / 3],
            )
        )

        print(f"Initial: {history[0]}")
        print(f"Final:   {history[-1]}")

    def test_closetollm1_start(self):
        """Test starting from LLM1 only."""
        print("\n=== LLM1 Start ===")
        dynamics = DiscreteReplicatorDynamics(
            agent_cfgs=self.agent_cfgs, mechanism=self.mechanism
        )
        history = dynamics.run_dynamics(
            initial_population=np.array([0.9, 0.05, 0.05]), steps=100, lr_nu=0.1
        )

        # Check initial distribution is all LLM1
        self.assertTrue(
            np.allclose(
                [history[0]["LLM1"], history[0]["LLM2"], history[0]["LLM3"]],
                [0.9, 0.05, 0.05],
            )
        )

        print(f"Initial: {history[0]}")
        print(f"Final:   {history[-1]}")


if __name__ == "__main__":
    unittest.main()
