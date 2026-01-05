# """Tests for discrete replicator dynamics."""

# import unittest
# import numpy as np
# from typing import Sequence

# from src.ranking_evaluations.replicator_dynamics import DiscreteReplicatorDynamics
# from src.ranking_evaluations.population_payoffs import PopulationPayoffs
# from src.mechanisms.base import Mechanism
# from tests.fakes.general_fakes import FakeAction, FakeAgent
# from tests.fakes.fake_move import FakeMove
# from tests.fakes.fake_mechanism import FakeMechanism


# bimatrix = [
#     [(1, 1), (0.5, 2), (0, 0)],  # LLM1 vs LLM1, LLM2, LLM3
#     [(2, 0.5), (0, 0), (0, 0)],  # LLM2 vs LLM1, LLM2, LLM3
#     [(0, 0), (0, 0), (2, 2)],    # LLM3 vs LLM1, LLM2, LLM3
# ]


# class TestReplicatorDynamics(unittest.TestCase):
#     """Test discrete replicator dynamics with different initial distributions."""

#     def setUp(self):
#         """Set up the agents and the payoff matrix data."""

#         self.agent_cfgs = [
#             {"llm": {"model": "LLM1"}},
#             {"llm": {"model": "LLM2"}},
#             {"llm": {"model": "LLM3"}},
#         ]

#         # 2. Create the Agents manually (Source of Truth)
#         self.players = []
#         for i, cfg in enumerate(self.agent_cfgs):
#             self.players.append(
#                 FakeAgent(uid=i, model_type=cfg["llm"]["model"])
#             )

#         # 3. Build the Payoff Table externally
#         self.precomputed_payoffs = self._construct_bimatrix_payoffs()
#         self.mechanism = FakeMechanism(
#             precomputed_payoffs=self.precomputed_payoffs
#         )

#     def _construct_bimatrix_payoffs(self) -> PopulationPayoffs:
#         """Helper to construct the 3x3 Rock-Paper-Scissors style matrix."""
#         payoffs = PopulationPayoffs(players=self.players)

#         # Matrix Definition:
#         #               FakeAgent1      FakeAgent2      FakeAgent3
#         # FakeAgent1: (1,1)          (0.5,2)          (0,0)
#         # FakeAgent2: (2,0.5)        (0,0)            (0,0)
#         # FakeAgent3: (0,0)          (0,0)            (2,2)
#         moves_over_rounds = [
#             [
#                 FakeMove(uid=0, points=1.0, action=FakeAction.HOLD),
#                 FakeMove(uid=1, points=1.0, action=FakeAction.PASS),
#             ],
#         ]

#         for i, model_i in enumerate(players_by_model.keys()):
#             for j, model_j in enumerate(players_by_model.keys()):
#                 row_val, col_val = bimatrix[i][j]

#                 # Select distinct agents
#                 if model_i == model_j:
#                     p1 = players_by_model[model_i][0]
#                     p2 = players_by_model[model_i][1]
#                 else:
#                     p1 = players_by_model[model_i][0]
#                     p2 = players_by_model[model_j][0]

#                 moves = [
#                     make_fake_move(p1.uid, row_val, FakeAction.HOLD),
#                     make_fake_move(p2.uid, col_val, FakeAction.PASS),
#                 ]
#                 payoffs.add_profile([moves])

#         return payoffs

#     def test_uniform_start(self):
#         """Test starting from uniform distribution."""
#         print("\n=== Uniform Start ===")
#         dynamics = DiscreteReplicatorDynamics(
#             agent_cfgs=self.agent_cfgs, mechanism=self.mechanism
#         )

#         history = dynamics.run_dynamics(
#             initial_population="uniform", steps=100, lr_nu=0.1
#         )

#         self.assertTrue(
#             np.allclose(
#                 [history[0]["LLM1"], history[0]["LLM2"], history[0]["LLM3"]],
#                 [1 / 3, 1 / 3, 1 / 3],
#             )
#         )
#         print(f"Initial: {history[0]}")
#         print(f"Final Distribution: {history[-1]}")

#     def test_closetollm1_start(self):
#         """Test starting from LLM1 skewed distribution."""
#         print("\n=== LLM1 Start ===")
#         dynamics = DiscreteReplicatorDynamics(
#             agent_cfgs=self.agent_cfgs, mechanism=self.mechanism
#         )

#         target_init = np.array([0.9, 0.05, 0.05])
#         history = dynamics.run_dynamics(
#             initial_population=target_init, steps=100, lr_nu=0.1
#         )

#         self.assertTrue(
#             np.allclose(
#                 [history[0]["LLM1"], history[0]["LLM2"], history[0]["LLM3"]],
#                 target_init,
#             )
#         )
#         print(f"Initial: {history[0]}")
#         print(f"Final Distribution: {history[-1]}")


# if __name__ == "__main__":
#     unittest.main()
