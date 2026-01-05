# """Tests for deviation rating."""

# import unittest
# from typing import Sequence

# from src.ranking_evaluations.deviation_rating import DeviationRating
# from src.ranking_evaluations.population_payoffs import PopulationPayoffs
# from src.mechanisms.base import Mechanism
# from tests.fakes.general_fakes import FakeAction, make_move


# class BiasedShapleyMechanism(Mechanism):
#     """Mechanism with biased Shapley payoffs from Table 2 of the paper.

#     This is a symmetric 2-player game with 4 strategies (R, P, S, N):
#     - R, P, S form a biased rock-paper-scissors cycle
#     - N is the Nash equilibrium mixture
#     """

#     def __init__(self):
#         """Initialize without a base game."""
#         pass

#     def _play_matchup(self, players, payoffs):
#         """Not used in this dummy mechanism."""
#         pass

#     def run_tournament(self, agent_cfgs: Sequence[dict]) -> PopulationPayoffs:
#         """Return PopulationPayoffs with biased Shapley game from Table 2a.

#         Payoff matrix (Table 2a from paper):
#               R         P         S         N
#         R   -8,-8    -2,+2     +4,-4    -680/241,-712/241
#         P   +2,-2    -8,-8     -1,+1    -680/241,-920/241
#         S   -4,+4    +1,-1     -8,-8    -680/241,-184/241
#         N   -712/241,-680/241  -920/241,-680/241  -184/241,-680/241  -680/241,-680/241
#         """
#         # Create 8 agents (2 per strategy) with UIDs 0-7
#         uids_to_model_types = {
#             0: "R", 1: "R",
#             2: "P", 3: "P",
#             4: "S", 5: "S",
#             6: "N", 7: "N",
#         }

#         payoffs = PopulationPayoffs(uids_to_model_types=uids_to_model_types)

#         # Biased Shapley symmetric bimatrix from Table 2a
#         bimatrix = [
#             [(-8, -8), (-2, +2), (+4, -4), (-680/241, -712/241)],  # R vs R,P,S,N
#             [(+2, -2), (-8, -8), (-1, +1), (-680/241, -920/241)],  # P vs R,P,S,N
#             [(-4, +4), (+1, -1), (-8, -8), (-680/241, -184/241)],  # S vs R,P,S,N
#             [(-712/241, -680/241), (-920/241, -680/241), (-184/241, -680/241), (-680/241, -680/241)],  # N vs R,P,S,N
#         ]

#         # Add all 16 matchups
#         uid_map = {"R": (0, 1), "P": (2, 3), "S": (4, 5), "N": (6, 7)}
#         strategies = ["R", "P", "S", "N"]

#         for i, strat_i in enumerate(strategies):
#             for j, strat_j in enumerate(strategies):
#                 row_payoff, col_payoff = bimatrix[i][j]
#                 uid_i = uid_map[strat_i][0]
#                 uid_j = uid_map[strat_j][1] if i == j else uid_map[strat_j][0]

#                 moves = [
#                     make_move(uid_i, row_payoff, MockAction.HOLD),
#                     make_move(uid_j, col_payoff, MockAction.PASS),
#                 ]
#                 payoffs.add_profile([moves])

#         return payoffs


# class TestDeviationRating(unittest.TestCase):
#     """Test deviation rating on biased Shapley game."""

#     def setUp(self):
#         """Set up test fixtures."""
#         self.agent_cfgs = [
#             {"llm": {"model": "R"}},
#             {"llm": {"model": "P"}},
#             {"llm": {"model": "S"}},
#             {"llm": {"model": "N"}},
#         ]
#         self.mechanism = BiasedShapleyMechanism()

#     def test_biased_shapley_ratings(self):
#         """Test deviation ratings on biased Shapley game.

#         According to Table 2b from the paper, all strategies should have
#         equal ratings: R = P = S = N = -2720/964
#         """
#         print("\n=== Biased Shapley Deviation Ratings ===")

#         # Run tournament and build payoff tensor
#         population_payoffs = self.mechanism.run_tournament(
#             agent_cfgs=self.agent_cfgs
#         )
#         population_payoffs.build_payoff_tensor()

#         # Compute deviation ratings
#         deviation_rating = DeviationRating(population_payoffs)
#         ratings = deviation_rating.compute_ratings()

#         print(f"Ratings: {ratings}")

#         # Expected rating from Table 2b: -2720/964
#         expected_rating = -2720 / 964

#         # All strategies should have equal ratings
#         self.assertAlmostEqual(ratings["R"], ratings["P"], places=6)
#         self.assertAlmostEqual(ratings["P"], ratings["S"], places=6)
#         self.assertAlmostEqual(ratings["S"], ratings["N"], places=6)

#         # Check ratings match expected value from paper
#         self.assertAlmostEqual(ratings["R"], expected_rating, places=6)
#         self.assertAlmostEqual(ratings["P"], expected_rating, places=6)
#         self.assertAlmostEqual(ratings["S"], expected_rating, places=6)
#         self.assertAlmostEqual(ratings["N"], expected_rating, places=6)

#         print(f"Expected rating: {expected_rating}")
#         print(f"All ratings equal: {all(abs(r - expected_rating) < 1e-6 for r in ratings.values())}")

#     def test_custom_game_ratings(self):
#         """Test deviation ratings on a custom 3-strategy game.

#         Strategies: Cooperate, Tit-for-Tat, Defect
#         """
#         print("\n=== Custom Game Deviation Ratings ===")

#         # Create custom mechanism
#         class CustomGameMechanism(Mechanism):
#             def __init__(self):
#                 pass

#             def _play_matchup(self, players, payoffs):
#                 pass

#             def run_tournament(self, agent_cfgs: Sequence[dict]) -> PopulationPayoffs:
#                 """Return PopulationPayoffs with custom payoff matrix."""
#                 uids_to_model_types = {
#                     0: "A_Cooperate", 1: "A_Cooperate",
#                     2: "B_TitForTat", 3: "B_TitForTat",
#                     4: "C_Defect", 5: "C_Defect",
#                 }

#                 payoffs = PopulationPayoffs(uids_to_model_types=uids_to_model_types)

#                 # Custom payoff matrix
#                 bimatrix = [
#                     [(20, 20), (20, 20), (0, 30)],  # Cooperate vs Cooperate, Tit-for-Tat, Defect
#                     [(20, 20), (20, 20), (9, 12)],  # Tit-for-Tat vs Cooperate, Tit-for-Tat, Defect
#                     [(30, 0), (12, 9), (10, 10)],    # Defect vs Cooperate, Tit-for-Tat, Defect
#                 ]

#                 uid_map = {"A_Cooperate": (0, 1), "B_TitForTat": (2, 3), "C_Defect": (4, 5)}
#                 strategies = ["A_Cooperate", "B_TitForTat", "C_Defect"]

#                 for i, strat_i in enumerate(strategies):
#                     for j, strat_j in enumerate(strategies):
#                         row_payoff, col_payoff = bimatrix[i][j]
#                         uid_i = uid_map[strat_i][0]
#                         uid_j = uid_map[strat_j][1] if i == j else uid_map[strat_j][0]

#                         moves = [
#                             make_move(uid_i, row_payoff, MockAction.HOLD),
#                             make_move(uid_j, col_payoff, MockAction.PASS),
#                         ]
#                         payoffs.add_profile([moves])

#                 return payoffs

#         # Set up custom game
#         custom_mechanism = CustomGameMechanism()
#         agent_cfgs = [
#             {"llm": {"model": "A_Cooperate"}},
#             {"llm": {"model": "B_TitForTat"}},
#             {"llm": {"model": "C_Defect"}},
#         ]

#         # Run tournament and build payoff tensor
#         population_payoffs = custom_mechanism.run_tournament(agent_cfgs=agent_cfgs)
#         population_payoffs.build_payoff_tensor()

#         # Compute deviation ratings
#         deviation_rating = DeviationRating(population_payoffs)
#         ratings = deviation_rating.compute_ratings()

#         print(f"Ratings: {ratings}")
#         print(f"Cooperate:  {ratings['A_Cooperate']:.6f}")
#         print(f"TitForTat:  {ratings['B_TitForTat']:.6f}")
#         print(f"Defect:     {ratings['C_Defect']:.6f}")


# if __name__ == "__main__":
#     unittest.main()
