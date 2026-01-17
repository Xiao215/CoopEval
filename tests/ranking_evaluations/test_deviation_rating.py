"""Tests for deviation rating."""

import unittest
from typing import Sequence

from src.ranking_evaluations.deviation_rating import DeviationRating
from src.ranking_evaluations.matchup_payoffs import MatchupPayoffs
from src.mechanisms.base import Mechanism
from src.games.base import Move
from tests.fakes.general_fakes import FakeAction, FakeAgent


class BiasedShapleyMechanism(Mechanism):
    """Mechanism with biased Shapley payoffs from Table 2 of the paper.

    This is a symmetric 2-player game with 4 strategies (R, P, S, N):
    - R, P, S form a biased rock-paper-scissors cycle
    - N is the Nash equilibrium mixture
    """

    def __init__(self):
        """Initialize without a base game."""
        pass

    def _play_matchup(self, players, payoffs):
        """Not used in this dummy mechanism."""
        pass

    def run_tournament(self, agent_cfgs: Sequence[dict]) -> MatchupPayoffs:
        """Return MatchupPayoffs with biased Shapley game from Table 2a.

        Payoff matrix (Table 2a from paper):
              R         P         S         N
        R   -8,-8    -2,+2     +4,-4    -680/241,-712/241
        P   +2,-2    -8,-8     -1,+1    -680/241,-920/241
        S   -4,+4    +1,-1     -8,-8    -680/241,-184/241
        N   -712/241,-680/241  -920/241,-680/241  -184/241,-680/241  -680/241,-680/241
        """
        # Create agents (2 per strategy)
        agent_map = {}  # maps strategy to list of agents
        player_id = 0

        for strat in ["R", "P", "S", "N"]:
            agent_map[strat] = []
            for _ in range(2):
                agent = FakeAgent(llm_config={"model": strat}, player_id=player_id)
                agent_map[strat].append(agent)
                player_id += 1

        payoffs = MatchupPayoffs()

        # Biased Shapley symmetric bimatrix from Table 2a
        bimatrix = [
            [(-8, -8), (-2, +2), (+4, -4), (-680/241, -712/241)],  # R vs R,P,S,N
            [(+2, -2), (-8, -8), (-1, +1), (-680/241, -920/241)],  # P vs R,P,S,N
            [(-4, +4), (+1, -1), (-8, -8), (-680/241, -184/241)],  # S vs R,P,S,N
            [(-712/241, -680/241), (-920/241, -680/241), (-184/241, -680/241), (-680/241, -680/241)],  # N vs R,P,S,N
        ]

        # Add all 16 matchups
        strategies = ["R", "P", "S", "N"]

        for i, strat_i in enumerate(strategies):
            for j, strat_j in enumerate(strategies):
                row_payoff, col_payoff = bimatrix[i][j]
                # Use different agents for same vs same matchups
                agent_i = agent_map[strat_i][0]
                agent_j = agent_map[strat_j][1] if i == j else agent_map[strat_j][0]

                moves = [
                    Move(
                        player=agent_i,
                        action=FakeAction.HOLD,
                        points=row_payoff,
                        response="",
                        trace_id="",
                    ),
                    Move(
                        player=agent_j,
                        action=FakeAction.PASS,
                        points=col_payoff,
                        response="",
                        trace_id="",
                    ),
                ]
                payoffs.add_profile([moves])

        return payoffs


class TestDeviationRating(unittest.TestCase):
    """Test deviation rating on biased Shapley game."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent_cfgs = [
            {"llm": {"model": "R"}},
            {"llm": {"model": "P"}},
            {"llm": {"model": "S"}},
            {"llm": {"model": "N"}},
        ]
        self.mechanism = BiasedShapleyMechanism()

    def test_biased_shapley_ratings(self):
        """Test deviation ratings on biased Shapley game.

        According to Table 2b from the paper, all strategies should have
        equal ratings: R = P = S = N = -2720/964
        """
        print("\n=== Biased Shapley Deviation Ratings ===")

        # Run tournament and build payoff tensor
        population_payoffs = self.mechanism.run_tournament(
            agent_cfgs=self.agent_cfgs
        )
        population_payoffs.build_payoff_tensor()

        # Compute deviation ratings
        deviation_rating = DeviationRating(population_payoffs)
        ratings = deviation_rating.compute_ratings()

        print(f"Ratings: {ratings}")

        # Expected rating from Table 2b: -2720/964
        expected_rating = -2720 / 964

        # All strategies should have equal ratings
        self.assertAlmostEqual(ratings["R"], ratings["P"], places=6)
        self.assertAlmostEqual(ratings["P"], ratings["S"], places=6)
        self.assertAlmostEqual(ratings["S"], ratings["N"], places=6)

        # Check ratings match expected value from paper
        self.assertAlmostEqual(ratings["R"], expected_rating, places=6)
        self.assertAlmostEqual(ratings["P"], expected_rating, places=6)
        self.assertAlmostEqual(ratings["S"], expected_rating, places=6)
        self.assertAlmostEqual(ratings["N"], expected_rating, places=6)

        print(f"Expected rating: {expected_rating}")
        print(f"All ratings equal: {all(abs(r - expected_rating) < 1e-6 for r in ratings.values())}")

    def test_custom_game_ratings(self):
        """Test deviation ratings on a custom 3-strategy game.

        Strategies: Cooperate, Tit-for-Tat, Defect
        """
        print("\n=== Custom Game Deviation Ratings ===")

        # Create custom mechanism
        class CustomGameMechanism(Mechanism):
            def __init__(self):
                pass

            def _play_matchup(self, players, payoffs):
                pass

            def run_tournament(self, agent_cfgs: Sequence[dict]) -> MatchupPayoffs:
                """Return MatchupPayoffs with custom payoff matrix."""
                # Create agents (2 per strategy)
                agent_map = {}
                player_id = 0
                for strat in ["A_Cooperate", "B_TitForTat", "C_Defect"]:
                    agent_map[strat] = []
                    for _ in range(2):
                        agent = FakeAgent(llm_config={"model": strat}, player_id=player_id)
                        agent_map[strat].append(agent)
                        player_id += 1

                payoffs = MatchupPayoffs()

                # Custom payoff matrix
                bimatrix = [
                    [(20, 20), (20, 20), (0, 30)],  # Cooperate vs Cooperate, Tit-for-Tat, Defect
                    [(20, 20), (20, 20), (9, 12)],  # Tit-for-Tat vs Cooperate, Tit-for-Tat, Defect
                    [(30, 0), (12, 9), (10, 10)],    # Defect vs Cooperate, Tit-for-Tat, Defect
                ]

                strategies = ["A_Cooperate", "B_TitForTat", "C_Defect"]

                for i, strat_i in enumerate(strategies):
                    for j, strat_j in enumerate(strategies):
                        row_payoff, col_payoff = bimatrix[i][j]
                        # Use different agents for same vs same matchups
                        agent_i = agent_map[strat_i][0]
                        agent_j = agent_map[strat_j][1] if i == j else agent_map[strat_j][0]

                        moves = [
                            Move(
                                player=agent_i,
                                action=FakeAction.HOLD,
                                points=row_payoff,
                                response="",
                                trace_id="",
                            ),
                            Move(
                                player=agent_j,
                                action=FakeAction.PASS,
                                points=col_payoff,
                                response="",
                                trace_id="",
                            ),
                        ]
                        payoffs.add_profile([moves])

                return payoffs

        # Set up custom game
        custom_mechanism = CustomGameMechanism()
        agent_cfgs = [
            {"llm": {"model": "A_Cooperate"}},
            {"llm": {"model": "B_TitForTat"}},
            {"llm": {"model": "C_Defect"}},
        ]

        # Run tournament and build payoff tensor
        population_payoffs = custom_mechanism.run_tournament(agent_cfgs=agent_cfgs)
        population_payoffs.build_payoff_tensor()

        # Compute deviation ratings
        deviation_rating = DeviationRating(population_payoffs)
        ratings = deviation_rating.compute_ratings()

        print(f"Ratings: {ratings}")
        print(f"Cooperate:  {ratings['A_Cooperate']:.6f}")
        print(f"TitForTat:  {ratings['B_TitForTat']:.6f}")
        print(f"Defect:     {ratings['C_Defect']:.6f}")

    def test_3x3_symmetric_game(self):
        """Test deviation ratings on a custom 3x3 symmetric game.

        Payoff tensor for Player 1:
        [[1, 1, 1],
         [2, 2, 2],
         [0, 3, 3]]

        Player 2's payoffs are symmetric (transpose of Player 1's payoffs).
        """
        print("\n=== 3x3 Symmetric Game Deviation Ratings ===")

        # Create custom mechanism
        class SymmetricGame3x3Mechanism(Mechanism):
            def __init__(self):
                pass

            def _play_matchup(self, players, payoffs):
                pass

            def run_tournament(self, agent_cfgs: Sequence[dict]) -> MatchupPayoffs:
                """Return MatchupPayoffs with 3x3 symmetric payoff matrix."""
                # Create agents (2 per strategy)
                agent_map = {}
                player_id = 0
                for strat in ["A", "B", "C"]:
                    agent_map[strat] = []
                    for _ in range(2):
                        agent = FakeAgent(llm_config={"model": strat}, player_id=player_id)
                        agent_map[strat].append(agent)
                        player_id += 1

                payoffs = MatchupPayoffs()

                # Payoff matrix - symmetric game
                # Player 1's payoff matrix: [[1,1,1], [2,2,2], [0,3,3]]
                # For symmetric game, bimatrix[i][j] = (player1_payoff[i][j], player1_payoff[j][i])
                bimatrix = [
                    [(1, 1), (1, 2), (1, 0)],  # A vs A, B, C
                    [(2, 1), (2, 2), (2, 3)],  # B vs A, B, C
                    [(0, 1), (3, 2), (3, 3)],  # C vs A, B, C
                ]

                strategies = ["A", "B", "C"]

                for i, strat_i in enumerate(strategies):
                    for j, strat_j in enumerate(strategies):
                        row_payoff, col_payoff = bimatrix[i][j]
                        # Use different agents for same vs same matchups
                        agent_i = agent_map[strat_i][0]
                        agent_j = agent_map[strat_j][1] if i == j else agent_map[strat_j][0]

                        moves = [
                            Move(
                                player=agent_i,
                                action=FakeAction.HOLD,
                                points=row_payoff,
                                response="",
                                trace_id="",
                            ),
                            Move(
                                player=agent_j,
                                action=FakeAction.PASS,
                                points=col_payoff,
                                response="",
                                trace_id="",
                            ),
                        ]
                        payoffs.add_profile([moves])

                return payoffs

        # Set up game
        mechanism = SymmetricGame3x3Mechanism()
        agent_cfgs = [
            {"llm": {"model": "A"}},
            {"llm": {"model": "B"}},
            {"llm": {"model": "C"}},
        ]

        # Run tournament and build payoff tensor
        population_payoffs = mechanism.run_tournament(agent_cfgs=agent_cfgs)
        population_payoffs.build_payoff_tensor()

        # Compute deviation ratings
        deviation_rating = DeviationRating(population_payoffs)
        ratings = deviation_rating.compute_ratings()

        print(f"Ratings: {ratings}")
        print(f"Strategy A: {ratings['A']:.6f}")
        print(f"Strategy B: {ratings['B']:.6f}")
        print(f"Strategy C: {ratings['C']:.6f}")


if __name__ == "__main__":
    unittest.main()
