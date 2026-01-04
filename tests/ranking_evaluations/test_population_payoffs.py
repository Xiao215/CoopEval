import unittest

from src.ranking_evaluations.population_payoffs import PopulationPayoffs
from tests.fakes.general_fakes import FakeAgent, FakeAction, make_fake_move


class TestPopulationPayoffs(unittest.TestCase):
    def setUp(self) -> None:
        self.discount = 0.5

        # Define Agents (The source of truth)
        self.agent_1 = FakeAgent(uid=1, model_type="model_a")
        self.agent_2 = FakeAgent(uid=2, model_type="model_b")
        self.players = [self.agent_1, self.agent_2]

        # Profile 1: 3 rounds
        # Discount weights (0.5): [0.5, 0.25, 0.25]
        # Agent 1 points: [3.0, 1.0, 5.0] -> 1.5 + 0.25 + 1.25 = 3.0
        # Agent 2 points: [1.0, 3.0, 5.0] -> 0.5 + 0.75 + 1.25 = 2.5
        self.first_profile = [
            [
                make_fake_move(1, 3.0, FakeAction.HOLD),
                make_fake_move(2, 1.0, FakeAction.PASS),
            ],
            [
                make_fake_move(1, 1.0, FakeAction.PASS),
                make_fake_move(2, 3.0, FakeAction.HOLD),
            ],
            [
                make_fake_move(1, 5.0, FakeAction.HOLD),
                make_fake_move(2, 5.0, FakeAction.PASS),
            ],
        ]

        # Profile 2: 3 rounds
        # Discount weights (0.5): [0.5, 0.25, 0.25]
        # Agent 1 points: [6.0, 2.0, 0.0] -> 3.0 + 0.5 + 0.0 = 3.5
        # Agent 2 points: [0.0, 2.0, 6.0] -> 0.0 + 0.5 + 1.5 = 2.0
        self.second_profile = [
            [
                make_fake_move(1, 6.0, FakeAction.PASS),
                make_fake_move(2, 0.0, FakeAction.HOLD),
            ],
            [
                make_fake_move(1, 2.0, FakeAction.HOLD),
                make_fake_move(2, 2.0, FakeAction.PASS),
            ],
            [
                make_fake_move(1, 0.0, FakeAction.PASS),
                make_fake_move(2, 6.0, FakeAction.HOLD),
            ],
        ]

    def test_init_requires_players(self) -> None:
        """Test that initializing without players raises ValueError."""
        with self.assertRaises(ValueError):
            PopulationPayoffs(players=[])

    def test_model_average_payoff_with_discount(self) -> None:
        # Arrange
        payoffs = PopulationPayoffs(
            players=self.players, discount=self.discount
        )

        # Act
        payoffs.add_profile(self.first_profile)
        averages = payoffs.model_average_payoff()

        # Assert
        #
        self.assertAlmostEqual(averages["model_a"], 3.0)
        self.assertAlmostEqual(averages["model_b"], 2.5)

    def test_model_average_payoff_multiple_profiles(self) -> None:
        # Arrange
        payoffs = PopulationPayoffs(
            players=self.players, discount=self.discount
        )

        # Act
        payoffs.add_profile(self.first_profile)
        payoffs.add_profile(self.second_profile)
        averages = payoffs.model_average_payoff()

        # Assert
        # Model A: Avg(3.0, 3.5) = 3.25
        # Model B: Avg(2.5, 2.0) = 2.25
        self.assertAlmostEqual(averages["model_a"], 3.25)
        self.assertAlmostEqual(averages["model_b"], 2.25)

    def test_fitness_computes_expected_payoff_against_population(self) -> None:
        # Arrange
        payoffs = PopulationPayoffs(
            players=self.players, discount=self.discount
        )
        payoffs.add_profile(self.first_profile)
        payoffs.build_payoff_tensor()

        # Act
        # A: 3.0
        # B: 2.5
        fitness = payoffs.fitness({"model_a": 0.6, "model_b": 0.4})

        # Assert
        # fitness[i] = expected payoff for model i against the population
        # model_a gets 3.0 vs model_b, 0.0 vs model_a (not in data)
        # fitness[model_a] = 0.0 × 0.6 + 3.0 × 0.4 = 1.2
        # model_b gets 2.5 vs model_a, 0.0 vs model_b (not in data)
        # fitness[model_b] = 2.5 × 0.6 + 0.0 × 0.4 = 1.5
        self.assertAlmostEqual(fitness["model_a"], 1.2)
        self.assertAlmostEqual(fitness["model_b"], 1.5)

    def test_fitness_requires_probabilities_sum_to_one(self) -> None:
        payoffs = PopulationPayoffs(
            players=self.players, discount=self.discount
        )
        payoffs.add_profile(self.first_profile)
        payoffs.build_payoff_tensor()

        with self.assertRaises(ValueError):
            payoffs.fitness({"model_a": 0.6, "model_b": 0.5})

    def test_to_json_and_from_json_round_trip(self) -> None:
        # Arrange
        payoffs = PopulationPayoffs(
            players=self.players, discount=self.discount
        )
        payoffs.add_profile(self.first_profile)
        payoffs.add_profile(self.second_profile)

        expected_average = payoffs.model_average_payoff()

        # Act: Serialize
        payload = payoffs.to_json()

        # Assert JSON Structure
        self.assertEqual(payload["discount"], self.discount)
        # Check that debug map exists and is correct
        self.assertEqual(
            payload["debug_uids_map"], {"1": "model_a", "2": "model_b"}
        )

        # Check matchups
        self.assertEqual(
            len(payload["matchups"]), 1
        )  # Only 1 unique profile key (uids 1,2)
        stored_matchup = payload["matchups"][0]
        self.assertCountEqual(stored_matchup["uids"], [1, 2])
        # Should have 2 matches recorded for this profile
        self.assertEqual(len(stored_matchup["payoffs"]), 2)

        # Act: Reconstruct
        # NOTE: We must provide players to from_json now!
        reconstructed = PopulationPayoffs.from_json(
            payload, players=self.players
        )
        reconstructed_average = reconstructed.model_average_payoff()

        # Assert Logic Preservation
        self.assertAlmostEqual(reconstructed.discount, self.discount)
        self.assertAlmostEqual(
            reconstructed_average["model_a"], expected_average["model_a"]
        )
        self.assertAlmostEqual(
            reconstructed_average["model_b"], expected_average["model_b"]
        )

    def test_add_profile_rejects_empty_rounds(self) -> None:
        payoffs = PopulationPayoffs(
            players=self.players, discount=self.discount
        )

        with self.assertRaises(ValueError):
            payoffs.add_profile([])

if __name__ == "__main__":
    unittest.main()
