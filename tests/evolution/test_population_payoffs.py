import unittest

import numpy as np
from src.evolution.population_payoffs import PopulationPayoffs
from tests.fakes.general_fakes import MockAgent, MockAction, make_move


class TestPopulationPayoffs(unittest.TestCase):
    def setUp(self) -> None:
        self.discount = 0.5
        self.mapping = {1: "model_a", 2: "model_b"}
        self.first_profile = [
            [
                make_move(1, 3.0, MockAction.HOLD),
                make_move(2, 1.0, MockAction.PASS),
            ],
            [
                make_move(1, 1.0, MockAction.PASS),
                make_move(2, 3.0, MockAction.HOLD),
            ],
            [
                make_move(1, 5.0, MockAction.HOLD),
                make_move(2, 5.0, MockAction.PASS),
            ],
        ]
        self.second_profile = [
            [
                make_move(1, 6.0, MockAction.PASS),
                make_move(2, 0.0, MockAction.HOLD),
            ],
            [
                make_move(1, 2.0, MockAction.HOLD),
                make_move(2, 2.0, MockAction.PASS),
            ],
            [
                make_move(1, 0.0, MockAction.PASS),
                make_move(2, 6.0, MockAction.HOLD),
            ],
        ]

    def test_model_average_payoff_with_discount(self) -> None:
        payoffs = PopulationPayoffs(discount=self.discount, uids_to_model_types=self.mapping)
        payoffs.add_profile(self.first_profile)

        averages = payoffs.model_average_payoff()

        self.assertAlmostEqual(averages["model_a"], 3.0)
        self.assertAlmostEqual(averages["model_b"], 2.5)

    def test_model_average_payoff_multiple_profiles(self) -> None:
        payoffs = PopulationPayoffs(discount=self.discount, uids_to_model_types=self.mapping)
        payoffs.add_profile(self.first_profile)
        payoffs.add_profile(self.second_profile)

        averages = payoffs.model_average_payoff()

        self.assertAlmostEqual(averages["model_a"], 3.25)
        self.assertAlmostEqual(averages["model_b"], 2.25)

    def test_fitness_computes_expected_payoff_against_population(self) -> None:
        payoffs = PopulationPayoffs(discount=self.discount, uids_to_model_types=self.mapping)
        payoffs.add_profile(self.first_profile)
        payoffs.build_payoff_tensor()

        fitness = payoffs.fitness({"model_a": 0.6, "model_b": 0.4})

        # fitness[i] = expected payoff for model i against the population
        # model_a gets 3.0 vs model_b, 0.0 vs model_a (not in data)
        # fitness[model_a] = 0.0 × 0.6 + 3.0 × 0.4 = 1.2
        # model_b gets 2.5 vs model_a, 0.0 vs model_b (not in data)
        # fitness[model_b] = 2.5 × 0.6 + 0.0 × 0.4 = 1.5
        self.assertAlmostEqual(fitness["model_a"], 1.2)
        self.assertAlmostEqual(fitness["model_b"], 1.5)

    def test_fitness_requires_probabilities_sum_to_one(self) -> None:
        payoffs = PopulationPayoffs(discount=self.discount, uids_to_model_types=self.mapping)
        payoffs.add_profile(self.first_profile)
        payoffs.build_payoff_tensor()

        with self.assertRaises(ValueError):
            payoffs.fitness({"model_a": 0.6, "model_b": 0.5})

    def test_init_rejects_misaligned_player_model_types(self) -> None:
        players = [MockAgent(1, "model_a"), MockAgent(2, "model_b")]
        mismatched_mapping = {1: "model_a", 2: "model_c"}

        with self.assertRaises(ValueError):
            PopulationPayoffs(players=players, uids_to_model_types=mismatched_mapping)

    def test_to_json_and_from_json_round_trip(self) -> None:
        payoffs = PopulationPayoffs(discount=self.discount, uids_to_model_types=self.mapping)
        payoffs.add_profile(self.first_profile)
        payoffs.add_profile(self.second_profile)

        expected_average = payoffs.model_average_payoff()
        payload = payoffs.to_json()

        self.assertEqual(payload["discount"], self.discount)
        self.assertEqual(payload["uids_to_model_types"], {"1": "model_a", "2": "model_b"})
        self.assertEqual(len(payload["matchups"]), 1)
        stored_matchup = payload["matchups"][0]
        self.assertCountEqual(stored_matchup["uids"], [1, 2])
        self.assertEqual(np.array(stored_matchup["payoffs"]).shape[0], 2)

        reconstructed = PopulationPayoffs.from_json(payload)
        reconstructed_average = reconstructed.model_average_payoff()

        self.assertAlmostEqual(reconstructed.discount, self.discount)
        self.assertAlmostEqual(reconstructed_average["model_a"], expected_average["model_a"])
        self.assertAlmostEqual(reconstructed_average["model_b"], expected_average["model_b"])

    def test_add_profile_rejects_empty_rounds(self) -> None:
        payoffs = PopulationPayoffs(discount=self.discount, uids_to_model_types=self.mapping)

        with self.assertRaises(ValueError):
            payoffs.add_profile([])


if __name__ == "__main__":
    unittest.main()
