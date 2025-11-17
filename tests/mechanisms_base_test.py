import unittest


from src.mechanisms.base import RepetitiveMechanism
from tests.mocks.general_mocks import MockAction, make_move


class TestRepetitiveMechanismHistory(unittest.TestCase):
    def setUp(self) -> None:
        self.history = RepetitiveMechanism.History()
        self.rounds = [
            [
                make_move(1, 1.0, MockAction.HOLD),
                make_move(2, 2.0, MockAction.PASS),
                make_move(3, 3.0, MockAction.HOLD),
            ],
            [
                make_move(1, 1.5, MockAction.HOLD),
                make_move(3, 2.5, MockAction.PASS),
                make_move(5, 3.5, MockAction.HOLD),
            ],
            [
                make_move(2, 2.0, MockAction.HOLD),
                make_move(3, 3.0, MockAction.PASS),
                make_move(4, 4.0, MockAction.HOLD),
            ],
            [
                make_move(1, 6.0, MockAction.PASS),
                make_move(2, 7.0, MockAction.HOLD),
                make_move(5, 5.0, MockAction.HOLD),
            ],
        ]
        for r in self.rounds:
            self.history.append(r)

    def testAppend_tracksIndicesAndCumulativeCounts(self) -> None:
        self.history.append(
            [
                make_move(1, 10.0, MockAction.HOLD),
                make_move(7, 20.0, MockAction.PASS),
                make_move(8, 30.0, MockAction.HOLD),
            ]
        )
        self.assertEqual(len(self.history.records), 5)
        self.assertEqual(self.history.player_round_indices["agent-1"], [0, 1, 3, 4, 5])
        self.assertEqual(self.history.player_round_indices["agent-2"], [0, 2, 3])

        self.assertEqual(
            self.history.player_cumulative_actions["agent-1"],
            [
                {MockAction.HOLD: 1},
                {MockAction.HOLD: 2},
                {MockAction.HOLD: 2, MockAction.PASS: 1},
                {MockAction.HOLD: 3, MockAction.PASS: 1},
            ],
        )
        self.assertEqual(
            self.history.player_cumulative_actions["agent-2"],
            [
                {MockAction.PASS: 1},
                {MockAction.PASS: 1, MockAction.HOLD: 1},
                {MockAction.PASS: 1, MockAction.HOLD: 2},
            ],
        )

    def testAppend_rejectEmptyRound(self) -> None:
        history = RepetitiveMechanism.History()
        with self.assertRaises(ValueError):
            history.append([])

    def testGetPriorRounds_respectsLookbackAndDepth(self) -> None:
        prior = self.history.get_prior_rounds("agent-1", lookback_rounds=1, lookup_depth=2)
        self.assertEqual(prior, self.rounds[:2])

        self.assertEqual(self.history.get_prior_rounds("agent-1", lookback_rounds=3, lookup_depth=1), [])
        self.assertEqual(
            self.history.get_prior_rounds("unknown", lookback_rounds=0, lookup_depth=1), []
        )
        with self.assertRaises(ValueError):
            self.history.get_prior_rounds("agent-1", lookback_rounds=-1, lookup_depth=1)
        with self.assertRaises(ValueError):
            self.history.get_prior_rounds("agent-1", lookback_rounds=0, lookup_depth=0)

    def testGetPriorActionDistribution(self) -> None:
        dist = self.history.get_prior_action_distribution("agent-1", lookback_rounds=1)
        self.assertEqual(dist, {MockAction.HOLD: 2})

        dist_all = self.history.get_prior_action_distribution("agent-2", lookback_rounds=0)
        self.assertEqual(dist_all, {MockAction.PASS: 3})

        self.assertEqual(
            self.history.get_prior_action_distribution("agent-1", lookback_rounds=5), {}
        )
        self.assertEqual(
            self.history.get_prior_action_distribution("unknown", lookback_rounds=0), {}
        )
        with self.assertRaises(ValueError):
            self.history.get_prior_action_distribution("agent-1", lookback_rounds=-1)


if __name__ == "__main__":
    unittest.main()
