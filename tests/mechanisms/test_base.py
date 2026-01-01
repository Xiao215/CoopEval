import unittest


from src.mechanisms.base import RepetitiveMechanism
from tests.fakes.general_fakes import FakeAction, make_fake_move


class TestRepetitiveMechanismHistory(unittest.TestCase):
    def setUp(self) -> None:
        self.history = RepetitiveMechanism.History()
        self.rounds = [
            [
                make_fake_move(1, 1.0, FakeAction.HOLD),
                make_fake_move(2, 2.0, FakeAction.PASS),
                make_fake_move(3, 3.0, FakeAction.HOLD),
            ],
            [
                make_fake_move(1, 1.5, FakeAction.HOLD),
                make_fake_move(3, 2.5, FakeAction.PASS),
                make_fake_move(5, 3.5, FakeAction.HOLD),
            ],
            [
                make_fake_move(2, 2.0, FakeAction.HOLD),
                make_fake_move(3, 3.0, FakeAction.PASS),
                make_fake_move(4, 4.0, FakeAction.HOLD),
            ],
            [
                make_fake_move(1, 6.0, FakeAction.PASS),
                make_fake_move(2, 7.0, FakeAction.HOLD),
                make_fake_move(5, 5.0, FakeAction.HOLD),
            ],
        ]
        for r in self.rounds:
            self.history.append(r)

    def test_append_tracks_indices_and_cumulative_counts(self) -> None:
        self.history.append(
            [
                make_fake_move(1, 10.0, FakeAction.HOLD),
                make_fake_move(7, 20.0, FakeAction.PASS),
                make_fake_move(8, 30.0, FakeAction.HOLD),
            ]
        )
        self.assertEqual(len(self.history.records), 5)
        self.assertEqual(self.history.player_round_indices["fake-agent-1"], [0, 1, 3, 4])
        self.assertEqual(self.history.player_round_indices["fake-agent-2"], [0, 2, 3])

        self.assertEqual(
            self.history.player_cumulative_actions["fake-agent-1"],
            [
                {FakeAction.HOLD: 1},
                {FakeAction.HOLD: 2},
                {FakeAction.HOLD: 2, FakeAction.PASS: 1},
                {FakeAction.HOLD: 3, FakeAction.PASS: 1},
            ],
        )
        self.assertEqual(
            self.history.player_cumulative_actions["fake-agent-2"],
            [
                {FakeAction.PASS: 1},
                {FakeAction.PASS: 1, FakeAction.HOLD: 1},
                {FakeAction.PASS: 1, FakeAction.HOLD: 2},
            ],
        )

    def test_append_reject_empty_round(self) -> None:
        history = RepetitiveMechanism.History()
        with self.assertRaises(ValueError):
            history.append([])

    def test_get_prior_rounds_respects_lookback_and_depth(self) -> None:
        prior = self.history.get_prior_rounds("fake-agent-1", lookback_rounds=1, lookup_depth=2)
        self.assertEqual(prior, self.rounds[:2])

        self.assertEqual(self.history.get_prior_rounds("fake-agent-1", lookback_rounds=3, lookup_depth=1), [])
        self.assertEqual(
            self.history.get_prior_rounds("unknown", lookback_rounds=0, lookup_depth=1), []
        )
        with self.assertRaises(ValueError):
            self.history.get_prior_rounds("fake-agent-1", lookback_rounds=-1, lookup_depth=1)
        with self.assertRaises(ValueError):
            self.history.get_prior_rounds("fake-agent-1", lookback_rounds=0, lookup_depth=0)

    def test_get_prior_action_distribution(self) -> None:
        dist = self.history.get_prior_action_distribution("fake-agent-1", lookback_rounds=1)
        self.assertEqual(dist, {FakeAction.HOLD: 2})

        dist_all = self.history.get_prior_action_distribution("fake-agent-2", lookback_rounds=0)
        self.assertEqual(dist_all, {FakeAction.PASS: 1, FakeAction.HOLD: 2})

        self.assertEqual(
            self.history.get_prior_action_distribution("fake-agent-1", lookback_rounds=5), None
        )
        self.assertEqual(
            self.history.get_prior_action_distribution("unknown", lookback_rounds=0), None
        )
        with self.assertRaises(ValueError):
            self.history.get_prior_action_distribution("fake-agent-1", lookback_rounds=-1)


if __name__ == "__main__":
    unittest.main()
