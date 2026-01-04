import unittest
from itertools import combinations

from src.utils.round_robin import RoundRobin
from tests.fakes.general_fakes import FakeAgent

# Assuming your class is in a file named tournament.py
# from tournament import RoundRobin


class TestRoundRobin(unittest.TestCase):

    def setUp(self):
        """Setup common test data."""
        self.even_players = [FakeAgent(i) for i in range(4)]  # 4 Players
        self.odd_players = [FakeAgent(i) for i in range(5)]  # 5 Players

    def _get_all_matches(self, schedule):
        """Helper to flatten the schedule into a list of sorted match tuples."""
        all_matches = []
        for round_matches in schedule:
            for match in round_matches:
                all_matches.append(tuple(sorted(match)))
        return all_matches

    def test_k2_even_players_completeness(self):
        """
        Scenario: 4 Players (Even), K=2.
        Expectation:
        - 3 Rounds total (N-1).
        - 2 matches per round.
        - Everyone plays everyone else exactly once.
        """
        scheduler = RoundRobin(self.even_players, group_size=2)
        schedule = scheduler.generate_schedule(randomize_order=False)

        # 1. Check Round Count
        # Formula for Round Robin rounds (even N) is N-1
        self.assertEqual(len(schedule), 3, "Should have 3 rounds for 4 players")

        # 2. Check Match Count per Round
        # 4 players = 2 matches per round
        for i, round_matches in enumerate(schedule):
            self.assertEqual(
                len(round_matches), 2, f"Round {i} should have 2 matches"
            )

        # 3. Check Uniqueness (Everyone plays everyone)
        flat_matches = self._get_all_matches(schedule)
        unique_matches = set(flat_matches)

        # Expected total matches = N*(N-1)/2 = 4*3/2 = 6
        self.assertEqual(
            len(unique_matches), 6, "Should be exactly 6 unique pairings"
        )

        # Verify specific expected pairs exist
        expected_pairs = set(combinations(self.even_players, 2))
        # Convert combinations to sorted tuples for comparison
        expected_pairs = {tuple(sorted(p)) for p in expected_pairs}

        self.assertEqual(
            unique_matches,
            expected_pairs,
            "Scheduled pairs do not match mathematical combinations",
        )

    def test_k2_odd_players_completeness(self):
        """
        Scenario: 5 Players (Odd), K=2.
        Expectation:
        - 5 Rounds total (N rounds for odd N).
        - 2 matches per round (1 player sits out/BYE).
        - Everyone plays everyone else exactly once.
        """
        scheduler = RoundRobin(self.odd_players, group_size=2)
        schedule = scheduler.generate_schedule(randomize_order=False)

        # 1. Check Round Count
        # For odd N, Circle method uses N+1 (even) pool size. Rounds = (N+1)-1 = N.
        self.assertEqual(len(schedule), 5, "Should have 5 rounds for 5 players")

        # 2. Check Match Count
        # 5 players -> 1 sits out -> 2 matches
        for round_matches in schedule:
            self.assertEqual(len(round_matches), 2)

        # 3. Check Completeness
        flat_matches = self._get_all_matches(schedule)
        unique_matches = set(flat_matches)

        # Expected: 5*4/2 = 10 matches
        self.assertEqual(len(unique_matches), 10)

    def test_no_simultaneous_duplicates(self):
        """
        CRITICAL: Ensure a player never appears twice in the same round.
        """
        scheduler = RoundRobin(self.odd_players, group_size=2)
        schedule = scheduler.generate_schedule()

        for round_idx, round_matches in enumerate(schedule):
            seen_players = []
            for match in round_matches:
                seen_players.extend(match)

            # If set length < list length, someone is duplicated
            self.assertEqual(
                len(seen_players),
                len(set(seen_players)),
                f"Player duplicate found in Round {round_idx}",
            )

    def test_k3_grouping(self):
        """
        Scenario: 5 Players, K=3.
        Expectation: Every unique triplet combination occurs eventually.
        """
        k = 3
        scheduler = RoundRobin(self.odd_players, group_size=k)
        schedule = scheduler.generate_schedule()

        flat_matches = self._get_all_matches(schedule)
        unique_matches = set(flat_matches)

        # Expected combinations: 5 Choose 3 = 10
        expected_count = 10
        self.assertEqual(len(unique_matches), expected_count)

        # Ensure correct match size
        for match in flat_matches:
            self.assertEqual(len(match), k, "Match size must equal K")

    def test_deterministic_vs_random(self):
        """
        Verify that randomize_order=False is reproducible,
        and randomize_order=True changes structure.
        """
        scheduler = RoundRobin(self.even_players, group_size=2)

        # Deterministic check
        sched1 = scheduler.generate_schedule(randomize_order=False)
        sched2 = scheduler.generate_schedule(randomize_order=False)
        self.assertEqual(
            sched1, sched2, "False flag must produce identical schedules"
        )

        # Random check (Small chance of collision, but very unlikely for larger N)
        # Note: For N=4, the rounds are few, but the match order inside rounds should shift
        sched_rand1 = scheduler.generate_schedule(randomize_order=True)
        sched_rand2 = scheduler.generate_schedule(randomize_order=True)

        # We check if the structure is exactly the same (order of rounds/matches)
        # It is highly probable they differ
        self.assertNotEqual(
            sched_rand1, sched_rand2, "Random flag should produce variations"
        )


if __name__ == "__main__":
    unittest.main()
