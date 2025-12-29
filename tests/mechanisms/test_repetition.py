import unittest
from unittest.mock import MagicMock, patch

from src.games.base import Move
from src.evolution.population_payoffs import PopulationPayoffs
from src.mechanisms.prompts import (
    REPETITION_MECHANISM_PROMPT,
    REPETITION_NO_HISTORY_DESCRIPTION,
)
from src.mechanisms.repetition import Repetition
from tests.fakes.general_fakes import MockAction, make_move, MockAgent

class ScriptedGame:
    """Minimal base game that returns pre-baked moves and records prompts."""

    def __init__(self, num_players: int, rounds: list[list[Move]]):
        self.num_players = num_players
        self.num_actions = 2
        self.prompt = "stub prompt"
        self.rounds = rounds
        self.play_calls: list[list[str]] = []

    def play(self, additional_info, players, action_map=lambda x: x):
        self.play_calls.append(additional_info)
        return self.rounds[len(self.play_calls) - 1]


class TestRepetitionMechanism(unittest.TestCase):
    def setUp(self) -> None:
        self.players = [MockAgent(1), MockAgent(2)]
        self.base_game = ScriptedGame(
            num_players=len(self.players),
            rounds=[
                [
                    make_move(1, 1.0, MockAction.HOLD),
                    make_move(2, 2.0, MockAction.PASS),
                ],
                [
                    make_move(1, 1.5, MockAction.PASS),
                    make_move(2, 2.5, MockAction.HOLD),
                ],
            ],
        )
        self.mechanism = Repetition(self.base_game, num_rounds=2, discount=0.9)

    def test_build_history_prompts_without_history(self) -> None:
        prompts = self.mechanism._build_history_prompts(self.players, round_idx=1)

        self.assertEqual(len(prompts), len(self.players))
        expected = REPETITION_MECHANISM_PROMPT.format(
            round_idx=1,
            history_context=REPETITION_NO_HISTORY_DESCRIPTION,
        ).strip()
        self.assertListEqual(prompts, [expected] * len(self.players))

    def test_build_history_prompts_with_full_history_per_player(self) -> None:
        self.mechanism.history.append(self.base_game.rounds[0])

        prompts = self.mechanism._build_history_prompts(self.players, round_idx=2)

        self.assertEqual(len(prompts), len(self.players))
        self.assertNotEqual(prompts[0], prompts[1])
        self.assertIn("Round 1", prompts[0])
        self.assertIn("You: MockAction.HOLD", prompts[0])
        self.assertIn("Player#2: MockAction.PASS", prompts[0])
        self.assertIn("You: MockAction.PASS", prompts[1])
        self.assertIn("Player#1: MockAction.HOLD", prompts[1])

    def test_format_recent_history_rejects_non_positive_depth(self) -> None:
        with self.assertRaises(ValueError):
            self.mechanism._format_recent_history(self.players, self.players[0], lookup_depth=0)

    def test_serialize_records_converts_moves_to_dicts(self) -> None:
        records = [
            [
                make_move(1, 3.0, MockAction.HOLD),
                make_move(2, 4.0, MockAction.PASS),
            ]
        ]

        serialized = Repetition._serialize_records(records)

        self.assertEqual(
            serialized,
            [
                [
                    {
                        "uid": 1,
                        "player_name": "agent-1",
                        "action": "MockAction.HOLD",
                        "points": 3.0,
                        "response": "",
                    },
                    {
                        "uid": 2,
                        "player_name": "agent-2",
                        "action": "MockAction.PASS",
                        "points": 4.0,
                        "response": "",
                    },
                ]
            ],
        )

    def test_play_matchup_records_rounds_and_reports_results(self) -> None:
        payoffs = MagicMock(spec=PopulationPayoffs)

        with patch("src.mechanisms.repetition.LOGGER.log_record") as mock_log_record:
            self.mechanism._play_matchup(self.players, payoffs)

        self.assertEqual(len(self.base_game.play_calls), self.mechanism.num_rounds)
        first_call_info, second_call_info = self.base_game.play_calls
        self.assertEqual(len(first_call_info), len(self.players))
        self.assertTrue(
            all(REPETITION_NO_HISTORY_DESCRIPTION in prompt for prompt in first_call_info)
        )
        self.assertTrue(all("Round 1" in prompt for prompt in second_call_info))

        expected_records = [list(r) for r in self.base_game.rounds]
        payoffs.add_profile.assert_called_once_with(expected_records)
        mock_log_record.assert_called_once_with(
            record=Repetition._serialize_records(expected_records),
            file_name=self.mechanism.record_file,
        )


if __name__ == "__main__":
    unittest.main()
