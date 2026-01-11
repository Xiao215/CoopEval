"""Test LLM instance for fast local testing without API calls."""

import json
import random
import re
from typing import Any

from src.agents.base import LLM


class TestInstance(LLM):
    """A fake LLM that returns canned responses for testing mechanisms without API calls."""

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        if self._matches(prompt, self._action_template_patterns()):
            # Parse the action template to determine number of actions
            num_actions = self._count_actions_in_template(prompt)

            weights = [random.random() for _ in range(num_actions)]
            total = sum(weights) or 1.0
            raw = [weight / total * 100 for weight in weights]
            ints = [int(value) for value in raw]
            remainder = 100 - sum(ints)
            if remainder:
                fractional = sorted(
                    enumerate(raw),
                    key=lambda item: item[1] - int(item[1]),
                    reverse=True,
                )
                for i in range(remainder):
                    ints[fractional[i % num_actions][0]] += 1

            distribution = {f"A{i}": value for i, value in enumerate(ints)}
            return json.dumps(distribution)
        return '{"A0": 100, "A1": 0}'  # always return action A0 with 100 points

    def _count_actions_in_template(self, prompt: str) -> int:
        """Count the number of action keys in the template by finding all A0, A1, A2, etc."""
        # First, try to find action definitions in the "Actions available" section
        # This works for all games that list actions like "- A0:", "- A1:", etc.
        action_keys = re.findall(r'^\s*-\s*A(\d+)', prompt, re.MULTILINE)
        assert action_keys, "No action keys found in the prompt."
        # Return the maximum action index + 1 (since A0 is the first action)
        return max(int(key) for key in action_keys) + 1

    def _action_template_patterns(self) -> tuple[str, ...]:
        return (
            r'\{\s*"A0"\s*:\s*(?:-?\d+|<[Ii][Nn][Tt]>)\s*,\s*"A1"\s*:\s*(?:-?\d+|<[Ii][Nn][Tt]>)(?:\s*,\s*(?:"A\d+"\s*:\s*(?:-?\d+|<[Ii][Nn][Tt]>)|\.\.\.))*\s*\}',
        )

    def _matches(self, prompt: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, prompt) for pattern in patterns)
