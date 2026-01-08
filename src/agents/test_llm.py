"""Local Hugging Face model runner used by :class:`Agent` implementations."""

import json
import random
import re
from typing import Any

from src.agents.base import LLM


class TestInstance(LLM):
    """A class to manage a Hugging Face LLM pipeline that can be moved between CPU and GPU."""

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        if self._matches(prompt, self._action_template_patterns()):
            other_players_match = re.search(r"(\d+)\s+other players", prompt)
            num_actions = (
                int(other_players_match.group(1)) if other_players_match else 2
            )
            if num_actions <= 0:
                num_actions = 2

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

    def _action_template_patterns(self) -> tuple[str, ...]:
        return (
            r'\{\s*"A0"\s*:\s*(?:-?\d+|<[Ii][Nn][Tt]>)\s*,\s*"A1"\s*:\s*(?:-?\d+|<[Ii][Nn][Tt]>)(?:\s*,\s*(?:"A\d+"\s*:\s*(?:-?\d+|<[Ii][Nn][Tt]>)|\.\.\.))*\s*\}',
        )

    def _matches(self, prompt: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, prompt) for pattern in patterns)
