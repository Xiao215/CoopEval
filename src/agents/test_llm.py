"""Local Hugging Face model runner used by :class:`Agent` implementations."""

from typing import Any

from src.agents.base import LLM


class TestInstance(LLM):
    """A class to manage a Hugging Face LLM pipeline that can be moved between CPU and GPU."""

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        # TODO make the test LLM capable with disarmament, mediation, contracting
        # Different cases of answering:
        return '{"A0": 100, "A1": 0}'  # always return action A0 with 100 points
