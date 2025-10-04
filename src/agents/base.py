"""Abstract interfaces shared by concrete LLM backends."""

from abc import ABC, abstractmethod
from typing import Any


class LLM(ABC):
    """Abstract base class for an LLM pipeline that can be moved between CPU and GPU."""

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Invoke the model with the given prompt."""
        raise NotImplementedError
