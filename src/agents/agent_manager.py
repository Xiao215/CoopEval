"""Agent abstractions and shared LLM caching utilities."""
import itertools
import textwrap
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable

from src.agents.base import LLM
from src.agents.client_api_llm import ClientAPILLM
from src.agents.hf_llm import HFInstance
from src.agents.test_llm import TestInstance
from src.logger_manager import LOGGER


class LLMManager:
    """Central cache for LLM backends, keyed by (model, provider)."""

    def __init__(self) -> None:
        self.llms = dict()

    def get_llm(self, model_name: str, provider: str) -> LLM:
        """Return a cached ``LLM`` implementation for ``model_name``/``provider``."""
        if model_name not in self.llms:
            if provider == "HFInstance":
                self.llms[model_name] = HFInstance(model_name)
            elif provider == "TestInstance":
                self.llms[model_name] = TestInstance()
            elif provider in {"OpenAI", "Gemini", "OpenRouter"}:
                # These providers expose OpenAI-compatible HTTP APIs, so reuse the hosted client wrapper
                self.llms[model_name] = ClientAPILLM(model_name, provider)
            else:
                raise ValueError(
                    f"Unknown provider {provider} for model {model_name}"
                )
        return self.llms[model_name]


class Agent(ABC):
    """
    Abstract base class for an LLM-based agent.
    """

    llm_manager = LLMManager()
    _instance_counter = itertools.count(1)

    def __init__(self, agent_config: dict) -> None:
        llm_config = agent_config["llm"]
        self.model_type = llm_config["model"]
        self.kwargs = llm_config.get("kwargs", {})
        self.pipeline = type(self).llm_manager.get_llm(
            self.model_type, llm_config["provider"]
        )
        self.player_id: int = agent_config["player_id"]
        self.agent_config = agent_config

    @abstractmethod
    def chat(
        self,
        messages: str,
    ) -> tuple[str, str]:
        """Chat with the agent using the provided messages."""
        raise NotImplementedError

    def invoke(self, messages: str) -> tuple[str, str]:
        """Invoke the agent's LLM pipeline with logging. Returns response and trace_id."""
        trace_id = str(uuid.uuid4())[:8]

        response = self.pipeline.invoke(messages, **self.kwargs)

        self._log_inference(messages, response, trace_id)
        return response, trace_id

    def _log_inference(self, prompt: str, response: str, trace_id: str) -> None:
        """Log the inference to the game log."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry = (
            f"===== Prompt [ID: {trace_id}] [{timestamp}] =====\n"
            f"agent: {self.name}\n"
            "prompt:\n"
            f"{prompt}\n"
            f"===== Response [ID: {trace_id}] [{timestamp}] =====\n"
            f"agent: {self.name}\n"
            "response:\n"
            f"{response}\n\n"
        )
        LOGGER.append_to_txt(entry, "game_log.txt")

    def chat_with_retries(
        self,
        base_prompt: str,
        parse_func: Callable[[str], Any],
        *,
        max_retries: int = 5,
    ) -> tuple[str, str, Any]:
        """Chat with the agent, retrying if unique parsing fails."""
        response = ""
        error_reason = ""

        for attempt in range(max_retries + 1):
            if attempt == 0:
                prompt = base_prompt
            else:
                prompt = self._build_retry_prompt(
                    base_prompt, response, error_reason
                )
            response, trace_id = self.chat(prompt)
            try:
                return response, trace_id, parse_func(response)
            except ValueError as e:
                error_reason = str(e)
                print(
                    f"Attempt {attempt + 1} of {self.name} to parse response failed: "
                    f"{self._truncate_string(error_reason)} from response {self._truncate_string(response)!r}"
                )
        raise ValueError(
            f"Failed to parse response for {self.name} after {1 + max_retries} attempts. "
            f"Last error: {error_reason}. Last response: {response!r}"
        )

    @staticmethod
    def _truncate_string(s: str, max_chars: int = 300) -> str:
        """Truncate string to show first and last max_chars characters."""
        if len(s) <= 2 * max_chars:
            return s
        return f"{s[:max_chars]}...[truncated due to length]...{s[-max_chars:]}"

    @staticmethod
    def _build_retry_prompt(
        base_prompt: str, bad_response: str, error_reason: str
    ) -> str:
        """Restate the prompt, show prior response and ask for regeneration."""
        return (
            f"{base_prompt}\n\n"
            f"Your previous response was:\n{bad_response}\n\n"
            f"That response is INVALID because: {error_reason}\n\n"
            f"Please give the new output again!"
        )

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the model type and prompt strategy."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return f"{self.agent_type}#P{self.player_id}"

    def serialize(self) -> dict:
        """Return the LLM configuration dictionary for this agent."""
        return self.agent_config

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name


class IOAgent(Agent):
    """Input/Output agent that forces models to reply with bare answers (no reasoning preamble)."""

    def chat(
        self,
        messages: str,
    ) -> tuple[str, str]:
        """Append terse formatting instructions so the LLM returns only the action JSON."""
        messages += textwrap.dedent(
            """
            Please ONLY provide the output to the above question.
            DO NOT provide any additional text or explanation.
            """
        )
        return self.invoke(messages)

    @property
    def agent_type(self) -> str:
        """Expose the `(IO)` suffix so downstream logging distinguishes prompt styles."""
        return f"{self.model_type}(IO)"


class CoTAgent(Agent):
    """Chain-of-Thought agent that explicitly asks the model to reason before answering."""

    def chat(
        self,
        messages: str,
    ) -> tuple[str, str]:
        """Add structured reasoning instructions, then fall back to the same logging pipeline."""
        messages += textwrap.dedent(
            """
            Think about the question step by step.
            Break it down into small steps.
            Explain your reasoning, and then provide the final answer.
            """
        )
        return self.invoke(messages)

    @property
    def agent_type(self) -> str:
        """Expose the `(CoT)` suffix to keep wandb/log output aligned with prompt type."""
        return f"{self.model_type}(CoT)"
