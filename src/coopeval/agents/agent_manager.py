"""Agent abstractions and shared model utilities."""

import textwrap
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)

from coopeval.agents.test_llm import TestModel
from coopeval.logger_manager import LOGGER

GENERATE_CONFIG_KEYS = set(GenerateConfig.model_fields)


class Agent(ABC):
    """Abstract agent wrapper that owns player metadata and an underlying LLM."""

    def __init__(self, agent_config: dict) -> None:
        """Store config, resolve the requested pipeline, and capture player metadata."""
        llm_config = agent_config["llm"]
        self.model_type = llm_config["model"]
        self.provider = llm_config["provider"]
        llm_kwargs = llm_config.get("kwargs", {})
        unknown_kwargs = set(llm_kwargs) - GENERATE_CONFIG_KEYS
        if unknown_kwargs:
            raise ValueError(
                f"Unsupported llm.kwargs for {self.provider}/{self.model_type}: "
                f"{sorted(unknown_kwargs)}. Expected Inspect GenerateConfig keys."
            )
        generate_config = GenerateConfig(**llm_kwargs)

        if self.provider == "TestModel":
            self.model = TestModel()
        else:
            model_name = f"{self.provider.lower()}/{self.model_type}"
            self.model = get_model(
                model_name,
                config=generate_config,
            )

        self.player_id: int = agent_config["player_id"]
        self.agent_config = agent_config

    @abstractmethod
    async def chat(
        self,
        messages: str,
    ) -> tuple[str, str]:
        """Generate one response + trace ID for the given serialized prompt."""
        raise NotImplementedError

    async def _generate_response(
        self,
        messages: str,
        system_prompt: str | None = None,
    ) -> tuple[str, str]:
        """Generate a model response and attach a short trace identifier."""
        trace_id = str(uuid.uuid4())[:8]

        inspect_messages = []
        if system_prompt:
            inspect_messages.append(ChatMessageSystem(content=system_prompt))
        inspect_messages.append(ChatMessageUser(content=messages))

        res = await self.model.generate(inspect_messages)
        response = res.message.text

        self._log_inference(messages, response, trace_id)
        return response, trace_id

    def _log_inference(self, prompt: str, response: str, trace_id: str) -> None:
        """Write a prompt/response pair to the shared game log with timestamps."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

    async def chat_with_retries(
        self,
        base_prompt: str,
        parse_func: Callable[[str], Any],
        *,
        max_retries: int = 5,
    ) -> tuple[str, str, Any]:
        """Retry `chat` with richer context until `parse_func` succeeds or retries exhaust."""
        response = ""
        error_reason = ""

        for attempt in range(max_retries + 1):
            if attempt == 0:
                prompt = base_prompt
            else:
                prompt = self._build_retry_prompt(
                    base_prompt, response, error_reason
                )
            response, trace_id = await self.chat(prompt)
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
        """Limit long log strings by keeping both the head and tail fragments."""
        if len(s) <= 2 * max_chars:
            return s
        return f"{s[:max_chars]}...[truncated due to length]...{s[-max_chars:]}"

    @staticmethod
    def _build_retry_prompt(
        base_prompt: str, bad_response: str, error_reason: str
    ) -> str:
        """Produce a corrective prompt that explains why the previous attempt failed."""
        return (
            f"{base_prompt}\n\n"
            f"Your previous response was:\n{bad_response}\n\n"
            f"That response is INVALID because: {error_reason}\n\n"
            f"Please give the new output again!"
        )

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the display prefix that differentiates IO vs CoT behavior."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Compose the user-facing agent label including the seat number."""
        return f"{self.agent_type}#P{self.player_id}"

    def serialize(self) -> dict:
        """Emit the original agent configuration for persistence."""
        return self.agent_config

    def __str__(self):
        """Return the printable identifier used throughout logs."""
        return self.name

    def __eq__(self, other):
        """Compare agents by their canonical string identity."""
        return self.name == other.name

    def __hash__(self):
        """Hash agents via their stable name representation."""
        return hash(self.name)

    def __lt__(self, other):
        """Order agents lexicographically by the string label."""
        return self.name < other.name


class IOAgent(Agent):
    """Input/Output agent that forces models to reply with bare answers (no reasoning preamble)."""

    async def chat(
        self,
        messages: str,
    ) -> tuple[str, str]:
        """Use Inspect's system prompt abstractions to enforce bare answers."""
        system_prompt = textwrap.dedent("""\
            Please ONLY provide the output to the above question.
            DO NOT provide any additional text or explanation.""")
        return await self._generate_response(
            messages,
            system_prompt=system_prompt,
        )

    @property
    def agent_type(self) -> str:
        """Expose the `(IO)` suffix so downstream logging distinguishes prompt styles."""
        return f"{self.model_type}(IO)"


class CoTAgent(Agent):
    """Chain-of-Thought agent that explicitly asks the model to reason before answering."""

    async def chat(
        self,
        messages: str,
    ) -> tuple[str, str]:
        """Use Inspect's system prompt abstractions to enforce Chain-of-Thought reasoning."""
        system_prompt = textwrap.dedent("""\
            Think about the question step by step.
            Break it down into small steps.
            Explain your reasoning, and then provide the final answer.""")
        return await self._generate_response(
            messages,
            system_prompt=system_prompt,
        )

    @property
    def agent_type(self) -> str:
        """Expose the `(CoT)` suffix to keep wandb/log output aligned with prompt type."""
        return f"{self.model_type}(CoT)"
