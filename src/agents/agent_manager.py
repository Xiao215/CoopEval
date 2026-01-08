"""Agent abstractions and shared LLM caching utilities."""

import itertools
from datetime import datetime
from abc import ABC, abstractmethod
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
                # Default to OpenAI API based models
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

        if llm_config.get("uid") is None:
            self.uid = next(type(self)._instance_counter)
            llm_config["uid"] = self.uid
        else:
            # UID is provided only for reconstructing from a prior run
            self.uid = int(llm_config["uid"])
            next_uid = next(type(self)._instance_counter)
            while next_uid <= self.uid:
                next_uid = next(type(self)._instance_counter)

        self.player_id: int = agent_config["player_id"]
        self.agent_config = agent_config

    @abstractmethod
    def chat(
        self,
        messages: str,
    ) -> str:
        """Chat with the agent using the provided messages."""
        raise NotImplementedError

    def invoke(self, messages: str) -> str:
        """Invoke the agent using the provided messages. No prompting added."""
        return self._invoke_with_logging(messages)

    def _invoke_with_logging(self, messages: str) -> str:
        response = self.pipeline.invoke(messages, **self.kwargs)
        self._log_inference(messages, response)
        return response

    def _log_inference(self, prompt: str, response: str) -> None:
        entry = (
            "=== inference ===\n"
            f"timestamp: {datetime.now().isoformat()}\n"
            f"agent: {self.name}\n"
            "prompt:\n"
            f"{prompt}\n"
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
    ) -> tuple[str, Any]:
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

            response = self.chat(prompt)

            try:
                return response, parse_func(response)
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
    def name(self) -> str:
        """Return the name of the agent."""
        raise NotImplementedError

    def get_agent_config(self) -> dict:
        """Return the LLM configuration dictionary for this agent."""
        return self.agent_config

    def __str__(self):
        return self.name


class IOAgent(Agent):
    """Input/Output Agent.
    This agent is designed to be the most basic llm agent. Given a message, answer it.
    """

    def chat(
        self,
        messages: str,
    ) -> str:
        """Chat with the agent using the provided messages."""
        messages += (
            "\nPlease ONLY provide the output to the above question."
            "DO NOT provide any additional text or explanation.\n"
        )
        return self._invoke_with_logging(messages)

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return f"{self.model_type}(IO)[#{self.uid}]#P{self.player_id}"


class CoTAgent(Agent):
    """Chain-of-Thought Agent.

    This agent wraps the prompt to ask the LLM to think step-by-step.
    """

    def chat(
        self,
        messages: str,
    ) -> str:
        """Chat with the agent using the provided messages."""
        messages += """
        Think about the question step by step, break it down into small steps, explain your reasoning, and then provide the final answer.
        """
        return self._invoke_with_logging(messages)

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return f"{self.model_type}(CoT)[#{self.uid}]#P{self.player_id}"
