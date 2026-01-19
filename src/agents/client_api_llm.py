"""Hosted-model :class:`LLM` implementation backed by OpenAI-compatible APIs."""

import time
from typing import Any

from openai import OpenAI, OpenAIError

from config import settings
from src.agents.base import LLM


class ClientAPILLM(LLM):
    """Thin wrapper around hosted LLM APIs (OpenAI-compatible clients)."""

    def __init__(self, model_name: str, provider: str):
        self.provider = provider
        self.model_name = model_name
        self.client = self._get_client()

    def _get_client(self) -> OpenAI:
        """Initialise an :class:`OpenAI` client for the configured provider."""
        match self.provider:
            case "OpenAI":
                return OpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    # Retry/timeout config
                    max_retries=3,
                    timeout=60,
                )
            case "Gemini":
                return OpenAI(
                    api_key=settings.GEMINI_API_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    max_retries=3,
                    timeout=60,
                )
            case "OpenRouter":
                return OpenAI(
                    api_key=settings.OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1",
                    max_retries=3,
                    timeout=60,
                )
            case _:
                raise ValueError(f"Unknown provider {self.provider}")

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Call the remote chat completion endpoint with basic retry/backoff."""
        # Extract OpenRouter-specific parameters that need to go in extra_body
        extra_body = {}
        if "reasoning" in kwargs:
            extra_body["reasoning"] = kwargs.pop("reasoning")

        # Simple retry/backoff around the API call
        delays = [2**i for i in range(1,8)]
        for attempt, delay in enumerate([0] + delays):
            if delay:
                time.sleep(delay)
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body=extra_body if extra_body else None,
                    **kwargs,
                )
                return completion.choices[0].message.content
            except OpenAIError as e:
                # On last attempt, re-raise
                if attempt == len(delays):
                    raise e
                continue
        raise RuntimeError("Unknown error invoking client API")
