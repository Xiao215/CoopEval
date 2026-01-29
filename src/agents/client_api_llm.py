"""Hosted-model :class:`LLM` implementation backed by OpenAI-compatible APIs."""

import json
import time
from typing import Any

import httpx
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
        # OpenRouter threads "reasoning" config through `extra_body`; strip it from kwargs before calling the client.
        extra_body = {}
        if "reasoning" in kwargs:
            extra_body["reasoning"] = kwargs.pop("reasoning")

        # Exponential backoff to survive transient HTTP / rate-limit failures.
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
                # Pull the first choice; do it inside the try to surface malformed payloads.
                choice = completion.choices[0]

                # Some providers embed per-choice errors (quota, filter hits); surface them as OpenAIError.
                if hasattr(choice, 'error') and choice.error:
                    error_msg = choice.error.get('message', 'Unknown error')
                    error_code = choice.error.get('code', 'unknown')
                    raise OpenAIError(f"API error (code {error_code}): {error_msg}")

                content = choice.message.content
                if not content or content.strip() == "":
                    raise OpenAIError("API returned empty response content")
                return content
            except Exception as e:
                # Treat everything as retryable (httpx, OpenAIError, JSON issues); bail only after exhausting delays.
                if attempt == len(delays):
                    raise e
                # Keep a breadcrumb so operators know why we're retrying without flooding logs.
                print(f"API call failed (attempt {attempt + 1}/{len(delays) + 1}): {type(e).__name__}: {e}")
                continue
        raise RuntimeError("Unknown error invoking client API")
