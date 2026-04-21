"""Inspect-compatible fake model for fast local tests without API calls."""

import json
import random
import re
from collections.abc import Callable
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    Model,
    ModelAPI,
    ModelOutput,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

DEFAULT_ACTION_RESPONSE = '{"A0": 100, "A1": 0}'

MEDIATOR_DESIGN_RE = re.compile(r"design and propose a mediator")
MEDIATOR_VOTE_RE = re.compile(r"mediator designs that have been proposed")
CONTRACT_DESIGN_RE = re.compile(
    r"payment contract.*design and propose one",
    re.DOTALL,
)
CONTRACT_VOTE_RE = re.compile(r"contract designs that have been proposed")
CONTRACT_CONFIRMATION_RE = re.compile(r"option to sign a payment contract")
ACTION_TEMPLATE_RE = re.compile(
    r'\{\s*"A0"\s*:\s*(?:-?\d+|<[Ii][Nn][Tt]>)'
    r'\s*,\s*"A1"\s*:\s*(?:-?\d+|<[Ii][Nn][Tt]>)'
    r'(?:\s*,\s*(?:"A\d+"\s*:\s*(?:-?\d+|<[Ii][Nn][Tt]>)|\.\.\.))*\s*\}'
)
ACTION_KEY_RE = re.compile(r"^\s*-\s*A(\d+)", re.MULTILINE)
ACTION_TOKEN_RE = re.compile(r"\bA(\d+)\b")


class TestModel(Model):
    """Fake Inspect model that returns deterministic-shape JSON responses."""

    def __init__(self) -> None:
        config = GenerateConfig()
        api = test_provider(
            model_name="testmodel/test-instance",
            config=config,
        )
        super().__init__(api=api, config=config)


class TestModelAPI(ModelAPI):
    """In-process Inspect model API used by TestModel."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig = GenerateConfig(),
        **_model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=api_key_vars or [],
            config=config,
        )

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        del tools, tool_choice, config
        prompt = self._prompt_text(input)
        return ModelOutput.from_content(
            self.model_name,
            self._response_for_prompt(prompt),
        )

    def _response_for_prompt(self, prompt: str) -> str:
        for pattern, responder in self._prompt_routes():
            if pattern.search(prompt):
                return responder(prompt)

        if ACTION_TEMPLATE_RE.search(prompt):
            return self._fake_action_response(prompt)
        return DEFAULT_ACTION_RESPONSE

    def _prompt_routes(
        self,
    ) -> tuple[tuple[re.Pattern[str], Callable[[str], str]], ...]:
        return (
            (MEDIATOR_DESIGN_RE, self._fake_mediator_design_response),
            (MEDIATOR_VOTE_RE, self._fake_mediator_vote_response),
            (CONTRACT_DESIGN_RE, self._fake_contract_design_response),
            (CONTRACT_VOTE_RE, self._fake_contract_vote_response),
            (
                CONTRACT_CONFIRMATION_RE,
                self._fake_contract_confirmation_response,
            ),
        )

    def _prompt_text(self, messages: list[ChatMessage]) -> str:
        return "\n\n".join(self._message_text(message) for message in messages)

    def _message_text(self, message: ChatMessage) -> str:
        text = getattr(message, "text", None)
        if isinstance(text, str):
            return text

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content

        return "\n".join(str(item) for item in content)

    def _fake_action_response(self, prompt: str) -> str:
        num_actions = self._count_actions_in_template(prompt)
        distribution = self._random_distribution(num_actions)

        return json.dumps(
            {f"A{i}": value for i, value in enumerate(distribution)}
        )

    def _fake_mediator_design_response(self, prompt: str) -> str:
        num_players = self._count_mediator_players(prompt)
        num_actions = self._count_actions_in_template(prompt)
        return json.dumps(
            {
                str(player_count): f"A{random.randrange(num_actions)}"
                for player_count in range(1, num_players + 1)
            }
        )

    def _fake_mediator_vote_response(self, prompt: str) -> str:
        num_mediators = self._count_labeled_items(
            prompt,
            label="Mediator",
            prefix="M",
        )
        approvals = self._random_approval_map("M", num_mediators)
        if not any(approvals.values()):
            approvals[f"M{random.randint(1, num_mediators)}"] = True
        return json.dumps(approvals)

    def _fake_contract_design_response(self, prompt: str) -> str:
        num_actions = self._count_actions_in_template(prompt)
        favored_action = random.randrange(num_actions)
        contract = [
            (
                random.randint(5, 20)
                if idx == favored_action
                else random.randint(-10, 5)
            )
            for idx in range(num_actions)
        ]
        return json.dumps({f"A{i}": value for i, value in enumerate(contract)})

    def _fake_contract_vote_response(self, prompt: str) -> str:
        num_contracts = self._count_labeled_items(
            prompt,
            label="Contract",
            prefix="C",
        )
        target = random.randint(1, num_contracts)
        approvals = {
            f"C{idx}": idx == target or random.random() < 0.25
            for idx in range(1, num_contracts + 1)
        }
        return json.dumps(approvals)

    def _fake_contract_confirmation_response(self, _prompt: str) -> str:
        return json.dumps({"sign": random.random() < 0.7})

    def _random_distribution(self, num_actions: int) -> list[int]:
        weights = [random.random() for _ in range(num_actions)]
        total = sum(weights) or 1.0
        raw = [weight / total * 100 for weight in weights]
        distribution = [int(value) for value in raw]

        fractional = sorted(
            enumerate(raw),
            key=lambda item: item[1] - int(item[1]),
            reverse=True,
        )
        for i in range(100 - sum(distribution)):
            distribution[fractional[i % num_actions][0]] += 1

        return distribution

    def _count_actions_in_template(self, prompt: str) -> int:
        action_indices = [int(key) for key in ACTION_KEY_RE.findall(prompt)]

        # Mediation introduces the delegate action in mechanism text rather
        # than in the base game's bullet list.
        action_indices.extend(int(key) for key in ACTION_TOKEN_RE.findall(prompt))

        if not action_indices:
            raise ValueError("No action keys found in the prompt.")
        return max(action_indices) + 1

    def _count_mediator_players(self, prompt: str) -> int:
        match = re.search(r"from 1 to (\d+)", prompt)
        if not match:
            raise ValueError("Cannot determine number of delegating players.")
        return int(match.group(1))

    def _count_labeled_items(
        self,
        prompt: str,
        *,
        label: str,
        prefix: str,
    ) -> int:
        headings = re.findall(
            rf"^{label} proposed by Player",
            prompt,
            re.MULTILINE,
        )
        if headings:
            return len(headings)

        ids = re.findall(rf"\b{prefix}(\d+)\b", prompt)
        if ids:
            return max(int(value) for value in ids)

        raise ValueError(f"Cannot determine number of {label.lower()}s.")

    def _random_approval_map(self, prefix: str, count: int) -> dict[str, bool]:
        return {
            f"{prefix}{idx}": random.choice([True, False])
            for idx in range(1, count + 1)
        }


@modelapi(name="testmodel")
def test_provider() -> type[ModelAPI]:
    return TestModelAPI
