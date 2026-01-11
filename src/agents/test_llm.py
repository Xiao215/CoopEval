"""Test LLM instance for fast local testing without API calls."""

import json
import random
import re
from typing import Any

from src.agents.base import LLM


class TestInstance(LLM):
    """A fake LLM that returns canned responses for testing mechanisms without API calls."""

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        if self._is_disarm_prompt(prompt):
            return self._fake_disarm_response(prompt)
        if self._is_mediator_design_prompt(prompt):
            return self._fake_mediator_design_response(prompt)
        if self._is_mediator_vote_prompt(prompt):
            return self._fake_mediator_vote_response(prompt)
        if self._is_contract_design_prompt(prompt):
            return self._fake_contract_design_response(prompt)
        if self._is_contract_vote_prompt(prompt):
            return self._fake_contract_vote_response(prompt)
        if self._is_contract_confirmation_prompt(prompt):
            return self._fake_contract_confirmation_response(prompt)
        if self._is_action_making_prompt(prompt):
            # Parse the action template to determine number of actions
            num_actions = self._count_actions_in_template(prompt)

            caps = self._parse_disarm_caps_from_prompt(prompt)
            if (
                caps
                and len(caps) >= num_actions
                and sum(caps[:num_actions]) >= 100
            ):
                ints = self._sample_capped_distribution(caps[:num_actions])
            else:
                ints = self._random_distribution(num_actions)

            distribution = {f"A{i}": value for i, value in enumerate(ints)}
            return json.dumps(distribution)
        return '{"A0": 100, "A1": 0}'  # always return action A0 with 100 points

    def _is_disarm_prompt(self, prompt: str) -> bool:
        return self._matches(prompt, self._disarm_prompt_patterns())

    def _is_action_making_prompt(self, prompt: str) -> bool:
        return self._matches(prompt, self._action_template_patterns())

    def _fake_disarm_response(self, prompt: str) -> str:
        if random.random() < 0.02:
            return '{"choice": "end"}'

        if re.search(r"upper bounds already sum to 100", prompt, re.IGNORECASE):
            return '{"choice": "pass"}'

        current_caps = self._parse_disarm_caps_from_prompt(prompt)
        if not current_caps or sum(current_caps) <= 100:
            raise ValueError("Cannot parse valid disarm caps from prompt.")

        new_caps = list(current_caps)
        for idx, cap in enumerate(new_caps):
            if cap > 0:
                new_caps[idx] = max(0, cap - random.randint(1, 10))
                break

        if new_caps == current_caps or sum(new_caps) < 100:
            return '{"choice": "pass"}'

        response: dict[str, int | str] = {"choice": "disarm"}
        response.update(
            {f"A{i}": int(value) for i, value in enumerate(new_caps)}
        )
        return json.dumps(response)

    def _parse_disarm_caps_from_prompt(self, prompt: str) -> list[int] | None:
        match = re.search(
            r"Your(?: current)? upper bounds:\s*(\{.*?\})",
            prompt,
            re.DOTALL,
        )
        if not match:
            return None

        caps_str = match.group(1)
        caps_pairs = re.findall(
            r'"A(\d+)"\s*=\s*(\d+)',
            caps_str,
        )
        if not caps_pairs:
            return None

        caps_by_index = {int(key): int(value) for key, value in caps_pairs}
        num_actions = max(caps_by_index) + 1
        return [caps_by_index.get(i, 0) for i in range(num_actions)]

    def _sample_capped_distribution(self, caps: list[int]) -> list[int]:
        num_actions = len(caps)
        order = list(range(num_actions))
        random.shuffle(order)

        remaining = 100
        remaining_caps_sum = sum(caps)
        distribution = [0] * num_actions

        for idx in order:
            remaining_caps_sum -= caps[idx]
            min_value = max(0, remaining - remaining_caps_sum)
            max_value = min(caps[idx], remaining)
            distribution[idx] = random.randint(min_value, max_value)
            remaining -= distribution[idx]

        return distribution

    def _random_distribution(self, num_actions: int) -> list[int]:
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
        return ints

    def _count_actions_in_template(self, prompt: str) -> int:
        """Count the number of action keys in the template by finding all A0, A1, A2, etc."""
        # First, try to find action definitions in the "Actions available" section
        # This works for all games that list actions like "- A0:", "- A1:", etc.
        action_keys = re.findall(r'^\s*-\s*A(\d+)', prompt, re.MULTILINE)
        assert action_keys, "No action keys found in the prompt."
        # Return the maximum action index + 1 (since A0 is the first action)
        return max(int(key) for key in action_keys) + 1

    def _fake_mediator_design_response(self, prompt: str) -> str:
        num_players = self._count_mediator_players_in_prompt(prompt)
        num_actions = self._count_actions_in_template(prompt)
        mediator = {
            str(player_count): f"A{random.randrange(num_actions)}"
            for player_count in range(1, num_players + 1)
        }
        return json.dumps(mediator)

    def _fake_mediator_vote_response(self, prompt: str) -> str:
        num_mediators = self._count_mediators_in_prompt(prompt)
        approvals = {
            f"M{i}": random.choice([True, False])
            for i in range(1, num_mediators + 1)
        }
        if not any(approvals.values()):
            approvals[f"M{random.randint(1, num_mediators)}"] = True
        return json.dumps(approvals)

    def _fake_contract_design_response(self, prompt: str) -> str:
        num_actions = self._count_actions_in_template(prompt)
        favored_action = random.randrange(num_actions)
        contract = []
        for idx in range(num_actions):
            if idx == favored_action:
                contract.append(random.randint(5, 20))
            else:
                contract.append(random.randint(-10, 5))
        return json.dumps(
            {f"A{i}": value for i, value in enumerate(contract)}
        )

    def _fake_contract_vote_response(self, prompt: str) -> str:
        num_contracts = self._count_contracts_in_prompt(prompt)
        target = random.randint(1, num_contracts)
        approvals = {}
        for idx in range(1, num_contracts + 1):
            if idx == target:
                approvals[f"C{idx}"] = True
            else:
                approvals[f"C{idx}"] = random.random() < 0.25
        return json.dumps(approvals)

    def _fake_contract_confirmation_response(self, prompt: str) -> str:
        return json.dumps({"sign": random.random() < 0.7})

    def _count_mediator_players_in_prompt(self, prompt: str) -> int:
        match = re.search(r"from 1 to (\d+)", prompt)
        if not match:
            raise ValueError("Cannot determine number of delegating players.")
        return int(match.group(1))

    def _count_mediators_in_prompt(self, prompt: str) -> int:
        mediators = re.findall(
            r"^Mediator proposed by Player",
            prompt,
            re.MULTILINE,
        )
        if mediators:
            return len(mediators)
        ids = re.findall(r"\bM(\d+)\b", prompt)
        if ids:
            return max(int(value) for value in ids)
        raise ValueError("Cannot determine number of mediators.")

    def _count_contracts_in_prompt(self, prompt: str) -> int:
        contracts = re.findall(
            r"^Contract proposed by Player",
            prompt,
            re.MULTILINE,
        )
        if contracts:
            return len(contracts)
        ids = re.findall(r"\bC(\d+)\b", prompt)
        if ids:
            return max(int(value) for value in ids)
        raise ValueError("Cannot determine number of contracts.")

    def _disarm_prompt_patterns(self) -> tuple[str, ...]:
        return (r'"choice"\s*:\s*"(?:disarm|pass|end)"',)

    def _mediator_design_prompt_patterns(self) -> tuple[str, ...]:
        return (r"design and propose a mediator",)

    def _mediator_vote_prompt_patterns(self) -> tuple[str, ...]:
        return (r"mediator designs that have been proposed",)

    def _action_template_patterns(self) -> tuple[str, ...]:
        return (
            r'\{\s*"A0"\s*:\s*(?:-?\d+|<[Ii][Nn][Tt]>)\s*,\s*"A1"\s*:\s*(?:-?\d+|<[Ii][Nn][Tt]>)(?:\s*,\s*(?:"A\d+"\s*:\s*(?:-?\d+|<[Ii][Nn][Tt]>)|\.\.\.))*\s*\}',
        )

    def _contract_design_prompt_patterns(self) -> tuple[str, ...]:
        return (r"(?s)payment contract.*design and propose one",)

    def _contract_vote_prompt_patterns(self) -> tuple[str, ...]:
        return (r"contract designs that have been proposed",)

    def _contract_confirmation_prompt_patterns(self) -> tuple[str, ...]:
        return (r"option to sign a payment contract",)

    def _matches(self, prompt: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, prompt) for pattern in patterns)

    def _is_mediator_design_prompt(self, prompt: str) -> bool:
        return self._matches(prompt, self._mediator_design_prompt_patterns())

    def _is_mediator_vote_prompt(self, prompt: str) -> bool:
        return self._matches(prompt, self._mediator_vote_prompt_patterns())

    def _is_contract_design_prompt(self, prompt: str) -> bool:
        return self._matches(prompt, self._contract_design_prompt_patterns())

    def _is_contract_vote_prompt(self, prompt: str) -> bool:
        return self._matches(prompt, self._contract_vote_prompt_patterns())

    def _is_contract_confirmation_prompt(self, prompt: str) -> bool:
        return self._matches(
            prompt, self._contract_confirmation_prompt_patterns()
        )
