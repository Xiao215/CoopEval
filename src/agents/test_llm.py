"""Test LLM instance for fast local testing without API calls."""

from typing import Any

from src.agents.base import LLM


class TestInstance(LLM):
    """A fake LLM that returns canned responses for testing mechanisms without API calls."""

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Return appropriate test responses based on the prompt content."""

        # Contract design (Contracting mechanism)
        if "design and propose" in prompt.lower() and "contract" in prompt.lower():
            return '{"A0": 5, "A1": -2}'  # Simple contract

        # Contract voting (Contracting mechanism)
        if "approval voting" in prompt.lower() and "contract" in prompt.lower():
            return '{"C1": true, "C2": false}'  # Approve first contract

        # Contract signing (Contracting mechanism)
        if "option to sign" in prompt.lower():
            return '{"sign": true}'  # Always sign

        # Mediator design (Mediation mechanism)
        if "design and propose" in prompt.lower() and "mediator" in prompt.lower():
            return '{"1": "A0", "2": "A1"}'  # Simple mediator

        # Mediator voting (Mediation mechanism)
        if "approval voting" in prompt.lower() and "mediator" in prompt.lower():
            return '{"M1": true, "M2": false}'  # Approve first mediator

        # Disarmament negotiation
        if "disarmament" in prompt.lower() or "upper bound" in prompt.lower():
            # Check if this is first call (caps at 100) or subsequent
            if '"A0"=100' in prompt or '"A1"=100' in prompt:
                # First round: continue with reduced caps
                return '{"choice": "disarm", "A0": 80, "A1": 20}'
            else:
                # Subsequent rounds: end negotiation
                return '{"choice": "end"}'

        # Default: Return action distribution (for base game)
        return '{"A0": 60, "A1": 40}'  # 60% cooperate, 40% defect
