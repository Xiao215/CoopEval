"""
Core LLM Judge functionality for text classification using custom taxonomies.
"""

import asyncio
import json
import textwrap
from typing import Any

from inspect_ai.model import GenerateConfig, Model


class LLMJudge:
    """
    Main class for performing LLM-based text classification.
    """

    def __init__(
        self, api_client: Model, taxonomy, temperature: float = 0
    ):
        """
        Initialize the LLM Judge.

        Args:
            api_client: Inspect AI Model instance
            taxonomy: Taxonomy instance with categories and definitions
            temperature: Temperature for LLM generation (0 = deterministic)
        """
        self.api_client = api_client
        self.taxonomy = taxonomy
        self.temperature = temperature

    def classify_text(self, text: str, max_tokens: int = 500) -> dict[str, Any]:
        """
        Classify a single text using the LLM judge (synchronous wrapper).

        Args:
            text: Text to classify
            max_tokens: Maximum tokens for LLM response

        Returns:
            Dictionary with classification results
        """
        return asyncio.run(self.classify_text_async(text, max_tokens))

    async def classify_text_async(self, text: str, max_tokens: int = 500) -> dict[str, Any]:
        """
        Classify a single text using the LLM judge (async version).

        Args:
            text: Text to classify
            max_tokens: Maximum tokens for LLM response

        Returns:
            Dictionary with classification results
        """
        prompt = self._build_classification_prompt(text)

        try:
            res = await self.api_client.generate(
                prompt,
                config=GenerateConfig(
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                ),
            )
            response = res.message.text

            json_result = self._extract_json_from_response(response)

            if json_result:
                result = json.loads(json_result)
                result["Confidence"] = float(result.get("Confidence", 0))
                return result
            else:
                return {
                    "Reasoning_behind_classification": response,
                    "Confidence": 0.0,
                    "justification_type": "Other",
                }

        except json.JSONDecodeError as exc:
            return {
                "Reasoning_behind_classification": f"Invalid JSON in response: {exc}",
                "Confidence": 0.0,
                "justification_type": "Failed classification",
            }

    def _build_classification_prompt(self, text: str) -> str:
        """Build the classification prompt using the taxonomy."""

        taxonomy_text = self.taxonomy.get_formatted_taxonomy()

        prompt_template = textwrap.dedent("""\
            Analyze the following text and categorize the decision-making strategy used.
            You may choose one, multiple or none of the classes. If none apply, classify as other.

            {taxonomy}

            Text to analyze:
            \"\"\"
            {text}
            \"\"\"

            IMPORTANT: Your response MUST be in valid JSON format EXACTLY as shown below. Do not include any explanatory text outside of the JSON structure.

            Example of the required JSON format:
            {{
              "Reasoning_behind_classification": "Explanation of your classification reasoning",
              "Confidence": 0.85,
              "justification_type": <Justification Type>, <Justification Type>, ...
            }}

            Ensure that:
            1. Your JSON is properly formatted with no trailing commas
            2. "Confidence" is a decimal number between 0 and 1, not a string
            3. For multiple justification types, list them as a comma-separated string
            4. Don't include any text outside the JSON object
            """)
        return prompt_template.format(taxonomy=taxonomy_text, text=text)

    @staticmethod
    def _extract_json_snippet(text: str) -> str | None:
        """Slice and validate the first JSON object embedded in ``text``."""
        if not text:
            return None

        brace_start = text.find("{")
        if brace_start == -1:
            return None

        decoder = json.JSONDecoder()
        try:
            _, end = decoder.raw_decode(text[brace_start:])
        except json.JSONDecodeError:
            return None

        return text[brace_start : brace_start + end]

    def _extract_json_from_response(self, response: str) -> str | None:
        """Return the JSON payload found inside an LLM response, if any."""

        snippet = self._extract_json_snippet(response)
        if snippet is not None:
            return snippet

        normalized = response.replace("'", '"')
        if normalized == response:
            return None

        return self._extract_json_snippet(normalized)
