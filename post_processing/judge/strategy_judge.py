#!/usr/bin/env python3
"""
LLM-based strategy classifier for repeated Prisoner's Dilemma matches.

This script reads the "readable" JSON produced by
``post_processing/jsonl_to_readable_json.py`` and, for each player in the
selected match, constructs a rich per-round history (actions, probabilities,
points, free-form reasoning, and synchronous opponent context).  The history is
sent to an OpenRouter-hosted model that applies a fixed rubric of strategy
labels.

Example usage
-------------

    python post_processing/judge/strategy_judge.py \
        --input outputs/2025/11/03/18:06_probabilistic_deepseek/ \
            Repetition_PrisonersDilemma_readable.json \
        --match-index 1 \
        --model anthropic/claude-3.5-sonnet \
        --output outputs/2025/11/03/18:06_probabilistic_deepseek/strategy_judgements.json

The output is a JSON document containing one entry per ``player_slot`` with the
classified ``label`` and short ``justification`` returned by the judge model,
along with the prompt metadata used for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from openai import OpenAI
from openai.types.chat import ChatCompletion

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_env_file() -> None:
    """Populate os.environ with entries from the project's `.env` file."""
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip()

load_env_file()


JUDGE_LABELS = [
    "ALWAYS_COOPERATE",
    "ALWAYS_DEFECT",
    "GRIM_TRIGGER",
    "TIT_FOR_TAT",
    "GENEROUS_TIT_FOR_TAT",
    "WIN_STAY_LOSE_SHIFT",
    "FINITE_TRIGGER_PUNISHMENT",
    "OTHER_STRATEGIC",
    "RANDOM",
]

JUDGE_PROMPT_TEMPLATE = """You are a strategy classifier for repeated Prisoner's Dilemma behaviour.

You receive the full decision history of one participant. Each round contains:
  - the participant's action,
  - actions and payoffs for all seats,
  - the participant's reasoning,
  - the opponent context at the same point in time.

Your task: infer which strategy the participant is most likely following.
Choose exactly ONE label from the list:
1. ALWAYS_COOPERATE
2. ALWAYS_DEFECT
3. GRIM_TRIGGER
4. TIT_FOR_TAT
5. GENEROUS_TIT_FOR_TAT
6. WIN_STAY_LOSE_SHIFT
7. FINITE_TRIGGER_PUNISHMENT
8. OTHER_STRATEGIC
9. RANDOM

Definitions:
- ALWAYS_COOPERATE: cooperates essentially every round regardless of history.
- ALWAYS_DEFECT: defects essentially every round regardless of history.
- GRIM_TRIGGER: starts cooperative; after the first opponent defection, defects forever.
- TIT_FOR_TAT: cooperates first round, then mirrors opponent's last action.
- GENEROUS_TIT_FOR_TAT: mostly mirrors last opponent action but occasionally forgives to restore cooperation.
- WIN_STAY_LOSE_SHIFT: repeats previous action after satisfactory outcome, switches after unsatisfactory outcome.
- FINITE_TRIGGER_PUNISHMENT: defaults to cooperation, punishes opponent defection for a finite window before attempting to restore cooperation.
- OTHER_STRATEGIC: purposeful pattern that does not match the definitions above.
- RANDOM: no coherent pattern; actions or rationales are inconsistent or noisy.

Input format (JSON):
{history_json}

Output format (JSON):
{{
  "label": "<one of {label_list}>",
  "justification": "<2-5 sentences referencing key rounds and logic>"
}}

Be concise but concrete: always mention the distinguishing behaviour.  Only
choose RANDOM when no coherent pattern fits.  If multiple labels seem plausible,
select the simplest one that explains most rounds.
"""


@dataclass
class PlayerRound:
    round_idx: int
    agent_action: str
    agent_points: float | None
    agent_probabilities: dict[str, int] | None
    agent_response: str | None
    opponents: list[dict[str, Any]]
    table_snapshot: list[dict[str, Any]]


def load_match(payload: Any, *, match_index: int | None) -> dict[str, Any]:
    """Extract a single match object from the readable JSON payload."""
    if isinstance(payload, list):
        if match_index is None:
            if len(payload) != 1:
                raise ValueError(
                    "Readable JSON contains multiple matches. "
                    "Specify --match-index to choose one."
                )
            return payload[0]
        if not (1 <= match_index <= len(payload)):
            raise ValueError(
                f"match-index {match_index} out of range "
                f"(payload contains {len(payload)} matches)."
            )
        return payload[match_index - 1]
    if isinstance(payload, dict):
        if match_index is not None:
            payload_index = payload.get("match_index")
            if payload_index is not None and payload_index != match_index:
                raise ValueError(
                    f"Requested match_index={match_index}, "
                    f"but JSON contains match_index={payload_index}."
                )
        return payload
    raise TypeError("Readable JSON must be an object or a list of objects.")


def iter_player_slots(match: dict[str, Any]) -> Sequence[str]:
    """Return the ordered list of player slots present in the match."""
    slots = []
    for rnd in match["rounds"]:
        for player in rnd["players"]:
            slot = player.get("player_slot") or f"{player['player_name']}#{player['uid']}"
            if slot not in slots:
                slots.append(slot)
    return slots


def build_player_history(match: dict[str, Any], player_slot: str) -> list[PlayerRound]:
    """Construct a per-round record focusing on ``player_slot``."""
    rounds: list[PlayerRound] = []
    for rnd in match["rounds"]:
        round_idx = rnd["round"]
        table_snapshot: list[dict[str, Any]] = []
        agent_entry: dict[str, Any] | None = None
        opponent_entries: list[dict[str, Any]] = []

        for entry in rnd["players"]:
            slot = entry.get("player_slot") or f"{entry['player_name']}#{entry['uid']}"
            row = {
                "player_slot": slot,
                "player_name": entry["player_name"],
                "action": entry.get("action"),
                "points": entry.get("points"),
                "probabilities": entry.get("probabilities"),
            }
            table_snapshot.append(row)

            enriched = {
                "player_slot": slot,
                "action": entry.get("action"),
                "points": entry.get("points"),
                "probabilities": entry.get("probabilities"),
                "response_excerpt": (entry.get("response") or "")[:500],
            }

            if slot == player_slot:
                agent_entry = entry
            else:
                opponent_entries.append(enriched)

        if agent_entry is None:
            # Player may not participate in every recorded round (should not happen), skip gracefully.
            continue

        rounds.append(
            PlayerRound(
                round_idx=round_idx,
                agent_action=agent_entry.get("action"),
                agent_points=agent_entry.get("points"),
                agent_probabilities=agent_entry.get("probabilities"),
                agent_response=agent_entry.get("response"),
                opponents=opponent_entries,
                table_snapshot=table_snapshot,
            )
        )
    return rounds


def serialize_history_for_prompt(
    player_slot: str,
    rounds: Sequence[PlayerRound],
) -> dict[str, Any]:
    """Prepare the JSON document fed to the judge prompt."""
    history = []
    for pr in rounds:
        row = {
            "round": pr.round_idx,
            "agent_action": pr.agent_action,
            "agent_points": pr.agent_points,
            "agent_probabilities": pr.agent_probabilities,
            "agent_reasoning": pr.agent_response,
            "opponent_context": pr.opponents,
            "table_snapshot": pr.table_snapshot,
        }
        history.append(row)
    return {
        "player_slot": player_slot,
        "rounds": history,
    }


class StrategyJudge:
    """Thin wrapper around an OpenRouter LLM that applies the strategy rubric."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 600,
    ) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY environment variable is required for judging."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_retries=3,
            timeout=90,
        )
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def classify(self, prompt: str) -> dict[str, Any]:
        """Send the prompt to the judge model and parse the JSON response."""
        completion: ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = completion.choices[0].message.content
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Judge response is not valid JSON: {response_text}"
            ) from exc

        label = parsed.get("label")
        if label not in JUDGE_LABELS:
            raise ValueError(
                f"Judge returned unsupported label {label!r}. "
                f"Expected one of {JUDGE_LABELS}."
            )
        if "justification" not in parsed:
            raise ValueError("Judge response missing 'justification' field.")
        return parsed


def build_prompt(serialized_history: dict[str, Any]) -> str:
    """Render the final prompt string sent to the LLM."""
    history_json = json.dumps(serialized_history, indent=2, ensure_ascii=False)
    return JUDGE_PROMPT_TEMPLATE.format(
        history_json=history_json,
        label_list=", ".join(JUDGE_LABELS),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Readable JSON file generated by jsonl_to_readable_json.py.",
    )
    parser.add_argument(
        "--match-index",
        type=int,
        help="Select a match when the JSON file stores multiple objects.",
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-3.5-sonnet",
        help="OpenRouter model identifier to use for judging.",
    )
    parser.add_argument(
        "--temperature",
        default=0.0,
        type=float,
        help="Temperature passed to the judge model.",
    )
    parser.add_argument(
        "--max-output-tokens",
        default=600,
        type=int,
        help="Cap on the number of tokens produced by the judge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination JSON file for judge results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling the judge API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    match = load_match(payload, match_index=args.match_index)

    judge: StrategyJudge | None = None
    if not args.dry_run:
        judge = StrategyJudge(
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )

    results = []
    for player_slot in iter_player_slots(match):
        history = build_player_history(match, player_slot)
        serialized = serialize_history_for_prompt(player_slot, history)
        prompt = build_prompt(serialized)

        if args.dry_run:
            print(f"--- Prompt for {player_slot} ---")
            print(prompt)
            print("--- end ---\n")
            judgement = None
        else:
            assert judge is not None  # for type checkers
            judgement = judge.classify(prompt)

        results.append(
            {
                "player_slot": player_slot,
                "prompt": prompt if args.dry_run else None,
                "history": serialized,
                "judgement": judgement,
            }
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    elif args.dry_run:
        print("Dry-run complete. Prompts printed above.")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
