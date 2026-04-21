#!/usr/bin/env python3
"""
Run post-hoc LLM judging on CoopEval decision justifications.

This adapter links CoopEval run artifacts (`records.jsonl`, `game_log.txt`) to
the bundled `coopeval.llm_judge` classifier and writes one JSONL row per judged
decision.

Highlights:
- Works across mechanism-specific `records.jsonl` schemas.
- Recovers missing inline responses via `trace_id` lookup in `game_log.txt`.
- Supports resumable runs by skipping already judged `trace_id`s in output.
- Filters defaults are tuned for justification analysis (`CoT`, min chars 80).

Examples:
    python scripts/llm_judge/run_justification_judge.py \
        data/clean/nomechanism1_20260319_153052/no_mechanism_prisoners_dilemma \
        --provider OpenAI \
        --model-name gpt-5.2 \
        --max-workers 50 \
        --output-name gpt5.2

    # Scan only (no API calls), write filtered candidate metadata
    python scripts/llm_judge/run_justification_judge.py \
      data/clean \
      --dry-run
"""

import asyncio
import argparse
import re
import string
import textwrap
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tqdm.asyncio import tqdm

from coopeval.config import JUDGE_OUTPUT_DIR
from coopeval.script_utils.display_helper import (
    detect_agent_type,
    extract_model_name,
    extract_player_id,
    normalize_filter,
)
from coopeval.script_utils.result_loader import iter_action_nodes
from coopeval.utils.json_io import (
    iter_jsonl,
    iter_jsonl_with_line_numbers,
    load_json,
    write_json,
    write_jsonl_record,
)
from inspect_ai.model import get_model
from coopeval.llm_judge.judge import LLMJudge
from coopeval.llm_judge.taxonomy import Taxonomy
from coopeval.llm_judge.config.cooperation_taxonomy import (
    COOPERATION_TAXONOMY,
)

SKIP_SCAN_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "configs",
    "node_modules",
    "slurm",
    "venv",
}

TRACE_BLOCK_RE = re.compile(
    r"===== Prompt \[ID: (?P<trace>[^\]]+)\] \[[^\]]+\] =====\n"
    r"agent: (?P<prompt_agent>[^\n]+)\n"
    r"prompt:\n(?P<prompt>.*?)\n"
    r"===== Response \[ID: (?P=trace)\] \[[^\]]+\] =====\n"
    r"agent: (?P<response_agent>[^\n]+)\n"
    r"response:\n(?P<response>.*?)(?=\n===== Prompt \[ID: |\Z)",
    re.S,
)

BUILTIN_TAXONOMY_LABEL = "builtin:cooperation_taxonomy"
RAW_OUTPUT_SUBDIR = "raw"
RAW_JSON_FILENAME = "judgement.jsonl"
RAW_SUMMARY_FILENAME = "judgement_summary.json"


def utc_now_iso() -> str:
    """Return UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def is_experiment_dir(path: Path) -> bool:
    """Check whether a directory looks like a CoopEval experiment folder."""
    return (
        (path / "config.json").exists()
        and (path / "records.jsonl").exists()
        and (path / "game_log.txt").exists()
    )


def discover_experiment_dirs(input_paths: list[Path]) -> list[Path]:
    """
    Discover experiment folders from one or more roots.

    Supports both:
    - direct experiment paths
    - parent dirs containing nested experiments (e.g., data/clean)
    """
    discovered: dict[str, Path] = {}

    for raw_path in input_paths:
        path = raw_path.resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {path}")

        if is_experiment_dir(path):
            discovered[str(path)] = path
            continue

        for records_path in path.rglob("records.jsonl"):
            run_dir = records_path.parent
            if any(part in SKIP_SCAN_DIRS for part in run_dir.parts):
                continue
            if is_experiment_dir(run_dir):
                resolved = run_dir.resolve()
                discovered[str(resolved)] = resolved

    return sorted(discovered.values())


def split_labels(justification: str) -> list[str]:
    """Parse comma-separated justification labels into a clean list."""
    if not isinstance(justification, str):
        return ["Other"]
    labels = [piece.strip() for piece in justification.split(",")]
    labels = [label for label in labels if label]
    return labels or ["Other"]


def build_output_record(
    *,
    event: dict[str, Any],
    taxonomy_identifier: str,
    save_response_text: bool,
    dry_run: bool,
    judged: dict[str, Any] | None = None,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    judge_input_chars: int | None = None,
) -> dict[str, Any]:
    """Build one output row from extracted event (+ optional judge result)."""
    record = dict(event)
    if not save_response_text:
        record.pop("response_text", None)

    record["processed_at_utc"] = utc_now_iso()
    record["taxonomy_path"] = taxonomy_identifier
    record["dry_run"] = dry_run

    if dry_run:
        record["classification_explanation"] = None
        record["classification_confidence"] = None
        record["classification_justification"] = None
        record["classification_labels"] = []
        return record

    if judged is None:
        judged = {
            "Reasoning_behind_classification": "Error in analysis: empty result",
            "Confidence": 0.0,
            "justification_type": "Failed classification",
        }

    justification = str(judged.get("justification_type", "Other"))
    labels = split_labels(justification)

    confidence_raw = judged.get("Confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0

    record["judge_provider"] = judge_provider
    record["judge_model"] = judge_model
    if judge_input_chars is not None:
        record["judge_input_chars"] = judge_input_chars
    record["classification_explanation"] = str(
        judged.get("Reasoning_behind_classification", "")
    )
    record["classification_confidence"] = confidence
    record["classification_justification"] = justification
    record["classification_labels"] = labels
    return record


def parse_game_log(path: Path) -> dict[str, dict[str, str]]:
    """
    Parse game log into trace_id -> {response_agent, response} mapping.

    This enables recovering the full response text when `records.jsonl` only
    stores `trace_id`.
    """
    content = path.read_text(encoding="utf-8", errors="replace")
    trace_map: dict[str, dict[str, str]] = {}

    for match in TRACE_BLOCK_RE.finditer(content):
        trace_id = match.group("trace")
        trace_map[trace_id] = {
            "response_agent": match.group("response_agent").strip(),
            "response": match.group("response").strip(),
        }

    return trace_map


def load_completed_trace_ids(output_path: Path) -> set[str]:
    """Load already-judged trace IDs from existing output JSONL."""
    completed: set[str] = set()
    if not output_path.exists():
        return completed

    for row in iter_jsonl(output_path, skip_invalid=True):
        trace_id = row.get("trace_id")
        if isinstance(trace_id, str):
            completed.add(trace_id)
    return completed


def build_judge_input(event: dict[str, Any]) -> str:
    """Build text payload sent to the external LLM judge."""
    template = textwrap.dedent("""\
        Game: ${game}
        Model response to classify:
        ${response_text}""")
    return string.Template(template).substitute(
        game=event["game"],
        response_text=event["response_text"],
    )


def should_keep_event(
    event: dict[str, Any],
    args: argparse.Namespace,
    mechanism_filter: set[str] | None,
    game_filter: set[str] | None,
) -> tuple[bool, str | None]:
    """Apply all event-level filters and return (keep, reason_if_dropped)."""
    if mechanism_filter and event["mechanism"].lower() not in mechanism_filter:
        return False, "filter_mechanism"
    if game_filter and event["game"].lower() not in game_filter:
        return False, "filter_game"

    agent_type = event["agent_type"]
    if args.agent_type == "cot" and agent_type != "CoT":
        return False, "filter_agent_type"
    if args.agent_type == "io" and agent_type != "IO":
        return False, "filter_agent_type"

    response_text = event.get("response_text", "")
    if not response_text:
        return False, "missing_response"
    if len(response_text.strip()) < args.min_response_chars:
        return False, "filter_response_length"

    return True, None


def iter_run_events(
    run_dir: Path, counters: Counter[str]
) -> Iterable[dict[str, Any]]:
    """Yield judge-ready action events from one experiment run directory."""
    config = load_json(run_dir / "config.json")
    mechanism = config["mechanism"]["type"]
    game = config["game"]["type"]
    trace_map = parse_game_log(run_dir / "game_log.txt")
    counters["runs_processed"] += 1

    def count_decode_error(_line_no: int, _exc: Exception) -> None:
        counters["records_json_decode_error"] += 1

    records_path = run_dir / "records.jsonl"
    for line_number, payload in iter_jsonl_with_line_numbers(
        records_path,
        skip_invalid=True,
        on_invalid=count_decode_error,
    ):
        for node in iter_action_nodes(payload, require_trace_id=True):
            counters["action_nodes_seen"] += 1

            trace_id = node["trace_id"]
            player = node["player"]
            response_text = trace_map[trace_id]["response"].strip()

            yield {
                "trace_id": trace_id,
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "batch_name": run_dir.parent.name,
                "record_line": line_number,
                "game": game,
                "mechanism": mechanism,
                "player": player,
                "player_id": extract_player_id(player),
                "agent_type": detect_agent_type(player),
                "model": extract_model_name(player),
                "action": str(node.get("action")),
                "points": node.get("points"),
                "mediated": bool(node.get("mediated", False)),
                "response_chars": len(response_text),
                "response_text": response_text,
            }


def summarize_output(output_path: Path) -> dict[str, Any]:
    """Build high-level summary statistics from output JSONL."""
    summary: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "output_file": str(output_path.resolve()),
    }
    if not output_path.exists():
        summary["rows"] = 0
        summary["unique_trace_ids"] = 0
        summary["mean_confidence"] = None
        summary["label_counts"] = {}
        summary["mechanism_counts"] = {}
        summary["game_counts"] = {}
        summary["model_counts"] = {}
        return summary

    label_counts: Counter[str] = Counter()
    mechanism_counts: Counter[str] = Counter()
    game_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    unique_trace_ids: set[str] = set()
    confidence_sum = 0.0
    confidence_n = 0
    rows = 0

    for row in iter_jsonl(output_path, skip_invalid=True):
        rows += 1

        trace_id = row.get("trace_id")
        if isinstance(trace_id, str):
            unique_trace_ids.add(trace_id)

        mechanism = row.get("mechanism")
        if isinstance(mechanism, str):
            mechanism_counts[mechanism] += 1

        game = row.get("game")
        if isinstance(game, str):
            game_counts[game] += 1

        model = row.get("model")
        if isinstance(model, str):
            model_counts[model] += 1

        confidence = row.get("classification_confidence")
        if isinstance(confidence, (int, float)):
            confidence_sum += float(confidence)
            confidence_n += 1

        labels = row.get("classification_labels")
        if isinstance(labels, list):
            for label in labels:
                if isinstance(label, str) and label.strip():
                    label_counts[label.strip()] += 1
        else:
            raw = row.get("classification_justification", "Other")
            if isinstance(raw, str):
                for label in split_labels(raw):
                    label_counts[label] += 1

    summary["rows"] = rows
    summary["unique_trace_ids"] = len(unique_trace_ids)
    summary["mean_confidence"] = (
        confidence_sum / confidence_n if confidence_n else None
    )
    summary["label_counts"] = dict(label_counts.most_common())
    summary["mechanism_counts"] = dict(mechanism_counts.most_common())
    summary["game_counts"] = dict(game_counts.most_common())
    summary["model_counts"] = dict(model_counts.most_common())
    return summary


def _slugify(value: str, fallback: str) -> str:
    """Convert string to filesystem-friendly slug, with fallback if empty."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    slug = slug.strip("-_.") or fallback
    return slug


def build_output_paths(output_name: str) -> tuple[Path, Path]:
    """Build output file paths based on output name, ensuring no directory traversal."""
    candidate = output_name.strip()
    if not candidate:
        raise ValueError("--output-name cannot be empty.")
    path = Path(candidate)
    if path.name != candidate:
        raise ValueError("--output-name must not include directory separators.")
    stem = path.stem or "run"
    slug = _slugify(stem, "run")
    raw_dir = JUDGE_OUTPUT_DIR / slug / RAW_OUTPUT_SUBDIR
    raw_jsonl = (raw_dir / RAW_JSON_FILENAME).resolve()
    summary_json = (raw_dir / RAW_SUMMARY_FILENAME).resolve()
    return raw_jsonl, summary_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Adapter: apply llm_judge classifications to CoopEval "
            "decision justifications."
        )
    )
    parser.add_argument(
        "input_paths",
        type=Path,
        nargs="+",
        help=(
            "Experiment directory or parent directory/directories containing "
            "CoopEval runs (e.g., data/clean/run3 or data/clean)."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["OpenAI", "OpenRouter"],
        default="openrouter",
        help="Judge API provider.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Provider model/deployment override (e.g., openrouter model name)."
        ),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help=(
            "Slug used to name the run directory under outputs/judge/. "
            "Raw files are written to outputs/judge/<output-name>/raw/"
            "(judgement.jsonl + judgement_summary.json)."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge model temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Max completion tokens for judge responses.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file instead of resumable append.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call judge API; only extract filtered candidate rows.",
    )
    parser.add_argument(
        "--agent-type",
        choices=["cot", "io", "all"],
        default="cot",
        help="Filter by agent type. Default is cot for justification quality.",
    )
    parser.add_argument(
        "--min-response-chars",
        type=int,
        default=80,
        help="Minimum response length after trimming.",
    )
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        default=None,
        help="Optional mechanism allowlist (case-insensitive).",
    )
    parser.add_argument(
        "--games",
        nargs="+",
        default=None,
        help="Optional game allowlist (case-insensitive).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on number of newly written rows.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=20,
        help="Flush output every N written rows.",
    )
    parser.add_argument(
        "--save-response-text",
        action="store_true",
        help="Persist full response text into output rows.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel judge workers for API calls. "
            "Set >1 to enable concurrent judging (default: 1)."
        ),
    )
    args = parser.parse_args()
    args.input_paths = [
        path.expanduser().resolve() for path in args.input_paths
    ]
    return args


def main():
    """Coordinate filtering, judging, and output writing for CoopEval runs."""
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

    output_path, summary_path = build_output_paths(args.output_name)
    raw_dir = output_path.parent
    print(f"Writing judged decisions to {output_path}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_experiment_dirs(args.input_paths)
    if not run_dirs:
        raise RuntimeError(
            "No experiment folders found. Expected directories containing "
            "config.json, records.jsonl, and game_log.txt."
        )

    print(f"Discovered {len(run_dirs)} experiment folder(s).")

    mechanism_filter = normalize_filter(args.mechanisms)
    game_filter = normalize_filter(args.games)

    taxonomy_label = BUILTIN_TAXONOMY_LABEL

    print(f"Taxonomy: {taxonomy_label}")

    completed_trace_ids: set[str] = set()
    if output_path.exists() and not args.overwrite:
        completed_trace_ids = load_completed_trace_ids(output_path)
        print(
            f"Loaded {len(completed_trace_ids)} completed trace IDs "
            f"from existing output for resume."
        )

    file_mode = "w" if args.overwrite else "a"

    judge_instance = None
    judge_provider = None
    judge_model = None
    if not args.dry_run:
        taxonomy_spec = {"data": COOPERATION_TAXONOMY}

        api_client = get_model(f"{args.provider.lower()}/{args.model_name}")
        if "path" in taxonomy_spec:
            taxonomy = Taxonomy.from_json_file(taxonomy_spec["path"])
        else:
            taxonomy = Taxonomy.from_dict(deepcopy(taxonomy_spec["data"]))
        judge_instance = LLMJudge(
            api_client=api_client,
            taxonomy=taxonomy,
            temperature=args.temperature,
        )
        judge_provider = args.provider.lower()
        judge_model = getattr(api_client, "model_name", None)

    counters: Counter[str] = Counter()
    seen_trace_ids: set[str] = set(completed_trace_ids)

    async def classify_event(
        event: dict[str, Any],
        semaphore: asyncio.Semaphore,
        out_file: Any,
        pbar: tqdm,
    ) -> None:
        """Classify one event and write it to the output file."""
        judge_input = build_judge_input(event)
        try:
            async with semaphore:
                judged = await judge_instance.classify_text_async(
                    judge_input,
                    max_tokens=args.max_tokens,
                )
            judge_input_chars = len(judge_input)
            record = build_output_record(
                event=event,
                taxonomy_identifier=taxonomy_label,
                save_response_text=args.save_response_text,
                dry_run=False,
                judged=judged,
                judge_provider=judge_provider,
                judge_model=judge_model,
                judge_input_chars=judge_input_chars,
            )
            counters["judged_rows"] += 1
        except Exception as e:
            counters["judge_errors"] += 1
            judged = {
                "Reasoning_behind_classification": f"Error in analysis: {e}",
                "Confidence": 0.0,
                "justification_type": "Failed classification",
            }
            record = build_output_record(
                event=event,
                taxonomy_identifier=taxonomy_label,
                save_response_text=args.save_response_text,
                dry_run=False,
                judged=judged,
                judge_provider=judge_provider,
                judge_model=judge_model,
                judge_input_chars=len(judge_input),
            )

        write_jsonl_record(out_file, record)
        pbar.update(1)
        if counters["judged_rows"] % args.flush_every == 0:
            out_file.flush()

    def iter_kept_events() -> Iterable[dict[str, Any]]:
        """Yield filtered events, applying resume and max-item limits."""
        selected_rows = 0
        for run_dir in run_dirs:
            for event in iter_run_events(run_dir, counters):
                trace_id = event["trace_id"]
                if trace_id in seen_trace_ids:
                    counters["skip_duplicate_or_resume"] += 1
                    continue

                keep, drop_reason = should_keep_event(
                    event=event,
                    args=args,
                    mechanism_filter=mechanism_filter,
                    game_filter=game_filter,
                )
                if not keep:
                    if drop_reason is not None:
                        counters[drop_reason] += 1
                    continue

                if (
                    args.max_items is not None
                    and selected_rows >= args.max_items
                ):
                    return

                seen_trace_ids.add(trace_id)
                selected_rows += 1
                if not args.dry_run:
                    counters["submitted_for_judging"] += 1
                yield event

    async def run_async():
        with open(output_path, file_mode, encoding="utf-8") as out_file:
            if args.dry_run:
                with tqdm(desc="Writing rows", unit="row") as pbar:
                    for event in iter_kept_events():
                        record = build_output_record(
                            event=event,
                            taxonomy_identifier=taxonomy_label,
                            save_response_text=args.save_response_text,
                            dry_run=True,
                        )
                        counters["dry_run_rows"] += 1
                        write_jsonl_record(out_file, record)
                        pbar.update(1)
            else:
                events = list(iter_kept_events())
                semaphore = asyncio.Semaphore(args.max_workers)
                with tqdm(total=len(events), desc="Judging", unit="row") as pbar:
                    tasks = [
                        classify_event(event, semaphore, out_file, pbar)
                        for event in events
                    ]
                    await asyncio.gather(*tasks)
            out_file.flush()

    asyncio.run(run_async())

    summary = summarize_output(output_path)
    summary["taxonomy_source"] = taxonomy_label
    summary["discovered_runs"] = len(run_dirs)
    summary["processing_counters"] = dict(counters)
    summary["max_items"] = args.max_items
    summary["agent_type_filter"] = args.agent_type
    summary["min_response_chars"] = args.min_response_chars
    summary["dry_run"] = args.dry_run
    summary["max_workers"] = args.max_workers

    write_json(summary_path, summary)

    print(f"\nWrote output rows: {counters['judged_rows'] or counters['dry_run_rows']}")
    print(f"Output JSONL: {output_path}")
    print(f"Summary JSON: {summary_path}")
    print("Counters:")
    for key, value in sorted(counters.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
