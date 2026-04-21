#!/usr/bin/env python3
"""
Normalize LLM-judge justification labels and generate cleaned summaries.

This script is intended for outputs produced by:
    scripts/llm_judge/run_justification_judge.py

It:
1. Normalizes noisy/variant labels into a canonical taxonomy.
2. Writes a cleaned JSONL with normalized labels.
3. Writes summary JSON and CSV tables (overall + by mechanism/game/model).
"""

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from coopeval.config import JUDGE_OUTPUT_DIR
from coopeval.script_utils.result_loader import (
    DEFAULT_SKIP_GAMES,
    should_skip_game_name,
)
from coopeval.utils.json_io import iter_jsonl, write_json, write_jsonl_record

RAW_JUDGEMENT_FILENAME = "judgement.jsonl"
NORMALIZED_SUBDIR = "normalized"
NORMALIZED_JSON_FILENAME = "normalized.jsonl"
NORMALIZED_SUMMARY_FILENAME = "normalized.summary.json"
NORMALIZED_LABEL_MAP_FILENAME = "normalized.label_map.json"
NORMALIZED_OVERALL_CSV = "normalized.counts_overall.csv"
NORMALIZED_MECH_CSV = "normalized.counts_by_mechanism.csv"
NORMALIZED_GAME_CSV = "normalized.counts_by_game.csv"
NORMALIZED_MODEL_CSV = "normalized.counts_by_model.csv"
CANONICAL_LABELS: list[str] = [
    "Individual utility maximization",
    "Strategic equilibrium focus",
    "Social welfare maximization",
    "Inequity aversion",
    "Reciprocity",
    "Strategic influence",
    "Trust evaluation",
    "Competitiveness",
    "Uncertainty evaluation",
    "Social norm conformity",
    "Rule misunderstanding",
    "Exploration-exploitation trade-off",
    "Risk aversion",
    "Strategy legibility",
    "Multidimensional reasoning",
    "Other",
]


def _norm_key(text: str) -> str:
    """Normalize free-form label text to a stable lookup key."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


CANONICAL_LOOKUP: dict[str, str] = {
    _norm_key(label): label for label in CANONICAL_LABELS
}


# Many-to-many alias mapping (some noisy labels map to multiple canonical ones).
ALIAS_TO_CANONICAL: dict[str, list[str]] = {
    _norm_key("Category1"): ["Individual utility maximization"],
    _norm_key("Category2"): ["Strategic equilibrium focus"],
    _norm_key("Category3"): ["Social welfare maximization"],
    _norm_key("Category4"): ["Inequity aversion"],
    _norm_key("Category5"): ["Reciprocity"],
    _norm_key("Category6"): ["Strategic influence"],
    _norm_key("Category7"): ["Trust evaluation"],
    _norm_key("Category8"): ["Competitiveness"],
    _norm_key("Category9"): ["Uncertainty evaluation"],
    _norm_key("Category10"): ["Social norm conformity"],
    _norm_key("Category11"): ["Rule misunderstanding"],
    _norm_key("Category12"): ["Exploration-exploitation trade-off"],
    _norm_key("Category13"): ["Risk aversion"],
    _norm_key("Category14"): ["Strategy legibility"],
    _norm_key("Category15"): ["Multidimensional reasoning"],
    _norm_key("1"): ["Individual utility maximization"],
    _norm_key("2"): ["Strategic equilibrium focus"],
    _norm_key("3"): ["Social welfare maximization"],
    _norm_key("4"): ["Inequity aversion"],
    _norm_key("5"): ["Reciprocity"],
    _norm_key("6"): ["Strategic influence"],
    _norm_key("7"): ["Trust evaluation"],
    _norm_key("8"): ["Competitiveness"],
    _norm_key("9"): ["Uncertainty evaluation"],
    _norm_key("10"): ["Social norm conformity"],
    _norm_key("11"): ["Rule misunderstanding"],
    _norm_key("12"): ["Exploration-exploitation trade-off"],
    _norm_key("13"): ["Risk aversion"],
    _norm_key("14"): ["Strategy legibility"],
    _norm_key("15"): ["Multidimensional reasoning"],
}


def parse_raw_labels(row: dict[str, Any]) -> list[str]:
    """Extract raw labels from row, supporting either list or string field."""
    labels = row.get("classification_labels")
    if isinstance(labels, list):
        clean = []
        for label in labels:
            if isinstance(label, str) and label.strip():
                clean.append(label.strip())
        if clean:
            return clean

    justification = row.get("classification_justification")
    if isinstance(justification, str):
        return [p.strip() for p in justification.split(",") if p.strip()]

    return []


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    """Deduplicate labels while preserving first-seen order."""
    out: list[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def normalize_labels(raw_labels: Sequence[str]) -> tuple[list[str], list[str]]:
    """
    Normalize raw labels into canonical taxonomy.

    Returns:
        (normalized_labels, unmapped_raw_labels)
    """
    normalized: list[str] = []
    unmapped: list[str] = []

    for raw in raw_labels:
        key = _norm_key(raw)
        if key in CANONICAL_LOOKUP:
            normalized.append(CANONICAL_LOOKUP[key])
            continue

        mapped = ALIAS_TO_CANONICAL.get(key)
        if mapped:
            normalized.extend(mapped)
        else:
            normalized.append("Other")
            unmapped.append(raw)

    normalized = unique_preserve_order(normalized)
    return normalized, unmapped


def write_group_csv(
    path: Path,
    grouped_counts: dict[str, Counter[str]],
    row_totals: dict[str, int],
) -> None:
    """Write group-level label counts and shares to CSV."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "label", "count", "share_of_group_pct"])
        for group in sorted(grouped_counts):
            total = row_totals.get(group, 0)
            counts = grouped_counts[group]
            for label, count in counts.most_common():
                share = 0.0 if total == 0 else 100.0 * count / total
                writer.writerow([group, label, count, f"{share:.2f}"])


def build_normalized_paths(normalized_dir: Path) -> dict[str, Path]:
    """Return dict of normalized artifact paths inside normalized_dir."""
    return {
        "jsonl": normalized_dir / NORMALIZED_JSON_FILENAME,
        "summary_json": normalized_dir / NORMALIZED_SUMMARY_FILENAME,
        "label_map_json": normalized_dir / NORMALIZED_LABEL_MAP_FILENAME,
        "overall_csv": normalized_dir / NORMALIZED_OVERALL_CSV,
        "mech_csv": normalized_dir / NORMALIZED_MECH_CSV,
        "game_csv": normalized_dir / NORMALIZED_GAME_CSV,
        "model_csv": normalized_dir / NORMALIZED_MODEL_CSV,
    }


def parse_args() -> argparse.Namespace:
    """Return parsed CLI args for label normalization."""
    parser = argparse.ArgumentParser(
        description=(
            "Normalize justification labels from run_justification_judge output "
            "and generate cleaned summaries."
        )
    )
    parser.add_argument(
        "input_name",
        type=str,
        help=(
            "Judge run identifier (same as --output-name for "
            "run_justification_judge)."
        ),
    )
    parser.add_argument(
        "--skip-games",
        nargs="*",
        default=DEFAULT_SKIP_GAMES,
        help=(
            "Games to drop before aggregation "
            "(default: %(default)s; pass with no values to keep all games)."
        ),
    )
    return parser.parse_args()


def validate_input_name(value: str) -> str:
    """Return a safe judge run identifier."""
    input_name = value.strip()
    if not input_name:
        raise ValueError("input_name cannot be empty.")
    if Path(input_name).name != input_name:
        raise ValueError("input_name must not include directory separators.")
    return input_name


def resolve_raw_judgement_path(input_name: str) -> Path:
    """Return the raw judge JSONL path for an input run."""
    run_dir = (JUDGE_OUTPUT_DIR / input_name).resolve()
    raw_dir = run_dir / "raw"
    input_path = (raw_dir / RAW_JUDGEMENT_FILENAME).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Judge JSONL not found: {input_path}")
    return input_path


def normalize_rows(
    input_path: Path,
    output_jsonl: Path,
    skip_games: Sequence[str] | None,
) -> dict[str, Any]:
    """Normalize judge rows, write JSONL output, and return aggregate stats."""
    rows = 0
    unique_trace_ids = set()
    confidence_sum = 0.0
    confidence_n = 0

    raw_label_counts: Counter[str] = Counter()
    normalized_label_counts: Counter[str] = Counter()
    unmapped_raw_counts: Counter[str] = Counter()

    mechanism_row_counts: Counter[str] = Counter()
    game_row_counts: Counter[str] = Counter()
    model_row_counts: Counter[str] = Counter()

    by_mechanism_counts: dict[str, Counter[str]] = defaultdict(Counter)
    by_game_counts: dict[str, Counter[str]] = defaultdict(Counter)
    by_model_counts: dict[str, Counter[str]] = defaultdict(Counter)

    unmapped_rows = 0
    total_label_instances_raw = 0
    total_label_instances_normalized = 0

    with output_jsonl.open("w", encoding="utf-8") as f_out:
        for row in iter_jsonl(input_path):
            trace_id = row.get("trace_id")
            if isinstance(trace_id, str):
                unique_trace_ids.add(trace_id)

            confidence = row.get("classification_confidence")
            if isinstance(confidence, (int, float)):
                confidence_sum += float(confidence)
                confidence_n += 1

            mechanism = str(row.get("mechanism", "Unknown"))
            game = str(row.get("game", "Unknown"))
            if should_skip_game_name(game, skip_games):
                continue
            rows += 1
            model = str(row.get("model", "Unknown"))
            mechanism_row_counts[mechanism] += 1
            game_row_counts[game] += 1
            model_row_counts[model] += 1

            raw_labels = parse_raw_labels(row)
            total_label_instances_raw += len(raw_labels)
            for label in raw_labels:
                raw_label_counts[label] += 1

            normalized_labels, unmapped = normalize_labels(raw_labels)
            total_label_instances_normalized += len(normalized_labels)

            if unmapped:
                unmapped_rows += 1
                for label in unmapped:
                    unmapped_raw_counts[label] += 1

            for label in normalized_labels:
                normalized_label_counts[label] += 1
                by_mechanism_counts[mechanism][label] += 1
                by_game_counts[game][label] += 1
                by_model_counts[model][label] += 1

            row["classification_labels_raw"] = raw_labels
            row["classification_labels_normalized"] = normalized_labels
            row["classification_unmapped_labels"] = unmapped
            row["classification_has_unmapped"] = bool(unmapped)
            row["classification_justification_normalized"] = ", ".join(
                normalized_labels
            )

            write_jsonl_record(f_out, row)

    return {
        "rows": rows,
        "unique_trace_ids": unique_trace_ids,
        "confidence_sum": confidence_sum,
        "confidence_n": confidence_n,
        "raw_label_counts": raw_label_counts,
        "normalized_label_counts": normalized_label_counts,
        "unmapped_raw_counts": unmapped_raw_counts,
        "mechanism_row_counts": mechanism_row_counts,
        "game_row_counts": game_row_counts,
        "model_row_counts": model_row_counts,
        "by_mechanism_counts": by_mechanism_counts,
        "by_game_counts": by_game_counts,
        "by_model_counts": by_model_counts,
        "unmapped_rows": unmapped_rows,
        "total_label_instances_raw": total_label_instances_raw,
        "total_label_instances_normalized": total_label_instances_normalized,
    }


def build_summary(
    input_path: Path,
    output_jsonl: Path,
    output_summary_json: Path,
    output_map_json: Path,
    out_paths: dict[str, Path],
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Build the normalized output summary payload."""
    confidence_n = stats["confidence_n"]
    return {
        "input_file": str(input_path),
        "normalized_output_file": str(output_jsonl),
        "rows": stats["rows"],
        "unique_trace_ids": len(stats["unique_trace_ids"]),
        "mean_confidence": (
            stats["confidence_sum"] / confidence_n if confidence_n else None
        ),
        "raw_label_counts": dict(stats["raw_label_counts"].most_common()),
        "normalized_label_counts": dict(
            stats["normalized_label_counts"].most_common()
        ),
        "mechanism_row_counts": dict(
            stats["mechanism_row_counts"].most_common()
        ),
        "game_row_counts": dict(stats["game_row_counts"].most_common()),
        "model_row_counts": dict(stats["model_row_counts"].most_common()),
        "mapping_stats": {
            "raw_unique_labels": len(stats["raw_label_counts"]),
            "normalized_unique_labels": len(stats["normalized_label_counts"]),
            "total_label_instances_raw": stats["total_label_instances_raw"],
            "total_label_instances_normalized": stats[
                "total_label_instances_normalized"
            ],
            "unmapped_rows": stats["unmapped_rows"],
            "unmapped_raw_label_counts": dict(
                stats["unmapped_raw_counts"].most_common()
            ),
        },
        "output_files": {
            "normalized_jsonl": str(output_jsonl),
            "summary_json": str(output_summary_json),
            "label_map_json": str(output_map_json),
            "counts_overall_csv": str(out_paths["overall_csv"]),
            "counts_by_mechanism_csv": str(out_paths["mech_csv"]),
            "counts_by_game_csv": str(out_paths["game_csv"]),
            "counts_by_model_csv": str(out_paths["model_csv"]),
        },
    }


def main() -> None:
    args = parse_args()
    input_name = validate_input_name(args.input_name)
    input_path = resolve_raw_judgement_path(input_name)

    run_dir = (JUDGE_OUTPUT_DIR / input_name).resolve()
    normalized_dir = run_dir / NORMALIZED_SUBDIR
    normalized_dir.mkdir(parents=True, exist_ok=True)

    out_paths = build_normalized_paths(normalized_dir)
    output_jsonl = out_paths["jsonl"]
    output_summary_json = out_paths["summary_json"]
    output_map_json = out_paths["label_map_json"]
    output_overall_csv = out_paths["overall_csv"]
    output_mech_csv = out_paths["mech_csv"]
    output_game_csv = out_paths["game_csv"]
    output_model_csv = out_paths["model_csv"]

    skip_games = tuple(args.skip_games)

    stats = normalize_rows(input_path, output_jsonl, skip_games)

    write_group_csv(
        output_overall_csv,
        grouped_counts={"ALL": stats["normalized_label_counts"]},
        row_totals={"ALL": stats["rows"]},
    )
    write_group_csv(
        output_mech_csv,
        grouped_counts=stats["by_mechanism_counts"],
        row_totals=dict(stats["mechanism_row_counts"]),
    )
    write_group_csv(
        output_game_csv,
        grouped_counts=stats["by_game_counts"],
        row_totals=dict(stats["game_row_counts"]),
    )
    write_group_csv(
        output_model_csv,
        grouped_counts=stats["by_model_counts"],
        row_totals=dict(stats["model_row_counts"]),
    )

    label_map_payload = {
        "canonical_labels": CANONICAL_LABELS,
        "alias_to_canonical": ALIAS_TO_CANONICAL,
    }
    write_json(output_map_json, label_map_payload)

    summary = build_summary(
        input_path,
        output_jsonl,
        output_summary_json,
        output_map_json,
        out_paths,
        stats,
    )
    write_json(output_summary_json, summary)

    print(f"Input rows: {stats['rows']}")
    print(f"Normalized output: {output_jsonl}")
    print(f"Summary: {output_summary_json}")
    print(
        "Top normalized labels: "
        f"{stats['normalized_label_counts'].most_common(10)}"
    )
    print(
        "Unmapped raw labels: "
        f"{dict(stats['unmapped_raw_counts'].most_common())}"
    )


if __name__ == "__main__":
    main()
