#!/usr/bin/env python3
"""Export canonical taxonomy dataset (counts + share) from normalized JSONL."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from coopeval.config import JUDGE_OUTPUT_DIR
from coopeval.script_utils.llm_judge_helpers import classification_labels, pct
from coopeval.utils.json_io import iter_jsonl

NORMALIZED_SUBDIR = "normalized"
DATASET_SUBDIR = "dataset"
DATASET_COUNTS = "taxonomy_by_game_mechanism_model_player.counts.csv"
DATASET_SHARES = "taxonomy_by_game_mechanism_model_player.share_pct.csv"


def parse_args() -> argparse.Namespace:
    """Return parsed CLI args for exporting the taxonomy dataset."""
    parser = argparse.ArgumentParser(
        description=(
            "Build canonical taxonomy datasets (counts + share) from normalized JSONL."
        )
    )
    parser.add_argument(
        "input_name",
        type=str,
        help="Judge run identifier (same as normalize_justification_labels input).",
    )
    return parser.parse_args()


def export_dataset(normalized_jsonl: Path, output_dir: Path) -> None:
    """Write canonical counts + share CSVs from normalized judge rows."""
    grouped_counts: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(
        Counter
    )
    row_counts: Counter[tuple[str, str, str, str]] = Counter()
    labels_seen: set[str] = set()

    for row in iter_jsonl(normalized_jsonl):
        labels = classification_labels(row)
        if not labels:
            continue
        game = str(row.get("game", "Unknown"))
        mechanism = str(row.get("mechanism", "Unknown"))
        model = str(row.get("model", "Unknown"))
        player = str(row.get("player", "Unknown"))
        key = (game, mechanism, model, player)
        row_counts[key] += 1
        for label in labels:
            grouped_counts[key][label] += 1
            labels_seen.add(label)

    if not grouped_counts:
        raise RuntimeError("No labeled rows found in normalized JSONL.")

    counts_path = output_dir / DATASET_COUNTS
    shares_path = output_dir / DATASET_SHARES

    with (
        counts_path.open("w", encoding="utf-8", newline="") as f_counts,
        shares_path.open("w", encoding="utf-8", newline="") as f_shares,
    ):
        counts_writer = csv.writer(f_counts)
        counts_writer.writerow(
            ["game", "mechanism", "model", "player", "label", "count"]
        )
        shares_writer = csv.writer(f_shares)
        shares_writer.writerow(
            [
                "game",
                "mechanism",
                "model",
                "player",
                "label",
                "share_pct",
                "group_count",
            ]
        )

        for key in sorted(grouped_counts):
            game, mechanism, model, player = key
            denom = row_counts[key]
            for label, count in grouped_counts[key].most_common():
                counts_writer.writerow(
                    [game, mechanism, model, player, label, count]
                )
                share_value = pct(count, denom)
                shares_writer.writerow(
                    [
                        game,
                        mechanism,
                        model,
                        player,
                        label,
                        f"{share_value:.4f}",
                        denom,
                    ]
                )

    print(
        f"Wrote {counts_path.name} and {shares_path.name} "
        f"(labels={len(labels_seen)}, groups={len(grouped_counts)})"
    )


def main() -> None:
    """CLI entry point for exporting taxonomy datasets."""
    args = parse_args()
    input_name = args.input_name.strip()
    if not input_name:
        raise ValueError("input_name cannot be empty.")
    if Path(input_name).name != input_name:
        raise ValueError("input_name must not include directory separators.")

    run_dir = (JUDGE_OUTPUT_DIR / input_name).resolve()
    normalized_dir = run_dir / NORMALIZED_SUBDIR
    normalized_jsonl = (normalized_dir / "normalized.jsonl").resolve()
    if not normalized_jsonl.exists():
        raise FileNotFoundError(f"Missing normalized JSONL: {normalized_jsonl}")

    output_dir = run_dir / DATASET_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dataset(normalized_jsonl, output_dir)


if __name__ == "__main__":
    main()
