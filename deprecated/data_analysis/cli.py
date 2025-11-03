from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import analyze
from .report import build_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Tournament Data Analysis")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs"),
        help="Directory containing run outputs (defaults to 'outputs')",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/analysis"),
        help="Destination directory for analysis artefacts",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip markdown report generation",
    )
    parser.add_argument(
        "--disarmament",
        action="store_true",
        help="Generate additional disarmament-specific metrics and plots",
    )

    args = parser.parse_args()

    outputs = analyze(args.root, args.out, disarmament=args.disarmament)
    if not args.no_report:
        report_path = build_markdown_report(args.out)
        outputs["markdown_report"] = report_path

    print("Analysis artefacts written:")
    for name, path in outputs.items():
        print(f" - {name}: {path}")


if __name__ == "__main__":
    main()
