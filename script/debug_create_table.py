#!/usr/bin/env python3
"""
Convenience entry point for exercising src.visualize.create_table without
typing long CLI invocations. Adjust the hard-coded paths when reproducing
other batches locally.
"""
import sys
from pathlib import Path

# Make src/ imports work when running the script directly from the repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualize.create_table import main

batch_paths = [
    "data/clean/all_but_rep_and_contr1_20260119_205910",
    "data/clean/contracting1_20260121_172329",
    "data/clean/repetition1_20260121_013429",
    "data/clean/reputationfirstorder1_20260125_182942",
    "data/clean/run2_20260124_025842",
    "data/clean/run3_20260125_095939",
]
output = "figures/allruns/tables_with_stderr/"
metrics = ["mean", "rd"]

# Pretend we called create_table.py from the command line so main() can reuse the argv contract
sys.argv = ["src/visualize/create_table.py", *batch_paths, "--output", output, "--metrics", *metrics]

if __name__ == "__main__":
    main()
