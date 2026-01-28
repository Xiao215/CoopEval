#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Specify Configs and simulate command line arguments
from src.visualize.create_table import main

batch_paths = ["data/clean/all_but_rep_and_contr1_20260119_205910", "data/clean/contracting1_20260121_172329", "data/clean/repetition1_20260121_013429", "data/clean/reputationfirstorder1_20260125_182942", "data/clean/run2_20260124_025842", "data/clean/run3_20260125_095939"]
output = "figures/allruns/tables_with_stderr/"
metrics = ["mean", "rd"]

sys.argv = ["src/visualize/create_table.py", *batch_paths, "--output", output, "--metrics", *metrics]

# Run the main script
if __name__ == "__main__":
    main()
