#!/usr/bin/env python3
"""
Small harness for exercising script.run_experiment with a curated config file.
It rewrites sys.argv so we can mimic CLI usage directly from an IDE/debugger.
"""
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from script.run_experiment import main, set_seed

CONFIG = "main/pd_testing.yaml"
# Uncomment to smoke-test reputations locally
# CONFIG = "main/pg_reputation_test.yaml"

# Reuse the same interface as the CLI entry point
sys.argv = ["script/run_experiment.py", "--config", CONFIG]

if __name__ == "__main__":
    set_seed()
    main()
