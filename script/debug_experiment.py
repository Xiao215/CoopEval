#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Specify Configs and simulate command line arguments
from script.run_experiment import set_seed, main
CONFIG = "main/mp_mediation.yaml"
# CONFIG = "main/pg_reputation_test.yaml"
sys.argv = ["script/run_experiment.py", "--config", CONFIG]

# ============ Legacy Configs for reference ============
# from script.run_evolution import set_seed, main
# CONFIG = "legacy/tg_testing_openrouter.yaml"
# sys.argv = ["run_evolution.py", "--config", CONFIG]


# Run the main script
if __name__ == "__main__":
    set_seed()
    main()
