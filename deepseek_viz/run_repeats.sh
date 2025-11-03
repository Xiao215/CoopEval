#!/usr/bin/env bash
# Run multiple repeats of deepseek-vs-deepseek for the probabilistic
# (mixed-strategy) and direct-action Prisoner's Dilemma experiments.
#
# The logger timestamps directories to the nearest minute, so we sleep between
# runs by default to keep outputs separated. Override SLEEP_BETWEEN_RUNS=0 to
# disable the delay.

set -euo pipefail

SLEEP_BETWEEN_RUNS=${SLEEP_BETWEEN_RUNS:-65}

prob_config="pd_repetition_deepseek_self.yaml"
direct_config="pd_repetition_direct_deepseek.yaml"

echo "Running probabilistic (mixed distribution) repeats..."
for run in 1 2 3 4 5; do
    echo "  ▶ Run ${run} using ${prob_config}"
    python -m script.run_evolution --config "${prob_config}"
    if [[ ${run} -lt 5 && ${SLEEP_BETWEEN_RUNS} -gt 0 ]]; then
        sleep "${SLEEP_BETWEEN_RUNS}"
    fi
done

echo "Running direct-action repeats..."
for run in 1 2 3 4 5; do
    echo "  ▶ Run ${run} using ${direct_config}"
    python -m script.run_evolution --config "${direct_config}"
    if [[ ${run} -lt 5 && ${SLEEP_BETWEEN_RUNS} -gt 0 ]]; then
        sleep "${SLEEP_BETWEEN_RUNS}"
    fi
done

echo "All repeats completed."
