#!/bin/bash
# Main runner script to execute run_experiment.py for all combinations of games and mechanisms
# Usage: ./main_runner.sh

export PYTHONPATH=.

# Don't exit on error - we want to continue even if individual experiments fail
set +e

# Trap Ctrl+C and cleanup
trap 'echo ""; echo "Interrupted! Batch summary saved to: ${BATCH_DIR}/batch_summary.json"; exit 130' INT

# =============================================================================
# CONFIGURATION - Edit these variables to customize your experiments
# =============================================================================

# Agents configuration (relative to configs/)
# AGENTS_CONFIG="agents/test_agents_6.yaml"
# AGENTS_CONFIG="agents/cheap_llms_3.yaml"
# # AGENTS_CONFIG="agents/sota_llms.yaml"
AGENTS_CONFIG="agents/few_strong_llms.yaml"

# Evaluation configuration (relative to configs/)
# EVALUATION_CONFIG="evaluation/default_evaluation.yaml"
EVALUATION_CONFIG="evaluation/no_deviation_ratings.yaml"

# Parallel execution settings
PARALLEL_EXPERIMENTS=4  # Number of experiments to run simultaneously
EXPERIMENT_WORKERS=2    # Number of parallel workers within each experiment (for LLM queries)

# Retry settings
RETRY_FAILED_EXPERIMENTS=true  # Set to false to skip failed experiments instead of retrying them

# Batch directory - set to existing path to resume, or leave empty for new batch with timestamp
# RESUME_BATCH_DIR="outputs/2026/01/12/01:57"
RESUME_BATCH_DIR=""

# List of game config paths (relative to configs/)
# Based on games in src/games/
GAME_CONFIGS=(
    # "games/matching_pennies.yaml"
    # "games/prisoners_dilemma.yaml"
    "games/public_goods.yaml"
    # "games/stag_hunt.yaml"
    # "games/travellers_dilemma.yaml"
    # "games/trust_game.yaml"
)

# List of mechanism config paths (relative to configs/)
# Based on mechanisms in src/mechanisms/
MECHANISM_CONFIGS=(
    # "mechanisms/no_mechanism.yaml"
    # "mechanisms/contracting.yaml"
    # "mechanisms/disarmament.yaml"
    # "mechanisms/mediation.yaml"
    "mechanisms/repetition.yaml"
    # "mechanisms/reputation.yaml"
)

# GAME_CONFIGS=("games/matching_pennies.yaml" "games/prisoners_dilemma.yaml")
# MECHANISM_CONFIGS=("mechanisms/no_mechanism.yaml" "mechanisms/repetition.yaml")

# =============================================================================
# END CONFIGURATION
# =============================================================================

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Activate conda environment
echo "Activating conda environment: llmcoop"

# Try to initialize conda if not already initialized
if [ -z "$CONDA_EXE" ]; then
    # Find conda installation
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        eval "$(conda shell.bash hook)" 2>/dev/null || true
    fi
fi

conda activate llmcoop

# Store the python path to use consistently throughout the script
# First try to get python from conda environment, fall back to which
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN="$(which python3)"
fi

# Verify activation
echo "Python path: $PYTHON_BIN"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Conda prefix: $CONDA_PREFIX"

# Counter for tracking experiments
total_experiments=$((${#GAME_CONFIGS[@]} * ${#MECHANISM_CONFIGS[@]}))
current=0

echo "=================================================="
echo "Running $total_experiments experiments"
echo "Games: ${#GAME_CONFIGS[@]}, Mechanisms: ${#MECHANISM_CONFIGS[@]}"
echo "=================================================="
echo ""

# =============================================================================
# BATCH SETUP
# =============================================================================

# Set up batch directory - either resume existing or create new
if [ -n "$RESUME_BATCH_DIR" ]; then
    BATCH_DIR="${PROJECT_ROOT}/${RESUME_BATCH_DIR}"
    echo "Resuming batch: $BATCH_DIR"

    # Verify the batch directory exists
    if [ ! -d "$BATCH_DIR" ]; then
        echo "ERROR: Resume batch directory does not exist: $BATCH_DIR"
        exit 1
    fi

    # Verify batch_summary.json exists
    if [ ! -f "${BATCH_DIR}/batch_summary.json" ]; then
        echo "ERROR: batch_summary.json not found in: $BATCH_DIR"
        echo "Cannot resume - this may not be a valid batch directory."
        exit 1
    fi
else
    BATCH_TIMESTAMP=$(date +"%Y/%m/%d/%H:%M")
    BATCH_DIR="${PROJECT_ROOT}/outputs/${BATCH_TIMESTAMP}"
    echo "Creating new batch: $BATCH_DIR"

    BATCH_CONFIGS_DIR="${BATCH_DIR}/configs"
    mkdir -p "$BATCH_DIR"
    mkdir -p "$BATCH_CONFIGS_DIR"

    # Initialize batch_summary.json (only for new batches)
    cat > "${BATCH_DIR}/batch_summary.json" << EOF
{
  "batch_start_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "batch_dir": "$BATCH_DIR",
  "total_experiments": $total_experiments,
  "completed_experiments": 0,
  "experiments": {}
}
EOF

    # Create batch_config.json (only for new batches)
    cat > "${BATCH_DIR}/batch_config.json" << EOF
{
  "agents_config": "$AGENTS_CONFIG",
  "evaluation_config": "$EVALUATION_CONFIG",
  "parallel_experiments": $PARALLEL_EXPERIMENTS,
  "experiment_workers": $EXPERIMENT_WORKERS,
  "games": [$(printf '"%s",' "${GAME_CONFIGS[@]}" | sed 's/,$//')],
  "mechanisms": [$(printf '"%s",' "${MECHANISM_CONFIGS[@]}" | sed 's/,$//')],
  "total_experiments": $total_experiments
}
EOF
fi

BATCH_CONFIGS_DIR="${BATCH_DIR}/configs"
mkdir -p "$BATCH_CONFIGS_DIR"

echo "Batch directory: $BATCH_DIR"
echo ""

# =============================================================================
# RESUME HELPER
# =============================================================================

is_experiment_completed() {
    local exp_name=$1
    local summary_file="${BATCH_DIR}/batch_summary.json"

    if [ ! -f "$summary_file" ]; then
        return 1  # false - summary doesn't exist
    fi

    # Check if experiment exists and should be skipped
    $PYTHON_BIN -c "
import json
import sys

summary_file = '${summary_file}'
exp_name = '${exp_name}'
retry_failed = '${RETRY_FAILED_EXPERIMENTS}'

try:
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    exp = summary.get('experiments', {}).get(exp_name)
    if not exp:
        sys.exit(1)  # false - experiment doesn't exist, should run

    status = exp.get('status')

    # If successful, always skip
    if status == 'success':
        sys.exit(0)  # true - skip

    # If failed, skip only if retry is disabled
    if status == 'failed':
        if retry_failed == 'false':
            sys.exit(0)  # true - skip (don't retry)
        else:
            sys.exit(1)  # false - retry

    # Otherwise (in_progress or missing status), don't skip
    sys.exit(1)  # false - don't skip, should run
except Exception as e:
    sys.exit(1)  # false - error reading file
"
    return $?
}

# =============================================================================
# PARALLEL EXECUTION HELPERS
# =============================================================================

# Function to run a single experiment
run_single_experiment() {
    local game=$1
    local mechanism=$2
    local current=$3

    # Generate experiment name
    game_name=$(basename "$game" .yaml)
    mechanism_name=$(basename "$mechanism" .yaml)
    experiment_name="${mechanism_name}_${game_name}"

    experiment_dir="${BATCH_DIR}/${experiment_name}"

    # Check if experiment should be skipped (already completed successfully or failed with retry disabled)
    if is_experiment_completed "$experiment_name"; then
        echo "[$current/$total_experiments] SKIPPING (already completed): $experiment_name"
        echo ""
        return 0
    fi

    # If we get here, the experiment needs to run (either new, in_progress, or failed with retry enabled)
    # Check if directory exists from a previous failed run
    if [ -d "$experiment_dir" ]; then
        # Directory exists - either crashed or failed with retry enabled
        if [ "$RETRY_FAILED_EXPERIMENTS" = true ]; then
            echo "[$current/$total_experiments] RETRYING (failed previously): $experiment_name"
            echo "  Removing previous experiment directory..."
            rm -rf "$experiment_dir"
        else
            echo "[$current/$total_experiments] SKIPPING (failed previously, retry disabled): $experiment_name"
            echo ""
            return 0
        fi
    else
        echo "[$current/$total_experiments] Running: $experiment_name"
    fi

    mkdir -p "$experiment_dir"

    echo "  Game: $game"
    echo "  Mechanism: $mechanism"
    echo "  Output: $experiment_dir"
    echo "--------------------------------------------------"

    # Mark experiment as in_progress in batch_summary.json BEFORE running
    # This prevents orphaned directories if the experiment crashes
    experiment_start=$(date +%s)
    experiment_start_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    $PYTHON_BIN -c "
import json
from pathlib import Path
import fcntl

summary_path = Path('${BATCH_DIR}/batch_summary.json')

# Use file locking for concurrent writes
with open(summary_path, 'r+') as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    summary = json.load(f)

    summary['experiments']['${experiment_name}'] = {
        'game': '${game}',
        'mechanism': '${mechanism}',
        'start_time': '${experiment_start_iso}',
        'status': 'in_progress',
        'output_dir': '${experiment_dir}'
    }

    f.seek(0)
    f.truncate()
    json.dump(summary, f, indent=2)
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
"

    # Create temp config in batch configs directory
    TEMP_CONFIG="${BATCH_CONFIGS_DIR}/${experiment_name}.yaml"
    cat > "$TEMP_CONFIG" << EOF
# Auto-generated config for game-mechanism combination
game_config: $game
mechanism_config: $mechanism
agents_config: $AGENTS_CONFIG
evaluation_config: $EVALUATION_CONFIG
name: $experiment_name
concurrency:
  max_workers: $EXPERIMENT_WORKERS
EOF

    # Run experiment with timing and output capture

    # Get relative path from project root to temp config
    TEMP_CONFIG_RELATIVE="${TEMP_CONFIG#$PROJECT_ROOT/configs/}"

    $PYTHON_BIN script/run_experiment.py \
        --config "$TEMP_CONFIG_RELATIVE" \
        --output-dir "$BATCH_DIR" \
        --experiment-name "$experiment_name" \
        > "${experiment_dir}/stdout.txt" 2> "${experiment_dir}/stderr.txt"

    exit_code=$?
    experiment_end=$(date +%s)
    experiment_end_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    duration=$((experiment_end - experiment_start))

    # Update batch_summary.json with file locking
    $PYTHON_BIN -c "
import json
from pathlib import Path
import fcntl

summary_path = Path('${BATCH_DIR}/batch_summary.json')

# Use file locking for concurrent writes
with open(summary_path, 'r+') as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    summary = json.load(f)

    summary['experiments']['${experiment_name}'] = {
        'game': '${game}',
        'mechanism': '${mechanism}',
        'start_time': '${experiment_start_iso}',
        'end_time': '${experiment_end_iso}',
        'duration_seconds': ${duration},
        'status': 'success' if ${exit_code} == 0 else 'failed',
        'exit_code': ${exit_code},
        'output_dir': '${experiment_dir}'
    }

    summary['completed_experiments'] = len([
        e for e in summary['experiments'].values()
        if e['status'] in ['success', 'failed']
    ])

    f.seek(0)
    f.truncate()
    json.dump(summary, f, indent=2)
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
"

    # Print status
    if [ $exit_code -eq 0 ]; then
        echo "✓ $experiment_name: Completed successfully (${duration}s)"
    else
        echo "✗ $experiment_name: Failed with exit code $exit_code (${duration}s)"
    fi

    echo ""
    return $exit_code
}

# Export function and variables for parallel execution
export -f run_single_experiment
export -f is_experiment_completed
export PYTHON_BIN
export BATCH_DIR
export BATCH_CONFIGS_DIR
export AGENTS_CONFIG
export EVALUATION_CONFIG
export EXPERIMENT_WORKERS
export RETRY_FAILED_EXPERIMENTS
export PROJECT_ROOT
export total_experiments

# =============================================================================
# MAIN EXPERIMENT LOOP (PARALLEL)
# =============================================================================

echo "Running experiments with parallel_experiments: $PARALLEL_EXPERIMENTS, experiment_workers: $EXPERIMENT_WORKERS"
echo ""

# Array to hold background job PIDs
declare -a job_pids=()
declare -a job_names=()

# Loop through all combinations and launch jobs
current=0
for game in "${GAME_CONFIGS[@]}"; do
    for mechanism in "${MECHANISM_CONFIGS[@]}"; do
        current=$((current + 1))

        # Wait if we've reached max concurrency
        while [ ${#job_pids[@]} -ge $PARALLEL_EXPERIMENTS ]; do
            # Check each job
            for i in "${!job_pids[@]}"; do
                pid=${job_pids[$i]}
                # Check if job is still running
                if ! kill -0 $pid 2>/dev/null; then
                    # Job finished, remove from array
                    wait $pid  # Collect exit status
                    unset 'job_pids[$i]'
                    unset 'job_names[$i]'
                fi
            done
            # Re-index arrays to remove gaps
            job_pids=("${job_pids[@]}")
            job_names=("${job_names[@]}")

            # If still at max capacity, sleep briefly
            if [ ${#job_pids[@]} -ge $PARALLEL_EXPERIMENTS ]; then
                sleep 1
            fi
        done

        # Launch experiment in background
        run_single_experiment "$game" "$mechanism" "$current" &
        job_pids+=($!)
        job_names+=("${mechanism}_$(basename "$game" .yaml)")
    done
done

# Wait for all remaining jobs to complete
echo ""
echo "Waiting for remaining experiments to complete..."
for pid in "${job_pids[@]}"; do
    wait $pid
done

echo "All experiments launched and completed."
echo ""

# =============================================================================
# BATCH FINALIZATION
# =============================================================================

# Finalize batch_summary.json with statistics
$PYTHON_BIN -c "
import json
from datetime import datetime
from pathlib import Path

summary_path = Path('${BATCH_DIR}/batch_summary.json')
with open(summary_path, 'r') as f:
    summary = json.load(f)

summary['batch_end_time'] = datetime.utcnow().isoformat() + 'Z'

experiments = summary['experiments']
total = len(experiments)
successful = len([e for e in experiments.values() if e['status'] == 'success'])
failed = len([e for e in experiments.values() if e['status'] == 'failed'])
total_duration = sum(e.get('duration_seconds', 0) for e in experiments.values())

summary['statistics'] = {
    'total': total,
    'successful': successful,
    'failed': failed,
    'total_duration_seconds': total_duration
}

with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print('=' * 60)
print('All experiments completed!')
print('=' * 60)
print(f'Results directory: ${BATCH_DIR}')
print(f'Total experiments: {total}')
print(f'Successful: {successful}')
print(f'Failed: {failed}')
print(f'Total time: {total_duration}s ({total_duration / 60:.1f} minutes)')
print('=' * 60)
"
