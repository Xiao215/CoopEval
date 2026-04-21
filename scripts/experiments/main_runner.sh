#!/bin/bash
# Main runner script to execute run_experiment.py for all combinations of games and mechanisms
# Usage: ./main_runner.sh

export PYTHONPATH=.

set +e

trap 'echo ""; echo "Interrupted! Batch summary saved to: ${BATCH_DIR}/batch_summary.json"; exit 130' INT

# Agents configuration (relative to configs/)
AGENTS_CONFIG="agents/test_agents_6.yaml"
# AGENTS_CONFIG="agents/cheap_llms_3.yaml"
# # AGENTS_CONFIG="agents/sota_llms.yaml"
# AGENTS_CONFIG="agents/few_strong_llms.yaml"

# Evaluation configuration (relative to configs/)
# EVALUATION_CONFIG="evaluation/default_evaluation.yaml"
EVALUATION_CONFIG="evaluation/no_deviation_ratings.yaml"

# Parallel execution settings
PARALLEL_EXPERIMENTS=4  # Number of experiments to run simultaneously
EXPERIMENT_WORKERS=3    # Number of parallel workers within each experiment (for LLM queries)
TOURNAMENT_WORKERS=3    # Number of parallel matchups within each tournament (1=sequential)

# Retry settings
RETRY_FAILED_EXPERIMENTS=true  # Set to false to skip failed experiments instead of retrying them

# Batch directory - set to existing path to resume, or leave empty for new batch with timestamp
# RESUME_BATCH_DIR="logs/2026/01/12/01:57"
RESUME_BATCH_DIR=""

# Based on games in src/games/
GAME_CONFIGS=(
    "games/matching_pennies.yaml"
    "games/prisoners_dilemma.yaml"
    "games/public_goods.yaml"
    "games/stag_hunt.yaml"
    "games/travellers_dilemma.yaml"
    "games/trust_game.yaml"
)

# Based on mechanisms in src/mechanisms/
MECHANISM_CONFIGS=(
    "mechanisms/no_mechanism.yaml"
    "mechanisms/contracting.yaml"
    "mechanisms/mediation.yaml"
    "mechanisms/repetition.yaml"
    "mechanisms/reputation.yaml"
    "mechanisms/reputation_first_order.yaml"
)

# GAME_CONFIGS=("games/matching_pennies.yaml" "games/prisoners_dilemma.yaml")
# MECHANISM_CONFIGS=("mechanisms/no_mechanism.yaml" "mechanisms/repetition.yaml")

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

echo "Activating conda environment: llmcoop"

if [ -z "$CONDA_EXE" ]; then
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        eval "$(conda shell.bash hook)" 2>/dev/null || true
    fi
fi

conda activate llmcoop

if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN="$(which python3)"
fi

echo "Python path: $PYTHON_BIN"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Conda prefix: $CONDA_PREFIX"

total_experiments=$((${#GAME_CONFIGS[@]} * ${#MECHANISM_CONFIGS[@]}))
current=0

echo "=================================================="
echo "Running $total_experiments experiments"
echo "Games: ${#GAME_CONFIGS[@]}, Mechanisms: ${#MECHANISM_CONFIGS[@]}"
echo "=================================================="
echo ""

if [ -n "$RESUME_BATCH_DIR" ]; then
    BATCH_DIR="${PROJECT_ROOT}/${RESUME_BATCH_DIR}"
    echo "Resuming batch: $BATCH_DIR"

    if [ ! -d "$BATCH_DIR" ]; then
        echo "ERROR: Resume batch directory does not exist: $BATCH_DIR"
        exit 1
    fi

    if [ ! -f "${BATCH_DIR}/batch_summary.json" ]; then
        echo "ERROR: batch_summary.json not found in: $BATCH_DIR"
        echo "Cannot resume - this may not be a valid batch directory."
        exit 1
    fi
else
    BATCH_TIMESTAMP=$(date +"%Y/%m/%d/%H:%M")
    BATCH_DIR="${PROJECT_ROOT}/logs/${BATCH_TIMESTAMP}"
    echo "Creating new batch: $BATCH_DIR"

    BATCH_CONFIGS_DIR="${BATCH_DIR}/configs"
    mkdir -p "$BATCH_DIR"
    mkdir -p "$BATCH_CONFIGS_DIR"

    cat > "${BATCH_DIR}/batch_summary.json" << EOF
{
  "batch_start_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "batch_dir": "$BATCH_DIR",
  "total_experiments": $total_experiments,
  "completed_experiments": 0,
  "experiments": {}
}
EOF

    cat > "${BATCH_DIR}/batch_config.json" << EOF
{
  "agents_config": "$AGENTS_CONFIG",
  "evaluation_config": "$EVALUATION_CONFIG",
  "parallel_experiments": $PARALLEL_EXPERIMENTS,
  "experiment_workers": $EXPERIMENT_WORKERS,
  "tournament_workers": $TOURNAMENT_WORKERS,
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

is_experiment_completed() {
    local exp_name=$1
    local summary_file="${BATCH_DIR}/batch_summary.json"

    if [ ! -f "$summary_file" ]; then
        return 1
    fi

    $PYTHON_BIN -c "
import json
import sys
import fcntl
from pathlib import Path

summary_path = Path('${summary_file}')
exp_name = '${exp_name}'
retry_failed = '${RETRY_FAILED_EXPERIMENTS}'

try:
    with summary_path.open('r') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        summary = json.load(f)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
except Exception:
    sys.exit(1)

exp = summary.get('experiments', {}).get(exp_name)
if not exp:
    sys.exit(1)

status = exp.get('status')

if status == 'success':
    sys.exit(0)

if status == 'failed':
    if retry_failed == 'false':
        sys.exit(0)
    else:
        sys.exit(1)

sys.exit(1)
"
    return $?
}

run_single_experiment() {
    local game=$1
    local mechanism=$2
    local current=$3

    game_name=$(basename "$game" .yaml)
    mechanism_name=$(basename "$mechanism" .yaml)
    experiment_name="${mechanism_name}_${game_name}"

    experiment_dir="${BATCH_DIR}/${experiment_name}"

    if is_experiment_completed "$experiment_name"; then
        echo "[$current/$total_experiments] SKIPPING (already completed): $experiment_name"
        echo ""
        return 0
    fi

    if [ -d "$experiment_dir" ]; then
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

    experiment_start=$(date +%s)
    experiment_start_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    $PYTHON_BIN -c "
import json
from pathlib import Path
import fcntl

summary_path = Path('${BATCH_DIR}/batch_summary.json')

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

    TEMP_CONFIG="${BATCH_CONFIGS_DIR}/${experiment_name}.yaml"
    cat > "$TEMP_CONFIG" << EOF
game_config: $game
mechanism_config: $mechanism
agents_config: $AGENTS_CONFIG
evaluation_config: $EVALUATION_CONFIG
name: $experiment_name
concurrency:
  max_workers: $EXPERIMENT_WORKERS
  tournament_workers: $TOURNAMENT_WORKERS
EOF

    TEMP_CONFIG_RELATIVE="${TEMP_CONFIG#$PROJECT_ROOT/configs/}"

    $PYTHON_BIN scripts/experiments/run_experiment.py \
        --config "$TEMP_CONFIG_RELATIVE" \
        --output-dir "$BATCH_DIR" \
        --experiment-name "$experiment_name" \
        > "${experiment_dir}/stdout.txt" 2> "${experiment_dir}/stderr.txt"

    exit_code=$?
    experiment_end=$(date +%s)
    experiment_end_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    duration=$((experiment_end - experiment_start))

    $PYTHON_BIN -c "
import json
from pathlib import Path
import fcntl

summary_path = Path('${BATCH_DIR}/batch_summary.json')

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

    if [ $exit_code -eq 0 ]; then
        echo "✓ $experiment_name: Completed successfully (${duration}s)"
    else
        echo "✗ $experiment_name: Failed with exit code $exit_code (${duration}s)"
    fi

    echo ""
    return $exit_code
}

export -f run_single_experiment
export -f is_experiment_completed
export PYTHON_BIN
export BATCH_DIR
export BATCH_CONFIGS_DIR
export AGENTS_CONFIG
export EVALUATION_CONFIG
export EXPERIMENT_WORKERS
export TOURNAMENT_WORKERS
export RETRY_FAILED_EXPERIMENTS
export PROJECT_ROOT
export total_experiments

echo "Running experiments with parallel_experiments: $PARALLEL_EXPERIMENTS, experiment_workers: $EXPERIMENT_WORKERS, tournament_workers: $TOURNAMENT_WORKERS"
echo ""

declare -a job_pids=()
declare -a job_names=()

current=0
for game in "${GAME_CONFIGS[@]}"; do
    for mechanism in "${MECHANISM_CONFIGS[@]}"; do
        current=$((current + 1))

        while [ ${#job_pids[@]} -ge $PARALLEL_EXPERIMENTS ]; do
            for i in "${!job_pids[@]}"; do
                pid=${job_pids[$i]}
                if ! kill -0 $pid 2>/dev/null; then
                    wait $pid
                    unset 'job_pids[$i]'
                    unset 'job_names[$i]'
                fi
            done
            job_pids=("${job_pids[@]}")
            job_names=("${job_names[@]}")

            if [ ${#job_pids[@]} -ge $PARALLEL_EXPERIMENTS ]; then
                sleep 1
            fi
        done

        run_single_experiment "$game" "$mechanism" "$current" &
        job_pids+=($!)
        job_names+=("${mechanism}_$(basename "$game" .yaml)")
    done
done

echo ""
echo "Waiting for remaining experiments to complete..."
for pid in "${job_pids[@]}"; do
    wait $pid
done

echo "All experiments launched and completed."
echo ""

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
