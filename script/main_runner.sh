#!/bin/bash
# Main runner script to execute run_experiment.py for all combinations of games and mechanisms
# Usage: ./main_runner.sh

# Don't exit on error - we want to continue even if individual experiments fail
set +e

# Trap Ctrl+C and cleanup
trap 'echo ""; echo "Interrupted! Batch summary saved to: ${BATCH_DIR}/batch_summary.json"; exit 130' INT

# =============================================================================
# CONFIGURATION - Edit these variables to customize your experiments
# =============================================================================

# Agents configuration (relative to configs/)
# AGENTS_CONFIG="agents/test_agents_3.yaml"
AGENTS_CONFIG="agents/cheap_llms_3.yaml"

# Evaluation configuration (relative to configs/)
EVALUATION_CONFIG="evaluation/default_evaluation.yaml"

# Concurrency setting (number of parallel workers)
CONCURRENCY=1

# Batch directory - set to existing path to resume, or leave empty for new batch with timestamp
# RESUME_BATCH_DIR="outputs/2026/01/11/23:00"
RESUME_BATCH_DIR=""

# List of game config paths (relative to configs/)
# Based on games in src/games/
GAME_CONFIGS=(
    "games/matching_pennies.yaml"
    "games/prisoners_dilemma.yaml"
    "games/public_goods.yaml"
    "games/stag_hunt.yaml"
    "games/travellers_dilemma.yaml"
    "games/trust_game.yaml"
)

# List of mechanism config paths (relative to configs/)
# Based on mechanisms in src/mechanisms/
MECHANISM_CONFIGS=(
    "mechanisms/no_mechanism.yaml"
    "mechanisms/contracting.yaml"
    "mechanisms/disarmament.yaml"
    # "mechanisms/mediation.yaml"
    "mechanisms/repetition.yaml"
    "mechanisms/reputation.yaml"
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

# Activate conda environment if needed (uncomment and modify as needed)
# source ~/anaconda3/bin/activate llmcoop

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
  "concurrency": $CONCURRENCY,
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

    # Check if experiment exists and has a status
    python3 -c "
import json
import sys

summary_file = '${summary_file}'
exp_name = '${exp_name}'

try:
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    exp = summary.get('experiments', {}).get(exp_name)
    if exp and exp.get('status') in ['success', 'failed']:
        sys.exit(0)  # true - experiment completed
    else:
        sys.exit(1)  # false - experiment not completed
except Exception as e:
    sys.exit(1)  # false - error reading file
"
    return $?
}

# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

# Loop through all combinations
for game in "${GAME_CONFIGS[@]}"; do
    for mechanism in "${MECHANISM_CONFIGS[@]}"; do
        current=$((current + 1))

        # Generate experiment name
        game_name=$(basename "$game" .yaml)
        mechanism_name=$(basename "$mechanism" .yaml)
        experiment_name="${mechanism_name}_${game_name}"

        # Skip if already completed (resume support)
        if is_experiment_completed "$experiment_name"; then
            echo "[$current/$total_experiments] SKIPPING (already completed): $experiment_name"
            echo ""
            continue
        fi

        # Check for orphaned/crashed experiment directory
        experiment_dir="${BATCH_DIR}/${experiment_name}"
        if [ -d "$experiment_dir" ]; then
            # Directory exists but experiment not marked as completed
            # This means it crashed - delete and retry
            echo "[$current/$total_experiments] RETRYING (crashed previously): $experiment_name"
            echo "  Removing crashed experiment directory..."
            rm -rf "$experiment_dir"
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

        python3 -c "
import json
from pathlib import Path

summary_path = Path('${BATCH_DIR}/batch_summary.json')
with open(summary_path, 'r') as f:
    summary = json.load(f)

summary['experiments']['${experiment_name}'] = {
    'game': '${game}',
    'mechanism': '${mechanism}',
    'start_time': '${experiment_start_iso}',
    'status': 'in_progress',
    'output_dir': '${experiment_dir}'
}

with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
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
  max_workers: $CONCURRENCY
EOF

        # Run experiment with timing and output capture

        # Get relative path from project root to temp config
        TEMP_CONFIG_RELATIVE="${TEMP_CONFIG#$PROJECT_ROOT/configs/}"

        python3 script/run_experiment.py \
            --config "$TEMP_CONFIG_RELATIVE" \
            --output-dir "$BATCH_DIR" \
            --experiment-name "$experiment_name" \
            > "${experiment_dir}/stdout.txt" 2> "${experiment_dir}/stderr.txt"

        exit_code=$?
        experiment_end=$(date +%s)
        experiment_end_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        duration=$((experiment_end - experiment_start))

        # Update batch_summary.json
        python3 -c "
import json
from pathlib import Path

summary_path = Path('${BATCH_DIR}/batch_summary.json')
with open(summary_path, 'r') as f:
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

with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
"

        # Print status
        if [ $exit_code -eq 0 ]; then
            echo "✓ Completed successfully (${duration}s)"
        else
            echo "✗ Failed with exit code $exit_code (${duration}s)"
        fi

        echo ""
    done
done

# =============================================================================
# BATCH FINALIZATION
# =============================================================================

# Finalize batch_summary.json with statistics
python3 -c "
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
