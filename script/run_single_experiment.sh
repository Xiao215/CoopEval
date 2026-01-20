#!/bin/bash
# Run a single experiment (called by run_batch.sh or SLURM job array)
# Usage: ./run_single_experiment.sh <experiment_index> <batch_dir>

set +e  # Don't exit on error - we want to track failures

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 <experiment_index> <batch_dir>"
    exit 1
fi

EXPERIMENT_INDEX=$1
BATCH_DIR=$2

# =============================================================================
# INITIALIZATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source shared utilities
source "$SCRIPT_DIR/batch_utils.sh"

# Change to project root
cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH=.

# Activate conda environment
activate_conda_env "llmcoop"

# Read manifest
MANIFEST_FILE="${BATCH_DIR}/experiments.json"

if [ ! -f "$MANIFEST_FILE" ]; then
    echo "ERROR: Manifest file not found: $MANIFEST_FILE"
    exit 1
fi

# =============================================================================
# EXTRACT EXPERIMENT METADATA
# =============================================================================

# Use Python to extract experiment details from manifest (no jq dependency)
# Export variables using eval to set them in the current shell
eval "$($PYTHON_BIN << EOF
import json
import sys

try:
    with open('${MANIFEST_FILE}', 'r') as f:
        experiments = json.load(f)

    if ${EXPERIMENT_INDEX} >= len(experiments):
        print("ERROR: Experiment index ${EXPERIMENT_INDEX} not found in manifest", file=sys.stderr)
        sys.exit(1)

    exp = experiments[${EXPERIMENT_INDEX}]

    # Print shell variable assignments
    print(f"GAME='{exp['game']}'")
    print(f"MECHANISM='{exp['mechanism']}'")
    print(f"EXP_NAME='{exp['experiment_name']}'")
    print(f"CONFIG_PATH='{exp['config_path']}'")
    print(f"AGENTS_CONFIG='{exp['agents_config']}'")
    print(f"EVALUATION_CONFIG='{exp['evaluation_config']}'")
    print(f"EXPERIMENT_WORKERS={exp['experiment_workers']}")
    print(f"TOURNAMENT_WORKERS={exp['tournament_workers']}")
    print(f"RETRY_FAILED={str(exp['retry_failed_experiments']).lower()}")

except Exception as e:
    print(f"ERROR: Failed to parse manifest: {e}", file=sys.stderr)
    sys.exit(1)
EOF
)"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to extract experiment metadata from manifest"
    exit 1
fi

EXPERIMENT_DIR="${BATCH_DIR}/${EXP_NAME}"

# Override config path to use local batch directory (for portability across machines)
# The manifest may contain absolute paths from a different machine
CONFIG_PATH="${BATCH_DIR}/configs/${EXP_NAME}.yaml"

# =============================================================================
# SLURM LOG SYMLINKS (if running in SLURM)
# =============================================================================

# Create descriptive symlinks for SLURM logs: <jobid>_<mechanism>_<game>_slurm.{out,err}
# Job ID first groups logs by submission when sorted alphabetically
if [ -n "$SLURM_ARRAY_JOB_ID" ] && [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_DIR="${BATCH_DIR}/slurm"
    SLURM_BASE="slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

    # Create symlinks with job ID first for easy grouping
    ln -sf "${SLURM_BASE}.out" "${SLURM_DIR}/${SLURM_ARRAY_JOB_ID}_${EXP_NAME}_slurm.out"
    ln -sf "${SLURM_BASE}.err" "${SLURM_DIR}/${SLURM_ARRAY_JOB_ID}_${EXP_NAME}_slurm.err"
fi

# =============================================================================
# PRE-EXECUTION CHECKS
# =============================================================================

# Check if already completed (resume support)
if is_experiment_completed "$EXP_NAME" "$BATCH_DIR" "$RETRY_FAILED"; then
    echo "[$EXPERIMENT_INDEX] SKIPPING (already completed): $EXP_NAME"
    exit 0
fi

# Clean up failed attempts if retrying
if [ -d "$EXPERIMENT_DIR" ]; then
    if [ "$RETRY_FAILED" = "true" ]; then
        echo "[$EXPERIMENT_INDEX] RETRYING (failed previously): $EXP_NAME"
        echo "  Removing previous experiment directory..."
        rm -rf "$EXPERIMENT_DIR"
    else
        echo "[$EXPERIMENT_INDEX] SKIPPING (failed previously, retry disabled): $EXP_NAME"
        exit 0
    fi
else
    echo "[$EXPERIMENT_INDEX] Running: $EXP_NAME"
fi

mkdir -p "$EXPERIMENT_DIR"

echo "  Game: $GAME"
echo "  Mechanism: $MECHANISM"
echo "  Output: $EXPERIMENT_DIR"
echo "--------------------------------------------------"

# =============================================================================
# STATUS TRACKING - IN PROGRESS
# =============================================================================

# Mark experiment as in_progress in batch_summary.json BEFORE running
experiment_start=$(date +%s)
experiment_start_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Build JSON for initial status
initial_json="{\"game\": \"${GAME}\", \"mechanism\": \"${MECHANISM}\", \"start_time\": \"${experiment_start_iso}\", \"output_dir\": \"${EXPERIMENT_DIR}\"}"
update_batch_summary "$EXP_NAME" "$BATCH_DIR" "in_progress" "$initial_json"

# =============================================================================
# CONFIG GENERATION
# =============================================================================

# Create temp config if it doesn't already exist
if [ ! -f "$CONFIG_PATH" ]; then
    cat > "$CONFIG_PATH" << EOF
# Auto-generated config for game-mechanism combination
game_config: $GAME
mechanism_config: $MECHANISM
agents_config: $AGENTS_CONFIG
evaluation_config: $EVALUATION_CONFIG
name: $EXP_NAME
concurrency:
  max_workers: $EXPERIMENT_WORKERS
  tournament_workers: $TOURNAMENT_WORKERS
EOF
fi

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

echo "Running experiment: $EXP_NAME"
echo "Config: $CONFIG_PATH"

# config_loader.py supports absolute paths, so use CONFIG_PATH directly
$PYTHON_BIN script/run_experiment.py \
    --config "$CONFIG_PATH" \
    --output-dir "$BATCH_DIR" \
    --experiment-name "$EXP_NAME" \
    > "${EXPERIMENT_DIR}/stdout.txt" 2> "${EXPERIMENT_DIR}/stderr.txt"

EXIT_CODE=$?

# =============================================================================
# STATUS TRACKING - COMPLETION
# =============================================================================

experiment_end=$(date +%s)
experiment_end_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
duration=$((experiment_end - experiment_start))

if [ $EXIT_CODE -eq 0 ]; then
    STATUS="success"
    echo "✓ $EXP_NAME: Completed successfully (${duration}s)"
else
    STATUS="failed"
    echo "✗ $EXP_NAME: Failed with exit code $EXIT_CODE (${duration}s)"
fi

# Update batch_summary.json with final status
final_json="{\"game\": \"${GAME}\", \"mechanism\": \"${MECHANISM}\", \"start_time\": \"${experiment_start_iso}\", \"end_time\": \"${experiment_end_iso}\", \"duration_seconds\": ${duration}, \"exit_code\": ${EXIT_CODE}, \"output_dir\": \"${EXPERIMENT_DIR}\"}"
update_batch_summary "$EXP_NAME" "$BATCH_DIR" "$STATUS" "$final_json"

echo ""

exit $EXIT_CODE
