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

# Use jq to extract experiment details from manifest
if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is not installed. Please install jq to use this script."
    exit 1
fi

EXPERIMENT=$(jq ".[$EXPERIMENT_INDEX]" "$MANIFEST_FILE")

if [ "$EXPERIMENT" == "null" ]; then
    echo "ERROR: Experiment index $EXPERIMENT_INDEX not found in manifest"
    exit 1
fi

GAME=$(echo "$EXPERIMENT" | jq -r '.game')
MECHANISM=$(echo "$EXPERIMENT" | jq -r '.mechanism')
EXP_NAME=$(echo "$EXPERIMENT" | jq -r '.experiment_name')
CONFIG_PATH=$(echo "$EXPERIMENT" | jq -r '.config_path')
AGENTS_CONFIG=$(echo "$EXPERIMENT" | jq -r '.agents_config')
EVALUATION_CONFIG=$(echo "$EXPERIMENT" | jq -r '.evaluation_config')
EXPERIMENT_WORKERS=$(echo "$EXPERIMENT" | jq -r '.experiment_workers')
TOURNAMENT_WORKERS=$(echo "$EXPERIMENT" | jq -r '.tournament_workers')
RETRY_FAILED=$(echo "$EXPERIMENT" | jq -r '.retry_failed_experiments')

EXPERIMENT_DIR="${BATCH_DIR}/${EXP_NAME}"

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
