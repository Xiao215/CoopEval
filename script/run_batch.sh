#!/bin/bash
# Main batch runner script - orchestrates experiment execution
# Usage: ./script/run_batch.sh [--local|--slurm] [--batch-name NAME] [--resume BATCH_DIR]
# An example of resuming could be
# ./script/run_batch.sh --slurm --resume outputs/test_resume_20260120_024307
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
# AGENTS_CONFIG="agents/sota_llms.yaml"
AGENTS_CONFIG="agents/few_strong_llms.yaml"

# Evaluation configuration (relative to configs/)
# EVALUATION_CONFIG="evaluation/default_evaluation.yaml"
EVALUATION_CONFIG="evaluation/no_deviation_ratings.yaml"

# Concurrency settings
EXPERIMENT_WORKERS=3    # Number of parallel workers within each experiment (for LLM queries)
TOURNAMENT_WORKERS=9    # Number of parallel matchups within each tournament (1=sequential)

# Retry settings
RETRY_FAILED_EXPERIMENTS=true  # Set to false to skip failed experiments instead of retrying them

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
    "mechanisms/mediation.yaml"
    "mechanisms/reputation.yaml"
    "mechanisms/repetition.yaml"
    "mechanisms/reputation_zero_order.yaml"

    # "mechanisms/disarmament.yaml"
)

# =============================================================================
# END CONFIGURATION
# =============================================================================

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

MODE=""
BATCH_NAME=""
RESUME_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            MODE="local"
            shift
            ;;
        --slurm)
            MODE="slurm"
            shift
            ;;
        --batch-name)
            BATCH_NAME="$2"
            shift 2
            ;;
        --resume)
            RESUME_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--local|--slurm] [--batch-name NAME] [--resume BATCH_DIR]"
            exit 1
            ;;
    esac
done

# Validate mode
if [ -z "$MODE" ] && [ -z "$RESUME_DIR" ]; then
    echo "ERROR: Must specify --local or --slurm"
    echo "Usage: $0 [--local|--slurm] [--batch-name NAME] [--resume BATCH_DIR]"
    exit 1
fi

# If resuming, mode is required
if [ -n "$RESUME_DIR" ] && [ -z "$MODE" ]; then
    echo "ERROR: Must specify --local or --slurm when resuming"
    echo "Usage: $0 --resume BATCH_DIR [--local|--slurm]"
    exit 1
fi

# =============================================================================
# INITIALIZATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source shared utilities
source "$SCRIPT_DIR/batch_utils.sh"

# Change to project root
cd "$PROJECT_ROOT"

# Activate conda environment
activate_conda_env "llmcoop"

# Calculate total experiments
total_experiments=$((${#GAME_CONFIGS[@]} * ${#MECHANISM_CONFIGS[@]}))

echo "=================================================="
echo "Batch Execution Mode: $MODE"
echo "Total experiments: $total_experiments"
echo "Games: ${#GAME_CONFIGS[@]}, Mechanisms: ${#MECHANISM_CONFIGS[@]}"
echo "=================================================="
echo ""

# =============================================================================
# BATCH DIRECTORY SETUP
# =============================================================================

if [ -n "$RESUME_DIR" ]; then
    # Resume existing batch
    BATCH_DIR="${PROJECT_ROOT}/${RESUME_DIR}"
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

    # Verify experiments.json exists
    if [ ! -f "${BATCH_DIR}/experiments.json" ]; then
        echo "ERROR: experiments.json not found in: $BATCH_DIR"
        echo "Cannot resume - this may not be a valid batch directory."
        exit 1
    fi
else
    # Create new batch
    if [ -n "$BATCH_NAME" ]; then
        # Use descriptive name + timestamp
        BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        BATCH_DIR="${PROJECT_ROOT}/outputs/${BATCH_NAME}_${BATCH_TIMESTAMP}"
        echo "Creating new batch with custom name: $BATCH_DIR"
    else
        # Use timestamp-only (backward compatible)
        BATCH_TIMESTAMP=$(date +"%Y/%m/%d/%H:%M")
        BATCH_DIR="${PROJECT_ROOT}/outputs/${BATCH_TIMESTAMP}"
        echo "Creating new batch with timestamp: $BATCH_DIR"
    fi

    # Create batch directories
    mkdir -p "$BATCH_DIR"
    BATCH_CONFIGS_DIR="${BATCH_DIR}/configs"
    mkdir -p "$BATCH_CONFIGS_DIR"

    if [ "$MODE" == "slurm" ]; then
        mkdir -p "${BATCH_DIR}/slurm"
    fi

    # Initialize batch_summary.json
    cat > "${BATCH_DIR}/batch_summary.json" << EOF
{
  "batch_start_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "batch_dir": "$BATCH_DIR",
  "total_experiments": $total_experiments,
  "completed_experiments": 0,
  "experiments": {}
}
EOF

    # Create batch_config.json
    cat > "${BATCH_DIR}/batch_config.json" << EOF
{
  "agents_config": "$AGENTS_CONFIG",
  "evaluation_config": "$EVALUATION_CONFIG",
  "experiment_workers": $EXPERIMENT_WORKERS,
  "tournament_workers": $TOURNAMENT_WORKERS,
  "retry_failed_experiments": $RETRY_FAILED_EXPERIMENTS,
  "games": [$(printf '"%s",' "${GAME_CONFIGS[@]}" | sed 's/,$//')],
  "mechanisms": [$(printf '"%s",' "${MECHANISM_CONFIGS[@]}" | sed 's/,$//')],
  "total_experiments": $total_experiments,
  "mode": "$MODE"
}
EOF
fi

BATCH_CONFIGS_DIR="${BATCH_DIR}/configs"
mkdir -p "$BATCH_CONFIGS_DIR"

echo "Batch directory: $BATCH_DIR"
echo ""

# =============================================================================
# MANIFEST GENERATION
# =============================================================================

MANIFEST_FILE="${BATCH_DIR}/experiments.json"

if [ ! -f "$MANIFEST_FILE" ]; then
    echo "Generating experiment manifest..."

    # Convert boolean to Python format (bash 3-compatible)
    if [ "$RETRY_FAILED_EXPERIMENTS" = "true" ]; then
        RETRY_FAILED_PY="True"
    else
        RETRY_FAILED_PY="False"
    fi

    # Use Python to generate clean JSON manifest
    $PYTHON_BIN << EOF
import json

experiments = []
index = 0

games = [$(printf '"%s",' "${GAME_CONFIGS[@]}" | sed 's/,$//')]
mechanisms = [$(printf '"%s",' "${MECHANISM_CONFIGS[@]}" | sed 's/,$//')]

for game in games:
    for mechanism in mechanisms:
        game_name = game.replace('games/', '').replace('.yaml', '')
        mech_name = mechanism.replace('mechanisms/', '').replace('.yaml', '')
        exp_name = f"{mech_name}_{game_name}"
        config_path = f"${BATCH_CONFIGS_DIR}/{exp_name}.yaml"

        experiments.append({
            "index": index,
            "game": game,
            "mechanism": mechanism,
            "experiment_name": exp_name,
            "config_path": config_path,
            "agents_config": "${AGENTS_CONFIG}",
            "evaluation_config": "${EVALUATION_CONFIG}",
            "experiment_workers": ${EXPERIMENT_WORKERS},
            "tournament_workers": ${TOURNAMENT_WORKERS},
            "retry_failed_experiments": ${RETRY_FAILED_PY}
        })
        index += 1

with open("${MANIFEST_FILE}", 'w') as f:
    json.dump(experiments, f, indent=2)

print(f"Generated manifest with {len(experiments)} experiments")
EOF

    echo "Manifest generated: $MANIFEST_FILE"
else
    echo "Using existing manifest: $MANIFEST_FILE"
fi

# Read number of experiments from manifest (using Python instead of jq)
num_experiments=$($PYTHON_BIN -c "import json; print(len(json.load(open('${MANIFEST_FILE}'))))")
echo "Total experiments in manifest: $num_experiments"
echo ""

# =============================================================================
# EXECUTION MODE DISPATCH
# =============================================================================

if [ "$MODE" == "local" ]; then
    echo "=================================================="
    echo "Running batch in LOCAL mode (sequential execution)"
    echo "=================================================="
    echo ""

    # Sequential execution
    for i in $(seq 0 $((num_experiments - 1))); do
        echo "[$((i+1))/$num_experiments] Running experiment $i..."
        "$SCRIPT_DIR/run_single_experiment.sh" "$i" "$BATCH_DIR"
        echo ""
    done

    # Finalize batch summary
    finalize_batch_summary "$BATCH_DIR"

    # Print final statistics
    echo ""
    print_batch_statistics "$BATCH_DIR"

elif [ "$MODE" == "slurm" ]; then
    echo "=================================================="
    echo "Running batch in SLURM mode (job array)"
    echo "=================================================="
    echo ""

    # Generate SLURM job array script from template
    SLURM_SCRIPT="${BATCH_DIR}/slurm/job_array.sh"

    # Copy template and substitute variables
    sed -e "s|{{PROJECT_ROOT}}|${PROJECT_ROOT}|g" \
        -e "s|{{SCRIPT_DIR}}|${SCRIPT_DIR}|g" \
        -e "s|{{BATCH_DIR}}|${BATCH_DIR}|g" \
        "$SCRIPT_DIR/slurm_array_template.sh" > "$SLURM_SCRIPT"

    chmod +x "$SLURM_SCRIPT"

    echo "SLURM script generated: $SLURM_SCRIPT"
    echo ""

    # Submit job array (logs in flat slurm/ directory)
    echo "Submitting SLURM job array..."
    JOB_ID=$(sbatch --parsable \
        --array=0-$((num_experiments - 1)) \
        --output="${BATCH_DIR}/slurm/slurm-%A_%a.out" \
        --error="${BATCH_DIR}/slurm/slurm-%A_%a.err" \
        "$SLURM_SCRIPT")

    if [ $? -eq 0 ]; then
        echo "=================================================="
        echo "SLURM job array submitted successfully!"
        echo "=================================================="
        echo "Job ID: $JOB_ID"
        echo "Array size: $num_experiments"
        echo "Batch directory: $BATCH_DIR"
        echo "Logs directory: ${BATCH_DIR}/slurm/"
        echo ""
        echo "Monitor job status:"
        echo "  squeue -j $JOB_ID"
        echo "  squeue -u \$USER"
        echo ""
        echo "Check batch progress:"
        echo "  watch -n 5 'python -c \"import json; d=json.load(open(\\\"${BATCH_DIR}/batch_summary.json\\\")); print(f\\\"Completed: {d.get(\\\\\\\"completed_experiments\\\\\\\", 0)}/{d.get(\\\\\\\"total_experiments\\\\\\\", 0)}\\\")\"'"
        echo ""
        echo "View logs (by job ID):"
        echo "  tail -f ${BATCH_DIR}/slurm/${JOB_ID}_*.out"
        echo "=================================================="
    else
        echo "ERROR: Failed to submit SLURM job array"
        exit 1
    fi
fi

echo "Batch execution initiated successfully"
