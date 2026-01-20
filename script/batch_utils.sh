#!/bin/bash
# Shared utility functions for batch processing
# Used by run_batch.sh and run_single_experiment.sh

# =============================================================================
# CONDA ENVIRONMENT ACTIVATION
# =============================================================================

activate_conda_env() {
    local env_name=$1

    echo "Activating conda environment: $env_name"

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

    conda activate "$env_name"

    # Store the python path to use consistently throughout the script
    # First try to get python from conda environment, fall back to which
    if [ -n "$CONDA_PREFIX" ]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    else
        PYTHON_BIN="$(which python3)"
    fi

    # Export for use in other scripts
    export PYTHON_BIN

    # Verify activation
    echo "Python path: $PYTHON_BIN"
    echo "Conda env: $CONDA_DEFAULT_ENV"
    echo "Conda prefix: $CONDA_PREFIX"
    echo ""
}

# =============================================================================
# BATCH SUMMARY UPDATES (WITH FILE LOCKING)
# =============================================================================

update_batch_summary() {
    local exp_name=$1
    local batch_dir=$2
    local status=$3
    local additional_json=$4  # Optional: JSON string with additional fields

    local summary_file="${batch_dir}/batch_summary.json"

    # Use Python with fcntl for thread-safe updates
    $PYTHON_BIN -c "
import json
import fcntl
from pathlib import Path

summary_path = Path('${summary_file}')

# Use file locking for concurrent writes
with open(summary_path, 'r+') as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    summary = json.load(f)

    # Get or create experiment entry
    exp_entry = summary['experiments'].get('${exp_name}', {})
    exp_entry['status'] = '${status}'

    # Merge additional fields if provided
    if '''${additional_json}''':
        try:
            additional_fields = json.loads('''${additional_json}''')
            exp_entry.update(additional_fields)
        except json.JSONDecodeError:
            pass  # Ignore invalid JSON

    summary['experiments']['${exp_name}'] = exp_entry

    # Update completed count
    summary['completed_experiments'] = len([
        e for e in summary['experiments'].values()
        if e['status'] in ['success', 'failed']
    ])

    f.seek(0)
    f.truncate()
    json.dump(summary, f, indent=2)
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
"
}

# =============================================================================
# EXPERIMENT COMPLETION CHECK
# =============================================================================

is_experiment_completed() {
    local exp_name=$1
    local batch_dir=$2
    local retry_failed=${3:-true}  # Default to retrying failed experiments

    local summary_file="${batch_dir}/batch_summary.json"

    if [ ! -f "$summary_file" ]; then
        return 1  # false - summary doesn't exist
    fi

    # Check if experiment exists and should be skipped
    $PYTHON_BIN -c "
import json
import sys

summary_file = '${summary_file}'
exp_name = '${exp_name}'
retry_failed = '${retry_failed}'

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
# BATCH STATISTICS PRINTING
# =============================================================================

print_batch_statistics() {
    local batch_dir=$1
    local summary_file="${batch_dir}/batch_summary.json"

    if [ ! -f "$summary_file" ]; then
        echo "ERROR: batch_summary.json not found in $batch_dir"
        return 1
    fi

    $PYTHON_BIN -c "
import json
from pathlib import Path

summary_path = Path('${summary_file}')
with open(summary_path, 'r') as f:
    summary = json.load(f)

experiments = summary.get('experiments', {})
total = len(experiments)
successful = len([e for e in experiments.values() if e.get('status') == 'success'])
failed = len([e for e in experiments.values() if e.get('status') == 'failed'])
in_progress = len([e for e in experiments.values() if e.get('status') == 'in_progress'])

# Calculate total duration if available
total_duration = sum(e.get('duration_seconds', 0) for e in experiments.values() if 'duration_seconds' in e)

print('=' * 60)
print('Batch Summary')
print('=' * 60)
print(f'Batch Directory: ${batch_dir}')
print(f'Total Experiments: {total}')
print(f'Successful: {successful}')
print(f'Failed: {failed}')
print(f'In Progress: {in_progress}')
if total_duration > 0:
    print(f'Total Duration: {total_duration}s ({total_duration / 60:.1f} minutes)')
print('=' * 60)
"
}

# =============================================================================
# BATCH SUMMARY FINALIZATION (called at end of batch)
# =============================================================================

finalize_batch_summary() {
    local batch_dir=$1
    local summary_file="${batch_dir}/batch_summary.json"

    $PYTHON_BIN -c "
import json
from datetime import datetime
from pathlib import Path

summary_path = Path('${summary_file}')
with open(summary_path, 'r') as f:
    summary = json.load(f)

summary['batch_end_time'] = datetime.utcnow().isoformat() + 'Z'

experiments = summary.get('experiments', {})
total = len(experiments)
successful = len([e for e in experiments.values() if e.get('status') == 'success'])
failed = len([e for e in experiments.values() if e.get('status') == 'failed'])
total_duration = sum(e.get('duration_seconds', 0) for e in experiments.values())

summary['statistics'] = {
    'total': total,
    'successful': successful,
    'failed': failed,
    'total_duration_seconds': total_duration
}

with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
"
}
