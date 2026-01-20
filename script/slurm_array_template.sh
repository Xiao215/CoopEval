#!/bin/bash
# =============================================================================
# SLURM CONFIGURATION - Edit these values for your cluster
# =============================================================================
#SBATCH --job-name=LLM_tournament
#SBATCH --account=aip-rgrosse
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G

# Uncomment and edit if GPUs are needed:
# #SBATCH --gres=gpu:l40s:2
# #SBATCH --gres=gpu:h100:1

# Uncomment to enable email notifications:
# #SBATCH --mail-user=your.email@example.com
# #SBATCH --mail-type=END,FAIL

# =============================================================================
# END SLURM CONFIGURATION
# =============================================================================

#SBATCH --chdir={{PROJECT_ROOT}}

echo "=========================================="
echo "SLURM Job Array Task"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Batch Directory: {{BATCH_DIR}}"
echo "=========================================="
echo ""

# Run single experiment worker
{{SCRIPT_DIR}}/run_single_experiment.sh "$SLURM_ARRAY_TASK_ID" "{{BATCH_DIR}}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID finished with exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
