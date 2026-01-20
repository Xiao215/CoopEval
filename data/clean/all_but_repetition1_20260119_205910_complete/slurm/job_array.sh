#!/bin/bash
# =============================================================================
# SLURM CONFIGURATION - Edit these values for your cluster
# =============================================================================
#SBATCH --job-name=LLM_tournament
#SBATCH --partition=cpu_opteron6272
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --exclude=marvel-0-[15-29],marvel-1-[7,13]

# Uncomment to enable email notifications:
# #SBATCH --mail-user=your.email@example.com
# #SBATCH --mail-type=END,FAIL

# Note: No GPUs needed - experiments are I/O bound (LLM API calls)
# The parallelism (EXPERIMENT_WORKERS, TOURNAMENT_WORKERS) uses threads
# for concurrent API calls, which don't require multiple CPU cores

# =============================================================================
# END SLURM CONFIGURATION
# =============================================================================

#SBATCH --chdir=/marvel/home/etewolde/agent-tournament

echo "=========================================="
echo "SLURM Job Array Task"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Batch Directory: /marvel/home/etewolde/agent-tournament/outputs/all_but_repetition1_20260119_205910"
echo "=========================================="
echo ""

# Run single experiment worker
/marvel/home/etewolde/agent-tournament/script/run_single_experiment.sh "$SLURM_ARRAY_TASK_ID" "/marvel/home/etewolde/agent-tournament/outputs/all_but_repetition1_20260119_205910"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID finished with exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
