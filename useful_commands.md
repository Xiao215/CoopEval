# Run batch

Run batch
- ./script/run_batch.sh [--local|--slurm] [--batch-name NAME]

Resume earlier run
- ./script/run_batch.sh --slurm --resume outputs/all_but_repetition1_20260119_205910_complete


# Visualize

Plot payoff tensors
- python src/visualize/plot_payoff_tensors.py data/clean/all_but_repetition1_20260119_205910_complete figures/all_but_repetition1/payoffs/

Print Latex Tables:
- python src/visualize/create_table.py outputs/all_but_repetition1_20260119_205910_complete --output figures/all_but_repetition1 [--no-stderr] [--metrics mean rd]