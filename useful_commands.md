# Run batch

Run batch
- ./script/run_batch.sh [--local|--slurm] [--batch-name NAME]

Resume earlier run
- ./script/run_batch.sh --slurm --resume outputs/all_but_repetition1_20260119_205910_complete


# Visualize

Plot payoff tensors
- python src/visualize/plot_payoff_tensors.py data/clean/all_but_repetition1_20260119_205910_complete figures/all_but_repetition1/payoffs/
- python src/visualize/plot_payoff_tensors.py data/clean/all_but_rep_and_contr1_20260119_205910 outputs/contracting1_20260121_172329 outputs/repetition1_20260121_013429 figures/run1/

Print Latex Tables:
- python src/visualize/create_table.py outputs/all_but_repetition1_20260119_205910_complete --output figures/all_but_repetition1 [--no-stderr] [--metrics mean rd]
- python src/visualize/create_table.py data/clean/all_but_rep_and_contr1_20260119_205910 outputs/contracting1_20260121_172329 outputs/repetition1_20260121_013429 --output figures/run1/tables/ --no-stderr --metrics mean rd