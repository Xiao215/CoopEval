# Run batch

Run batch
- ./script/run_batch.sh [--local|--slurm] [--batch-name NAME]

Resume earlier run
- ./script/run_batch.sh --slurm --resume outputs/all_but_repetition1_20260119_205910_complete

# Rerun evals

- python script/rerun_evaluations.py --batch-folders data/clean/all_but_rep_and_contr1_20260119_205910 data/clean/contracting1_20260121_172329 data/clean/repetition1_20260121_013429 data/clean/reputationfirstorder1_20260125_182942 data/clean/run2_20260124_025842 data/clean/run3_20260125_095939 --evaluation-config evaluation/default_evaluation.yaml --seed 42

# Visualize

Plot payoff tensors
- python src/visualize/plot_payoff_tensors.py data/clean/all_but_repetition1_20260119_205910_complete figures/all_but_repetition1/payoff_tensors/
- python src/visualize/plot_payoff_tensors.py data/clean/all_but_rep_and_contr1_20260119_205910 data/clean/contracting1_20260121_172329 data/clean/repetition1_20260121_013429 data/clean/reputationfirstorder1_20260125_182942 data/clean/run2_20260124_025842 data/clean/run3_20260125_095939 figures/allruns/payoff_tensors/

Print Latex Tables:
- python src/visualize/create_table.py outputs/all_but_repetition1_20260119_205910_complete --output figures/all_but_repetition1 [--no-stderr] [--metrics mean rd]
- python src/visualize/create_table.py data/clean/all_but_rep_and_contr1_20260119_205910 data/clean/contracting1_20260121_172329 data/clean/repetition1_20260121_013429 data/clean/reputationfirstorder1_20260125_182942 data/clean/run2_20260124_025842 data/clean/run3_20260125_095939 --output figures/allruns/tables/ --no-stderr --metrics mean rd dr