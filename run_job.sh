#!/bin/bash
#SBATCH --job-name=LLM_evolution_tournament
#SBATCH --account=aip-rgrosse
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --chdir=/project/aip-rgrosse/xiao215/agent-tournament
#SBATCH --mail-user=xiaoo.zhang@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL

# Run
source .venv312/bin/activate
export PYTHONPATH=.

python3 script/run_evolution.py --config legacy/prisoner_dilemma.yaml

# python3 script/run_evolution.py --config legacy/public_goods.yaml --wandb
#
# python3 script/run_evolution.py --config legacy/public_goods_toy.yaml --wandb

# python3 script/run_evolution.py --config legacy/prisoner_dilemma_io_vs_cot.yaml --wandb
# python3 script/run_evolution.py --config legacy/toy_pd.yaml
# python3 script/run_evolution.py --config legacy/toy_disarm.yaml
# python3 script/run_evolution.py --config legacy/toy_mediation.yaml

# python3 script/run_evolution.py --config legacy/toy_pd_client_api.yaml
# python3 script/run_evolution.py --config test/repetition/repetition_pd.yaml