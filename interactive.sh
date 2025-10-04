srun --account=aip-rgrosse --time=1:00:00 --gres=gpu:l40s:2 --mem=48G -c 32 --chdir=/project/aip-rgrosse/xiao215/agent-tournament --pty bash
srun --account=aip-rgrosse --time=1:00:00 --gres=gpu:h100:2 --mem=48G -c 32 --chdir=/project/aip-rgrosse/xiao215/agent-tournament --pty bash
sinfo -N --Format=NodeHost,Partition,GresUsed,Gres,CPUsState
squeue -u $USER
srun --account=aip-rgrosse --time=1:00:00 --gres=gpu:l40s:2 --mem=80G -c 24 --chdir=/project/aip-rgrosse/xiao215/agent-tournament --pty bash

srun --account=aip-rgrosse --time=1:00:00 --gres=gpu:h100:1 --mem=60G -c 32 --chdir=/project/aip-rgrosse/xiao215/agent-tournament --pty bash

# Increase -c to load model faster

srun --account=aip-rgrosse --time=1:00:00 --mem=24G -c 32 --chdir=/project/aip-rgrosse/xiao215/agent-tournament --pty bash
