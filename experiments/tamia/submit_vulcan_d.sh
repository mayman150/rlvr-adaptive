#!/usr/bin/env bash
# Vulcan (one 4-GPU L40S node): Pre-registration 18's AV seed 8 (training on GPU 0, then its two evaluation
# parts on GPUs 0 and 1) alongside Check 21's 1,024-token evaluations of HP and HQ (GPUs 1-3).
set -euo pipefail
R=/scratch/mayman/rlvr
P=$R/code/experiments/sagemaker/plans
M=$R/models/Qwen2.5-Math-1.5B
sbatch --account=aip-machado --nodes=1 --gpus-per-node=l40s:4 --cpus-per-task=64 --mem=0 --time=10:00:00 --job-name=vul_D \
  --output=$R/slurm/%x-%j.out <<SB
#!/bin/bash
module load StdEnv/2023 python/3.12.4 cuda/12.6
source $R/env29/bin/activate
cd $R/code/experiments/tamia
( python run_plan.py $P/v18_s8_AV.json $R/results/vul_D --model $M && \
  python run_plan.py $P/vul_D_eval.json $R/results/vul_D_eval --model $M --ckpts $R/results/vul_D --gpus 2 ) &
python run_plan.py $P/vul_T1024.json $R/results/vul_T1024 --model $M --ckpts $R/results --gpus 3 --first-gpu 1 &
wait
SB
