#!/usr/bin/env bash
# Submit one plan to a Vulcan L40S node (4 GPUs, one lane per GPU). Usage: submit_vulcan.sh PLAN_NAME OUT_NAME HOURS [CKPTS_DIR]
# (run on Vulcan; PLAN_NAME relative to code/experiments/sagemaker/plans)
set -euo pipefail
PLAN=$1; OUT=$2; HOURS=$3; CKPTS=${4:-}
R=/scratch/mayman/rlvr
sbatch --account=aip-machado --nodes=1 --gpus-per-node=l40s:4 --cpus-per-task=64 --mem=0 --time=${HOURS}:00:00 --job-name=$OUT \
  --output=$R/slurm/%x-%j.out <<SB
#!/bin/bash
module load StdEnv/2023 python/3.12.4 cuda/12.6
source $R/env29/bin/activate
cd $R/code/experiments/tamia
python run_plan.py $R/code/experiments/sagemaker/plans/$PLAN $R/results/$OUT --model $R/models/Qwen2.5-Math-1.5B ${CKPTS:+--ckpts $CKPTS}
SB
