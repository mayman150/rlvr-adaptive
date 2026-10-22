#!/usr/bin/env bash
# Submit one plan to a Tamia H100 node. Usage: submit.sh PLAN_NAME OUT_NAME HOURS [CKPTS_DIR]
# (run on Tamia from $SCRATCH/rlvr; PLAN_NAME relative to code/experiments/sagemaker/plans)
set -euo pipefail
PLAN=$1; OUT=$2; HOURS=$3; CKPTS=${4:-}
R=$SCRATCH/rlvr
sbatch --account=aip-machado --nodes=1 --gpus-per-node=h100:4 --time=${HOURS}:00:00 --job-name=$OUT \
  --output=$R/slurm/%x-%j.out <<SB
#!/bin/bash
module load StdEnv/2023 python/3.12.4 cuda/12.6
source $R/env/bin/activate
cd $R/code/experiments/tamia
python run_plan.py $R/code/experiments/sagemaker/plans/$PLAN $R/results/$OUT --model $R/models/Qwen2.5-Math-1.5B ${CKPTS:+--ckpts $CKPTS}
SB
