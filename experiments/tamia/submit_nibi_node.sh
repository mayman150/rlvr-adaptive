#!/usr/bin/env bash
# Nibi: train a plan on one whole 8-GPU H100 node, then evaluate it, in one job.
# Usage (on Nibi): submit_nibi_node.sh TRAIN_PLAN EVAL_PLAN OUT_NAME HOURS
set -euo pipefail
TRAIN=$1; EVAL=$2; OUT=$3; HOURS=$4
R=/scratch/mayman/rlvr
P=$R/code/experiments/sagemaker/plans
sbatch ${TEST_ONLY:+--test-only} --account=def-machado --nodes=1 --gpus-per-node=h100:8 --cpus-per-task=64 --mem=0 --time=${HOURS}:00:00 \
  --job-name=$OUT --output=$R/slurm/%x-%j.out <<SB
#!/bin/bash
module load StdEnv/2023 python/3.12.4 cuda/12.6
source $R/env29/bin/activate
cd $R/code/experiments/tamia
python run_plan.py $P/$TRAIN $R/results/$OUT --model $R/models/Qwen2.5-Math-1.5B --gpus 8
python run_plan.py $P/$EVAL $R/results/${OUT}_eval --model $R/models/Qwen2.5-Math-1.5B --ckpts $R/results/$OUT --gpus 8
SB
