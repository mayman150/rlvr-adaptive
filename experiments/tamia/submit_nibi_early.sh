#!/usr/bin/env bash
# Nibi: train the early-checkpoint Dr. GRPO runs, then evaluate them, in one 3-GPU job.
R=/scratch/mayman/rlvr
sbatch --account=def-machado_gpu --gpus=h100:3 --cpus-per-task=24 --mem=240G --time=3:00:00 --job-name=early --output=$R/slurm/%x-%j.out <<SB
#!/bin/bash
module load StdEnv/2023 python/3.12.4 cuda/12.6
source $R/env29/bin/activate
cd $R/code/experiments/tamia
python run_plan.py $R/code/experiments/sagemaker/plans/early_train.json $R/results/early_train --model $R/models/Qwen2.5-Math-1.5B
python run_plan.py $R/code/experiments/sagemaker/plans/early_eval.json $R/results/early_eval --model $R/models/Qwen2.5-Math-1.5B --ckpts $R/results/early_train
SB
