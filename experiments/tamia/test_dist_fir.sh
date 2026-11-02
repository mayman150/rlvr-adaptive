#!/usr/bin/env bash
# Fir: smoke tests of train_dist.py on one 4-GPU H100 node. (0) Gradient and AdamW-update parity with
# train_grpo.py on frozen rollouts (dist_parity.py) at 1, 2 and 3 ranks (3: uneven loads); (1) Qwen2.5-Math-1.5B, Dr. GRPO recipe of the
# 1.5B experiments on 2 GPUs, 10 steps (compare with a single-GPU baseline_train.json); (2) the same with
# allocation (law value); (3) Qwen2.5-Math-7B at 2,048 tokens on 4 GPUs, 2 steps each with fp32 and bf16
# FSDP parameters (memory and step time).
R=/scratch/mayman/rlvr
sbatch --account=def-machado --nodes=1 --gpus-per-node=h100:4 --cpus-per-task=32 --mem=400G --time=1:45:00 \
  --job-name=dist_smoke --output=$R/slurm/%x-%j.out <<SB
#!/bin/bash
module load StdEnv/2023 python/3.12.4 cuda/12.6
source $R/env29/bin/activate
cd $R/code/experiments/training
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
for n in 1 2 3; do
  torchrun --standalone --nproc-per-node \$n dist_parity.py --model $R/models/Qwen2.5-Math-1.5B --out $R/results/parity_w\$n.json
done
COMMON="--seed 8 --prompts-per-step 16 --group 8 --lr 1e-6 --train train.json --arms baseline"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 train_dist.py --model $R/models/Qwen2.5-Math-1.5B \
  \$COMMON --steps 10 --max-new 512 --gen-batch 64 --train-batch 4 --save-steps 10 --out $R/results/smoke_B15 &
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc-per-node 2 train_dist.py --model $R/models/Qwen2.5-Math-1.5B \
  \$COMMON --steps 10 --max-new 512 --gen-batch 64 --train-batch 4 --alloc-first 4 --alloc-cap 32 --alloc-value law \
  --out $R/results/smoke_AC15 &
wait
for dt in fp32 bf16; do
  torchrun --standalone --nproc-per-node 4 train_dist.py --model $R/models/Qwen2.5-Math-7B --param-dtype \$dt \
    \$COMMON --steps 2 --max-new 2048 --gen-batch 32 --train-batch 2 --save-steps 2 --out $R/results/smoke_B7_\$dt
done
SB
