#!/usr/bin/env bash
# Pre-registration 40: smoke test on one Nibi H100 (<= 2 hours): 3 RL steps each of the teacher-hint and tree lanes at
# 3,072 tokens (step time, memory, logs), then a tiny vLLM evaluation of the base. Usage (on Nibi): bash p40_smoke_nibi.sh
set -euo pipefail
source "$(dirname "$0")/p40_lanes.sh"
OUT=$R/results/p40_smoke; mkdir -p $OUT $R/slurm
sbatch --account=def-machado --gpus-per-node=h100:1 --cpus-per-task=12 --mem=120G --time=2:00:00 --job-name=p40_smoke \
  --output=$R/slurm/%x-%j.out <<SB
#!/bin/bash
module load StdEnv/2023 gcc python/3.12.4 cuda/12.6 opencv/4.13.0
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 OUT=$OUT STEPS=3
cd $T
source $R/env29/bin/activate
STEPS=3 bash -c "$(STEPS=3; source $(dirname "$0")/p40_lanes.sh; lane_cmd THINTs8 | sed 's/--steps 300/--steps 3/; s/--eval-steps 300/--eval-steps 3/; s/--save-steps [0-9,]*/--save-steps 3/')" > $OUT/THINT.log 2>&1; echo "THINT exit \$?"
STEPS=3 bash -c "$(STEPS=3; source $(dirname "$0")/p40_lanes.sh; lane_cmd TREEs8 | sed 's/--steps 300/--steps 3/; s/--eval-steps 300/--eval-steps 3/; s/--save-steps [0-9,]*/--save-steps 3/')" > $OUT/TREE.log 2>&1; echo "TREE exit \$?"
deactivate; source $R/envvllm/bin/activate
python -I $T/eval_vllm.py --ckpts $OUT/none --base $MODEL --out $OUT/eval --problems hard_val.json --limit 20 --n 8 --max-new 3072 \
  --temperatures 1.0 --base-temperatures 1.0 --sampled-steps 0 --expect-steps 0 --seed 40 > $OUT/eval.log 2>&1; echo "EVAL exit \$?"
SB
