#!/usr/bin/env bash
# Pre-registration 40 pilot, training: one Nibi node, 8 H100s, one lane per GPU. Usage (on Nibi): bash p40_train_nibi.sh OUT HOURS
set -euo pipefail
source "$(dirname "$0")/p40_lanes.sh"
OUTNAME=$1; HOURS=$2; OUT=$R/results/$OUTNAME
mkdir -p $OUT $R/slurm
SCRIPT=$OUT/job.sh
{
  echo "#!/bin/bash"
  echo "module load StdEnv/2023 python/3.12.4 cuda/12.6"
  echo "source $R/env29/bin/activate"
  echo "export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 OUT=$OUT"
  echo "cd $T"
  g=0
  for L in $LANES; do
    echo "( CUDA_VISIBLE_DEVICES=$g bash -c \"$(lane_cmd $L)\" > $OUT/$L.log 2>&1; echo \"$L exit \$?\" >> $OUT/status.txt ) &"
    g=$((g + 1))
  done
  echo "wait"
} > $SCRIPT
sbatch --account=def-machado --nodes=1 --gpus-per-node=h100:8 --cpus-per-task=64 --mem=0 --time=${HOURS}:00:00 \
  --job-name=$OUTNAME --output=$R/slurm/%x-%j.out $SCRIPT
