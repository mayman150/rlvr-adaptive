#!/usr/bin/env bash
# Pre-registration 40 pilot, evaluation: one Nibi node, 8 H100s, vLLM. For each run, every saved checkpoint on hard_val
# (64 samples) and MATH-511 (32 samples) at 3,072 tokens; plus the base model and both SFT starts (step 0).
# Usage (on Nibi): bash p40_eval_nibi.sh TRAIN_OUT HOURS [DEPENDENCY_JOBID]
set -euo pipefail
source "$(dirname "$0")/p40_lanes.sh"
OUTNAME=$1; HOURS=$2; DEP=${3:-}; OUT=$R/results/$OUTNAME
STEPSET=25,50,100,150,200,250,300
ev() {  # ev CKPTS_DIR EVAL_OUT [extra eval_vllm args...] -> one command per evaluation set
  local c=$1 o=$2; shift 2
  local pr n
  for pr_n in hard_val.json:64 eval_frontier_512.json:32; do
    pr=${pr_n%%:*}; n=${pr_n##*:}
    echo "python -I $T/eval_vllm.py --ckpts $c --out $o/eval_${pr%.json} --problems $pr --n $n --max-new 3072 --temperatures 1.0 --seed 40 $*"
  done
}
SCRIPT=$OUT/eval_job.sh
{
  echo "#!/bin/bash"
  echo "module load StdEnv/2023 gcc python/3.12.4 cuda/12.6 opencv/4.10.0"
  echo "source $R/envvllm/bin/activate"
  echo "export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8"
  echo "cd $T"
  g=0
  for L in $LANES; do
    cmds=$(ev $OUT/$L $OUT/$L --sampled-steps $STEPSET --expect-steps $STEPSET | paste -sd';' -)
    extra=""
    case $L in
      DRs8)   extra=";$(ev $OUT/none $OUT/BASE --base $MODEL --base-temperatures 1.0 --sampled-steps 0 --expect-steps 0 | paste -sd';' -)" ;;
      TSFTs8) extra=";$(ev $OUT/none $OUT/SFT_T_start --base $OUT/SFT_T/start_$SFT_DOSE --base-temperatures 1.0 --sampled-steps 0 --expect-steps 0 | paste -sd';' -)" ;;
      SSFTs8) extra=";$(ev $OUT/none $OUT/SFT_S_start --base $OUT/SFT_S/start_$SFT_DOSE --base-temperatures 1.0 --sampled-steps 0 --expect-steps 0 | paste -sd';' -)" ;;
    esac
    echo "( export CUDA_VISIBLE_DEVICES=$g; ${cmds}${extra} ) > $OUT/eval_$L.log 2>&1 &"
    g=$((g + 1))
  done
  echo "wait"
} > $SCRIPT
sbatch --account=def-machado --nodes=1 --gpus-per-node=h100:8 --cpus-per-task=64 --mem=0 --time=${HOURS}:00:00 \
  ${DEP:+--dependency=afterany:$DEP} --job-name=${OUTNAME}_eval --output=$R/slurm/%x-%j.out $SCRIPT
