#!/usr/bin/env bash
# Pre-registration 40 pilot: lane definitions (sourced by p40_train_nibi.sh). Each lane: optional SFT start, then RL.
R=/scratch/mayman/rlvr
T=$R/code/experiments/training
MODEL=$R/models/Qwen2.5-Math-1.5B
STEPS=${STEPS:-300}
RL_COMMON="--arms baseline --group 8 --prompts-per-step 16 --max-new 3072 --lr 1e-6 --gen-batch 128 --calib-prompts 0 --w 1 \
 --entropy-every 5 --train train.json --hard-problems p40_guided.json --hard-per-step 4 --eval-steps $STEPS --eval-limit 64 \
 --save-steps 25,50,100,150,200,250,300 --steps $STEPS"
HINT="--hint-levels 4 --hint-success 0.5 --hint-withdraw-step 200 --hint-max-frac 0.75"
SFT_DOSE=71000  # just under one pass over each bank (71,145 teacher, 71,252 self presented tokens)
lane_cmd() {  # lane_cmd NAME -> the shell commands for that lane, given OUT
  case $1 in
    DRs8)    echo "python $T/train_grpo.py --device cuda --model $MODEL --seed 8 $RL_COMMON --out \$OUT/DRs8" ;;
    DRs9)    echo "python $T/train_grpo.py --device cuda --model $MODEL --seed 9 $RL_COMMON --out \$OUT/DRs9" ;;
    TREEs8)  echo "python $T/train_grpo.py --device cuda --model $MODEL --seed 8 $RL_COMMON --tree-branches 4 --out \$OUT/TREEs8" ;;
    THINTs8) echo "python $T/train_grpo.py --device cuda --model $MODEL --seed 8 $RL_COMMON $HINT --hint-archive p40_bank_teacher.json --out \$OUT/THINTs8" ;;
    THINTs9) echo "python $T/train_grpo.py --device cuda --model $MODEL --seed 9 $RL_COMMON $HINT --hint-archive p40_bank_teacher.json --out \$OUT/THINTs9" ;;
    SHINTs8) echo "python $T/train_grpo.py --device cuda --model $MODEL --seed 8 $RL_COMMON $HINT --hint-archive p40_bank_self.json --out \$OUT/SHINTs8" ;;
    TSFTs8)  echo "python $T/sft_start.py --model $MODEL --archive p40_sft_teacher.json --problems p40_guided.json --seed 3030 --milestones $SFT_DOSE --max-new 3072 --out \$OUT/SFT_T && python $T/train_grpo.py --device cuda --model \$OUT/SFT_T/start_$SFT_DOSE --seed 8 $RL_COMMON --out \$OUT/TSFTs8" ;;
    SSFTs8)  echo "python $T/sft_start.py --model $MODEL --archive p40_sft_self.json --problems p40_guided.json --seed 3030 --milestones $SFT_DOSE --max-new 3072 --out \$OUT/SFT_S && python $T/train_grpo.py --device cuda --model \$OUT/SFT_S/start_$SFT_DOSE --seed 8 $RL_COMMON --out \$OUT/SSFTs8" ;;
  esac
}
LANES="DRs8 DRs9 TREEs8 THINTs8 THINTs9 TSFTs8 SHINTs8 SSFTs8"
