#!/usr/bin/env bash
# Pre-registration 39 (Stage 0): one ml.p4de.24xlarge job in us-east-1, generation and evaluation only (no training),
# about 7 hours (cap 10 hours), about $230. Checkpoint channel: p39-stage0-ckpts (45 GB, copied from our own runs).
set -euo pipefail
cd "$(dirname "$0")/.."
ada credentials update --account=165919938845 --provider=isengard --role=Admin --profile=stop-admin --once
out=$(PROFILE=stop-admin REGION=us-east-1 PLAN=plans/p39_stage0.json ARMS_TAG=p39-stage0 SEED=0 VOLUME=250 \
      CKPT_S3=s3://stop-rl-165919938845-us-east-1/checkpoints/p39-stage0-ckpts/ bash ./launch.sh ml.p4de.24xlarge 36000 2>&1)
echo "$out" | grep -E "started|rror"
echo "us-east-1:$(echo "$out" | grep -o 'worth-p39-stage0-s0-[0-9-]*' | head -1)" > p39_stage0_job.txt
cat p39_stage0_job.txt
