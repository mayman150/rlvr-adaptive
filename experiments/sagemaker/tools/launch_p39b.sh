#!/usr/bin/env bash
# Pre-registration 39, Amendment 1 (Stage 0b): one ml.p4de.24xlarge job in us-east-1; generation, evaluation and smoke
# tests only (no science training), about 3.5 hours (cap 6 hours), about $110.
set -euo pipefail
cd "$(dirname "$0")/.."
ada credentials update --account=165919938845 --provider=isengard --role=Admin --profile=stop-admin --once
out=$(PROFILE=stop-admin REGION=us-east-1 PLAN=plans/p39b_addon.json ARMS_TAG=p39b-addon SEED=0 VOLUME=300 \
      CKPT_S3=s3://stop-rl-165919938845-us-east-1/checkpoints/p39-stage0-ckpts/ bash ./launch.sh ml.p4de.24xlarge 21600 2>&1)
echo "$out" | grep -E "started|rror"
echo "us-east-1:$(echo "$out" | grep -o 'worth-p39b-addon-s0-[0-9-]*' | head -1)" > p39b_job.txt
cat p39b_job.txt
