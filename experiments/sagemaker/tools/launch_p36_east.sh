#!/usr/bin/env bash
# Relaunch P36 jobs 1-4 (ml.p4de.24xlarge) and the SmolLM3 viability evaluation (ml.g6e.2xlarge) in us-east-1,
# after their us-west-2 copies were stopped while still waiting for capacity (none ever started).
# Jobs 2-3 read the four P35 SFT starts copied to s3://stop-rl-165919938845-us-east-1/checkpoints/p35-sft-starts/.
set -euo pipefail
cd "$(dirname "$0")/.."
ada credentials update --account=165919938845 --provider=isengard --role=Admin --profile=stop-admin --once
CK=s3://stop-rl-165919938845-us-east-1/checkpoints/p35-sft-starts/
: > p36_jobs.txt
for k in 1 2 3 4; do
  if [ "$k" = 2 ] || [ "$k" = 3 ]; then ck=$CK; else ck=""; fi
  out=$(PROFILE=stop-admin REGION=us-east-1 PLAN=plans/p36_job$k.json ARMS_TAG=p36-job$k SEED=0 VOLUME=400 CKPT_S3=$ck \
        bash ./launch.sh ml.p4de.24xlarge 36000 2>&1)
  echo "$out" | grep -E "started|rror"
  j=$(echo "$out" | grep -o "worth-p36-job$k-s0-[0-9-]*" | head -1)
  [ -n "$j" ] && echo "us-east-1:$j" >> p36_jobs.txt
done
out=$(PROFILE=stop-admin REGION=us-east-1 PLAN=plans/smollm3_viability.json ARMS_TAG=smollm3-viab SEED=0 VOLUME=80 \
      bash ./launch.sh ml.g6e.2xlarge 7200 2>&1)
echo "$out" | grep -E "started|rror"
echo "us-east-1:$(echo "$out" | grep -o 'worth-smollm3-viab-s0-[0-9-]*' | head -1)" > smollm3_viab_job.txt
cat p36_jobs.txt smollm3_viab_job.txt
