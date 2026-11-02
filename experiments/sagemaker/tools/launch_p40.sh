#!/usr/bin/env bash
# Pre-registration 40 pilot on AWS: one ml.p4de.24xlarge (8 x A100) in us-west-2; 8 lanes (training then vLLM evaluation
# of every checkpoint), cap 60 hours (at most about $1,900; planned about $1,600).
set -euo pipefail
cd "$(dirname "$0")/.."
ada credentials update --account=165919938845 --provider=isengard --role=Admin --profile=stop-admin --once
out=$(PROFILE=stop-admin REGION=us-west-2 PLAN=plans/p40_pilot.json ARMS_TAG=p40-pilot SEED=0 VOLUME=500 bash ./launch.sh ml.p4de.24xlarge 216000 2>&1)
echo "$out" | grep -E "started|rror"
echo "us-west-2:$(echo "$out" | grep -o 'worth-p40-pilot-s0-[0-9-]*' | head -1)" > p40_job.txt
cat p40_job.txt
