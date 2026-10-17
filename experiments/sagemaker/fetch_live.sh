#!/usr/bin/env bash
# Sync the JSON results that running (or finished) plan jobs write to their S3 checkpoint
# prefix into experiments/training/results/<job>. Usage: fetch_live.sh <file of region:job lines>
set -uo pipefail
ACCOUNT=165919938845
HERE=$(cd "$(dirname "$0")" && pwd)
while IFS= read -r spec; do
  [ -z "${spec}" ] && continue
  r=${spec%%:*}; j=${spec#*:}
  aws --profile "${PROFILE:-stop-ro}" --region "${r}" s3 sync --quiet \
    "s3://stop-rl-${ACCOUNT}-${r}/checkpoints/${j}/" "$(dirname "${HERE}")/training/results/${j}/" \
    --exclude "*" --include "*.json"
done < "$1"
