#!/bin/bash
# Report-only: exit when a line matching PATTERN (grep -E) appears in a SageMaker job's log, or the job ends.
# Usage: wait_line.sh REGION JOB PATTERN [max-minutes]
R=$1; J=$2; P=$3; M=${4:-600}
for i in $(seq 1 $((M / 5))); do
  bash $(dirname $0)/lane_tail.sh $R $J 1 >/dev/null 2>&1
  hit=$(grep -E "$P" /tmp/lane_tail_$J.log | head -5)
  if [ -n "$hit" ]; then echo "$(date -u)"; echo "$hit" | cut -c1-300; exit 0; fi
  st=$(aws sagemaker describe-training-job --profile stop-ro --region $R --training-job-name $J --query TrainingJobStatus --output text 2>/dev/null)
  [ -n "$st" ] && [ "$st" != InProgress ] && { echo "$(date -u) job $st"; exit 0; }
  sleep 300
done
