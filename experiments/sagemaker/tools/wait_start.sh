#!/bin/bash
# Report-only: exit when the given job (REGION JOB) leaves Pending; prints the secondary status of extra jobs (region:job args).
R=$1; J=$2; shift 2
for i in $(seq 1 240); do
  ada credentials update --account=165919938845 --provider=isengard --role=ReadOnly --profile=stop-ro --once >/dev/null 2>&1
  st=$(aws sagemaker describe-training-job --profile stop-ro --region $R --training-job-name $J --query SecondaryStatus --output text 2>/dev/null)
  if [ -n "$st" ] && [ "$st" != Pending ]; then
    echo "$(date -u) $J $st"
    for x in "$@"; do echo "${x#*:} $(aws sagemaker describe-training-job --profile stop-ro --region ${x%%:*} --training-job-name ${x#*:} --query SecondaryStatus --output text)"; done
    exit 0
  fi
  sleep 120
done
