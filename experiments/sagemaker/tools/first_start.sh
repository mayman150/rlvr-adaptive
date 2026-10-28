#!/bin/bash
# Keep the first of several duplicate jobs (region:job lines in FILE) that leaves Pending; stop the others while they
# are still Pending (never a job that has started). Prints the kept job.
F=$1
for i in $(seq 1 360); do
  ada credentials update --account=165919938845 --provider=isengard --role=Admin --profile=stop-admin --once >/dev/null 2>&1
  kept=""
  while IFS=: read r j; do [ -z "$j" ] && continue
    s=$(aws sagemaker describe-training-job --profile stop-admin --region $r --training-job-name $j --query SecondaryStatus --output text 2>/dev/null)
    if [ -n "$s" ] && [ "$s" != Pending ] && [ "$s" != Stopping ] && [ "$s" != Stopped ] && [ "$s" != MaxWaitTimeExceeded ]; then kept="$r:$j"; break; fi
  done < $F
  if [ -n "$kept" ]; then
    while IFS=: read r j; do [ -z "$j" ] && continue; [ "$r:$j" = "$kept" ] && continue
      s=$(aws sagemaker describe-training-job --profile stop-admin --region $r --training-job-name $j --query SecondaryStatus --output text 2>/dev/null)
      [ "$s" = Pending ] && aws sagemaker stop-training-job --profile stop-admin --region $r --training-job-name $j && echo "stopped pending duplicate $j"
      [ "$s" != Pending ] && [ "$s" != Stopped ] && echo "WARNING: duplicate $j is $s (not stopped)"
    done < $F
    echo "$(date -u) kept $kept"; exit 0
  fi
  sleep 120
done
