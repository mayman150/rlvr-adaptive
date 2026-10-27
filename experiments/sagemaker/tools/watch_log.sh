#!/bin/bash
# Report-only: exit when a SageMaker job's log shows a first evaluation line, an error, or the job ends.
# Usage: watch_log.sh REGION JOB
R=$1; J=$2
for i in $(seq 1 120); do
  ada credentials update --account=165919938845 --provider=isengard --role=ReadOnly --profile=stop-ro --once >/dev/null 2>&1
  st=$(aws sagemaker describe-training-job --profile stop-ro --region $R --training-job-name $J --query TrainingJobStatus --output text 2>/dev/null) || { sleep 60; continue; }
  s=$(aws logs describe-log-streams --profile stop-ro --region $R --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name-prefix $J --query 'logStreams[0].logStreamName' --output text 2>/dev/null)
  if [ -n "$s" ] && [ "$s" != None ]; then
    out=$(aws logs get-log-events --profile stop-ro --region $R --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name $s --start-from-head --limit 10000 --query 'events[].message' --output text | tr '\t' '\n' | grep -E "Traceback|Error:|: pass@1|exit [1-9]|exception")
    if [ -n "$out" ]; then echo "$(date -u) status $st"; echo "$out" | head -20 | cut -c1-250; exit 0; fi
  fi
  [ "$st" != InProgress ] && { echo "$(date -u) job $st"; exit 0; }
  sleep 120
done
