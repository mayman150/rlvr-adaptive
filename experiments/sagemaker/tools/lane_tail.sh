#!/bin/bash
# Report-only: the last N lines of each lane prefix ([name#kind#part#laneG]) in a SageMaker job's log.
# Usage: lane_tail.sh REGION JOB [N]
R=$1; J=$2; N=${3:-3}
ada credentials update --account=165919938845 --provider=isengard --role=ReadOnly --profile=stop-ro --once >/dev/null 2>&1
s=$(aws logs describe-log-streams --profile stop-ro --region $R --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name-prefix $J --query 'logStreams[0].logStreamName' --output text)
tok=""; : > /tmp/lane_tail.log
while :; do
  if [ -z "$tok" ]; then res=$(aws logs get-log-events --profile stop-ro --region $R --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name $s --start-from-head --output json)
  else res=$(aws logs get-log-events --profile stop-ro --region $R --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name $s --next-token "$tok" --output json); fi
  echo "$res" | python3 -c "import sys,json; [print(e['message']) for e in json.load(sys.stdin)['events']]" >> /tmp/lane_tail.log
  nt=$(echo "$res" | python3 -c "import sys,json; print(json.load(sys.stdin)['nextForwardToken'])")
  [ "$nt" = "$tok" ] && break; tok=$nt
done
for p in $(grep -o '^\[[^]]*#lane[0-9]\]' /tmp/lane_tail.log | sort -u); do
  echo "== $p"; grep -F "$p" /tmp/lane_tail.log | grep -v "INFO\|WARNING\|it/s\]" | tail -n $N | cut -c1-260
done
grep -E "Traceback|Error|exception|exit [1-9]" /tmp/lane_tail.log | grep -v "INFO" | head -10 | cut -c1-260
