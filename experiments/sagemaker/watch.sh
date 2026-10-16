#!/usr/bin/env bash
# Watch jobs until each reaches a terminal state; print status changes.
# Usage: bash watch.sh region:job [region:job ...]
# Exits early, with a message, if read-only credentials expire and cannot be refreshed.
# Written for bash 3.2 (macOS): indexed arrays only.
set -uo pipefail
ACCOUNT=165919938845
specs=("$@")
n=${#specs[@]}
last=(); finished=()
for ((i = 0; i < n; i++)); do last[i]=""; finished[i]=0; done

while :; do
  pending=0
  for ((i = 0; i < n; i++)); do
    [ "${finished[i]}" = 1 ] && continue
    spec=${specs[i]}; region=${spec%%:*}; job=${spec#*:}
    s=$(aws sagemaker describe-training-job --training-job-name "${job}" --profile stop-ro \
      --region "${region}" --query '[TrainingJobStatus,SecondaryStatus,FailureReason]' --output text 2>&1)
    case "${s}" in
      *ExpiredToken*|*InvalidClientTokenId*)
        if ! ada credentials update --account="${ACCOUNT}" --provider=isengard --role=ReadOnly \
             --profile=stop-ro --once >/dev/null 2>&1; then
          echo "$(date +%H:%M:%S) credentials expired and refresh failed; run mwinit, then restart the watcher"
          exit 2
        fi
        pending=1; continue;;
    esac
    if [ "${s}" != "${last[i]}" ]; then
      echo "$(date +%H:%M:%S) ${job}: $(printf '%s' "${s}" | head -c 300)"
      last[i]="${s}"
    fi
    case "${s}" in
      Completed*|Failed*|Stopped*) finished[i]=1;;
      *) pending=1;;
    esac
  done
  [ "${pending}" = 0 ] && break
  sleep 300
done
echo "all jobs finished"
