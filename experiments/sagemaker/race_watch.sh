#!/usr/bin/env bash
# Keep the first of several queued jobs to get an instance and stop the rest.
# Usage: bash race_watch.sh region:job [region:job ...]
set -uo pipefail
ACCOUNT=165919938845

describe() {
  aws sagemaker describe-training-job --training-job-name "${2}" --profile stop-ro \
    --region "${1}" --query '[TrainingJobStatus,SecondaryStatus]' --output text 2>&1
}
refresh() {
  ada credentials update --account="${ACCOUNT}" --provider=isengard --role="$1" \
    --profile="$2" --once >/dev/null 2>&1
}

while true; do
  winner=""
  for spec in "$@"; do
    region=${spec%%:*}; job=${spec#*:}
    status=$(describe "${region}" "${job}")
    case "${status}" in
      *ExpiredToken*|*InvalidClientTokenId*) refresh ReadOnly stop-ro; status=$(describe "${region}" "${job}");;
    esac
    case "${status}" in
      InProgress*Pending*|InProgress*Starting*) ;;
      InProgress*|Completed*) winner="${spec}"; echo "$(date +%H:%M:%S) ${spec}: ${status}"; break;;
      *) echo "$(date +%H:%M:%S) ${spec}: ${status}";;
    esac
  done
  if [ -n "${winner}" ]; then
    refresh Admin stop-admin
    for spec in "$@"; do
      [ "${spec}" = "${winner}" ] && continue
      region=${spec%%:*}; job=${spec#*:}
      aws sagemaker stop-training-job --training-job-name "${job}" --profile stop-admin \
        --region "${region}" >/dev/null 2>&1 && echo "stopped ${spec}"
    done
    echo "WINNER ${winner}"
    exit 0
  fi
  sleep 60
done
