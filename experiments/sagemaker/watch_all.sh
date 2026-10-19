#!/usr/bin/env bash
# Exit when every listed job that is not Stopped has reached a final state; print each final state.
# Usage: watch_all.sh <file of region:job lines> [poll seconds]
set -uo pipefail
FILE=$1; POLL=${2:-600}
state() { aws sagemaker describe-training-job --training-job-name "$2" --profile stop-ro --region "$1" \
  --query '[TrainingJobStatus,SecondaryStatus]' --output text 2>&1; }
while true; do
  left=0
  while IFS= read -r spec; do
    [ -z "${spec}" ] && continue
    st=$(state "${spec%%:*}" "${spec#*:}")
    case "${st}" in
      *ExpiredToken*|*InvalidClientTokenId*) ada credentials update --account=165919938845 --provider=isengard --role=ReadOnly --profile=stop-ro --once >/dev/null 2>&1; left=$((left+1));;
      InProgress*) left=$((left+1));;
    esac
  done < "${FILE}"
  [ "${left}" -eq 0 ] && break
  sleep "${POLL}"
done
while IFS= read -r spec; do [ -n "${spec}" ] && echo "${spec#*:}: $(state "${spec%%:*}" "${spec#*:}")"; done < "${FILE}"
