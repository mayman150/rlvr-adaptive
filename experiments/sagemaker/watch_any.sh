#!/usr/bin/env bash
# Exit as soon as any listed job (region:job lines in a file) that was running at start reaches
# a final state, printing it. Usage: watch_any.sh <file> [poll seconds]
set -uo pipefail
ACCOUNT=165919938845
FILE=$1; POLL=${2:-300}
state() { aws sagemaker describe-training-job --training-job-name "$2" --profile stop-ro --region "$1" \
  --query '[TrainingJobStatus,SecondaryStatus,FailureReason]' --output text 2>&1; }
declare -a live=()
while IFS= read -r spec; do
  [ -z "${spec}" ] && continue
  st=$(state "${spec%%:*}" "${spec#*:}")
  case "${st}" in InProgress*|*ExpiredToken*) live+=("${spec}");; esac
done < "${FILE}"
echo "$(date +%H:%M:%S) watching ${#live[@]} jobs"
while true; do
  for spec in "${live[@]}"; do
    st=$(state "${spec%%:*}" "${spec#*:}")
    case "${st}" in
      *ExpiredToken*|*InvalidClientTokenId*)
        ada credentials update --account="${ACCOUNT}" --provider=isengard --role=ReadOnly --profile=stop-ro --once >/dev/null 2>&1 || { echo "credential refresh failed"; exit 2; };;
      InProgress*) ;;
      *) echo "$(date +%H:%M:%S) ${spec}: ${st}"; exit 0;;
    esac
  done
  sleep "${POLL}"
done
