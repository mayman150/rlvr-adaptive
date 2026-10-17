#!/usr/bin/env bash
# One line per job: status, secondary status, last message. Usage: status.sh <file of region:job lines>
while IFS= read -r spec; do
  [ -z "${spec}" ] && continue
  r=${spec%%:*}; j=${spec#*:}
  printf '%-44s %s\n' "${j}" "$(aws sagemaker describe-training-job --training-job-name "${j}" --profile "${PROFILE:-stop-ro}" --region "${r}" \
    --query '[TrainingJobStatus,SecondaryStatus,SecondaryStatusTransitions[-1].StatusMessage,FailureReason]' --output text 2>&1 | tr '\t' ' ' | cut -c1-110)"
done < "$1"
