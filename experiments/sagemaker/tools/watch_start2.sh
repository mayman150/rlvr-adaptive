#!/bin/bash
# Report-only: exit as soon as either of two jobs (region:job region:job) is downloading or training.
for i in $(seq 1 400); do
  ada credentials update --account=165919938845 --provider=isengard --role=ReadOnly --profile=stop-ro --once >/dev/null 2>&1
  a=$(aws sagemaker describe-training-job --profile stop-ro --region ${1%%:*} --training-job-name ${1#*:} --query SecondaryStatus --output text)
  b=$(aws sagemaker describe-training-job --profile stop-ro --region ${2%%:*} --training-job-name ${2#*:} --query SecondaryStatus --output text)
  case "$a $b" in *Training*|*Downloading*) echo "$(date -u) $1: $a | $2: $b"; exit 0;; esac
  sleep 60
done
