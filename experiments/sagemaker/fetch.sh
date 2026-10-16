#!/usr/bin/env bash
# Download a finished job's results into experiments/training/results/<job>.
# Usage: PROFILE=stop-ro ./fetch.sh <job-name>
set -euo pipefail

ACCOUNT=165919938845
REGION=${REGION:-us-west-2}
BUCKET=stop-rl-${ACCOUNT}-${REGION}
JOB=${1:?job name}
PROFILE=${PROFILE:?set PROFILE}
HERE=$(cd "$(dirname "$0")" && pwd)
DEST="$(dirname "${HERE}")/training/results/${JOB}"

mkdir -p "${DEST}"
aws --profile "${PROFILE}" --region "${REGION}" s3 cp \
  "s3://${BUCKET}/output/${JOB}/output/model.tar.gz" "${DEST}/model.tar.gz"
tar -xzf "${DEST}/model.tar.gz" -C "${DEST}" && rm "${DEST}/model.tar.gz"
echo "results in ${DEST}"
