#!/usr/bin/env bash
# Package experiments/training, upload it, and start one SageMaker training job.
# Usage: PROFILE=stop-admin [REGION=us-east-1] [SPOT=1] SEED=1 ./launch.sh [instance-type] [max-runtime-seconds]
set -euo pipefail

ACCOUNT=165919938845
REGION=${REGION:-us-west-2}
BUCKET=stop-rl-${ACCOUNT}-${REGION}
ROLE_ARN=arn:aws:iam::${ACCOUNT}:role/StopRLSageMakerExecution
IMAGE=763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker
INSTANCE=${1:-ml.g6e.xlarge}
MAX_RUNTIME=${2:-14400}
PROFILE=${PROFILE:?set PROFILE to a profile with write access}
AWS="aws --profile ${PROFILE} --region ${REGION}"

HERE=$(cd "$(dirname "$0")" && pwd)
EXPERIMENTS=$(dirname "${HERE}")
JOB=worth-train-s${SEED:-1}-$(date -u +%Y%m%d-%H%M%S)
STAGE=$(mktemp -d)
trap 'rm -rf "${STAGE}"' EXIT

cp "${HERE}/entry.py" "${HERE}/requirements.txt" "${STAGE}/"
rsync -a --exclude results --exclude __pycache__ "${EXPERIMENTS}/training" "${STAGE}/"
tar -czf "${STAGE}.tar.gz" -C "${STAGE}" .
${AWS} s3 cp "${STAGE}.tar.gz" "s3://${BUCKET}/code/${JOB}/sourcedir.tar.gz"
rm -f "${STAGE}.tar.gz"

if [ -n "${SPOT:-}" ]; then
  SPOT_FLAGS=(--enable-managed-spot-training
              --stopping-condition "MaxRuntimeInSeconds=${MAX_RUNTIME},MaxWaitTimeInSeconds=$((MAX_RUNTIME + 28800))")
else
  SPOT_FLAGS=(--stopping-condition "MaxRuntimeInSeconds=${MAX_RUNTIME}")
fi

${AWS} sagemaker create-training-job --training-job-name "${JOB}" \
  --role-arn "${ROLE_ARN}" \
  --algorithm-specification "TrainingImage=${IMAGE},TrainingInputMode=File" \
  --hyper-parameters "{
    \"sagemaker_program\": \"entry.py\",
    \"sagemaker_submit_directory\": \"s3://${BUCKET}/code/${JOB}/sourcedir.tar.gz\",
    \"sagemaker_region\": \"${REGION}\",
    \"seed\": \"${SEED:-1}\"
  }" \
  --resource-config "InstanceType=${INSTANCE},InstanceCount=1,VolumeSizeInGB=60" \
  --output-data-config "S3OutputPath=s3://${BUCKET}/output/" \
  ${SPOT_FLAGS[@]+"${SPOT_FLAGS[@]}"} \
  --tags Key=project,Value=stop-rl

echo "started ${JOB} on ${INSTANCE}${SPOT:+ (spot)} (cap ${MAX_RUNTIME}s)"
echo "status: ${AWS} sagemaker describe-training-job --training-job-name ${JOB} --query TrainingJobStatus"
