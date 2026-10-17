#!/usr/bin/env bash
# Launch the Pre-registration 3 runs as single-GPU jobs spread over regions and instance sizes.
# Each line: run seed region instance spot(0/1)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
while read -r run seed region instance spot; do
  [ -z "${run}" ] && continue
  if [ "${spot}" = 1 ]; then s=1; else s=; fi
  PLAN="${HERE}/plans/${PREFIX:-v2}_s${seed}_${run}.json" ARMS_TAG="${PREFIX:-v2}-${run}" SEED="${seed}" REGION="${region}" SPOT="${s}" \
    PROFILE=${PROFILE:-stop-admin} bash "${HERE}/launch.sh" "${instance}" 50400 2>&1 | grep "started\|rror" | cut -c1-160
  sleep 1
done
