#!/usr/bin/env bash
# Quota-aware launcher. Usage: [CAP=seconds] try_launch.sh LISTFILE
# Each line: PLAN TAG SEED REGION CKPT_PREFIX(or -) JOBSFILE [SPOT_FIRST]; tries g6e sizes on demand, then spot.
cd ~/Desktop/adaptive-rlvr/experiments/sagemaker
ada credentials update --account=165919938845 --provider=isengard --role=Admin --profile=stop-admin --once >/dev/null 2>&1
while read -r plan tag seed region ckpt jobs spotfirst; do
  [ -z "$plan" ] && continue
  if [ "$spotfirst" = 1 ]; then order="1 0"; else order="0 1"; fi
  done=""
  for sp in $order; do
    for type in ml.g6e.xlarge ml.g6e.2xlarge ml.g6e.4xlarge ml.g6e.8xlarge ml.g6e.16xlarge ml.g6e.12xlarge; do
      if [ "$sp" = 1 ]; then s=1; else s=; fi
      if [ "$ckpt" = "-" ]; then c=; else c=s3://stop-rl-165919938845-$region/$ckpt/; fi
      out=$(PLAN=plans/$plan ARMS_TAG=$tag SEED=$seed REGION=$region SPOT=$s VOLUME=150 CKPT_S3=$c PROFILE=stop-admin bash launch.sh $type ${CAP:-50400} 2>&1 | grep "started\|rror" | cut -c1-120)
      if echo "$out" | grep -q started; then
        j=$(echo "$out" | grep -o "worth-$tag-s$seed-[0-9-]*"); echo "$region:$j" >> $jobs; echo "$tag s$seed: $out"; done=1; break 2
      fi
    done
  done
  [ -z "$done" ] && echo "$tag s$seed $region: no quota"
done < $1
