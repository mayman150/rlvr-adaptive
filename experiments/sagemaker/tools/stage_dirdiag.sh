#!/bin/bash
# Server-side copy of the checkpoints the direction diagnostic needs into ckpt-sets/dirdiag/ (us-west-2).
ada credentials update --account=165919938845 --provider=isengard --role=Admin --profile=stop-admin --once >/dev/null 2>&1
W=s3://stop-rl-165919938845-us-west-2/ckpt-sets
for r in B1k8 PSR1k8 NSR1k8 AC1k8 HG1k8 HGm1k8 PG1k8 MH8 AVH8; do
  if [ $r = B1k8 ]; then inc="ckpt_step*"; else inc="ckpt_step150/*"; fi
  aws s3 sync --quiet --profile stop-admin --region us-west-2 $W/v15/$r/ $W/dirdiag/$r/ --exclude "*" --include "$inc" && echo "copied $r"
done
