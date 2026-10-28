#!/bin/bash
# Server-side copy of the checkpoints the packed P26 job needs into ckpt-sets/p26pack/ (us-west-2):
# the direction-diagnostic set and PG1k10 (P22 re-evaluation).
ada credentials update --account=165919938845 --provider=isengard --role=Admin --profile=stop-admin --once >/dev/null 2>&1
W=s3://stop-rl-165919938845-us-west-2/ckpt-sets
aws s3 sync --quiet --profile stop-admin --region us-west-2 $W/dirdiag/ $W/p26pack/ && echo "copied dirdiag set"
aws s3 sync --quiet --profile stop-admin --region us-west-2 $W/v15/PG1k10/ $W/p26pack/PG1k10/ --exclude "*" --include "ckpt_step*" && echo "copied PG1k10"
aws s3 ls --profile stop-admin --region us-west-2 $W/p26pack/
aws s3 ls --profile stop-admin --region us-west-2 $W/p26pack/PG1k10/
