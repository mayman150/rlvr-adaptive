"""Copy S3 prefixes into local directories (inside a job), skipping SageMaker upload markers; fails if a prefix is empty.
Usage: python s3_copy.py --bucket B --region R --pairs PREFIX=DEST [PREFIX=DEST ...] [--out --model ignored]"""
import argparse
from pathlib import Path

import boto3

ap = argparse.ArgumentParser()
ap.add_argument("--bucket", required=True)
ap.add_argument("--region", default="us-west-2")
ap.add_argument("--pairs", nargs="+", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--model", default=None)
a = ap.parse_args()
s3 = boto3.client("s3", region_name=a.region)
for pair in a.pairs:
    prefix, dest = pair.split("=", 1)
    prefix = prefix.rstrip("/") + "/"
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=a.bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel or rel.endswith(".sagemaker-uploaded"):
                continue
            (Path(dest) / rel).parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(a.bucket, obj["Key"], str(Path(dest) / rel))
            n += 1
    if n == 0:
        raise SystemExit(f"nothing under s3://{a.bucket}/{prefix}")
    print(f"copied {n} files to {dest}", flush=True)
