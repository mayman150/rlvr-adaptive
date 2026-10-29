"""Wait (polling S3) until a marker object exists, then download an S3 prefix to a local directory. Lets a separate
one-GPU job continue from a start that another job saves (synced from its /opt/ml/checkpoints). Fails after
--timeout seconds. Usage: python s3_fetch_wait.py --bucket B --marker KEY --prefix PREFIX --dest DIR [--out --model]"""
import argparse
import time
from pathlib import Path

import boto3

ap = argparse.ArgumentParser()
ap.add_argument("--bucket", required=True)
ap.add_argument("--marker", required=True, help="object key that must exist before downloading")
ap.add_argument("--prefix", required=True, help="key prefix to download (a saved start directory)")
ap.add_argument("--dest", required=True)
ap.add_argument("--timeout", type=int, default=14400)
ap.add_argument("--out", default=None)
ap.add_argument("--model", default=None)
a = ap.parse_args()
s3 = boto3.client("s3")
t0 = time.time()
while True:
    try:
        s3.head_object(Bucket=a.bucket, Key=a.marker)
        break
    except s3.exceptions.ClientError:
        if time.time() - t0 > a.timeout:
            raise SystemExit(f"timed out waiting for s3://{a.bucket}/{a.marker}")
        time.sleep(60)
time.sleep(120)  # let the checkpoint sync finish
dest = Path(a.dest)
n = 0
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=a.bucket, Prefix=a.prefix):
    for obj in page.get("Contents", []):
        rel = obj["Key"][len(a.prefix):].lstrip("/")
        if not rel or rel.endswith(".sagemaker-uploaded"):
            continue
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(a.bucket, obj["Key"], str(dest / rel))
        n += 1
print(f"downloaded {n} files to {dest}", flush=True)
