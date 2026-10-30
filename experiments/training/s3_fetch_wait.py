"""Wait (polling S3) until a marker object exists and a saved start is complete in S3, then download the start to a
local directory. Lets a separate one-GPU job continue from a start that another job saves (synced from its
/opt/ml/checkpoints). Completeness: the start's config.json, tokenizer files, start_meta.json and safetensors are
present, and the SHA-256 of its safetensors (sorted by name, as opd_start.py and sft_start.py compute it) equals
start_meta.json's; otherwise the download is retried. Fails after --timeout seconds.
Usage: python s3_fetch_wait.py --bucket B --marker KEY --prefix START_PREFIX --dest DIR [--out --model ignored]"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import boto3

REQUIRED = ("config.json", "tokenizer.json", "tokenizer_config.json", "start_meta.json")

ap = argparse.ArgumentParser()
ap.add_argument("--bucket", required=True)
ap.add_argument("--marker", required=True, help="object key that must exist before downloading")
ap.add_argument("--prefix", required=True, help="key prefix of the saved start directory")
ap.add_argument("--dest", required=True)
ap.add_argument("--timeout", type=int, default=14400)
ap.add_argument("--region", default="us-west-2", help="the bucket's region (the job may run elsewhere)")
ap.add_argument("--out", default=None)
ap.add_argument("--model", default=None)
a = ap.parse_args()
s3 = boto3.client("s3", region_name=a.region)
prefix = a.prefix.rstrip("/") + "/"
dest = Path(a.dest)
t0 = time.time()


def exists(key):
    try:
        s3.head_object(Bucket=a.bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def attempt():
    if not (exists(a.marker) and exists(prefix + "start_meta.json")):
        return "marker or start_meta.json not yet in S3"
    shutil.rmtree(dest, ignore_errors=True)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=a.bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel or rel.endswith(".sagemaker-uploaded"):
                continue
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(a.bucket, obj["Key"], str(dest / rel))
    missing = [f for f in REQUIRED if not (dest / f).exists()]
    shards = sorted(dest.glob("*.safetensors"))
    if missing or not shards:
        return f"incomplete download (missing {missing or 'safetensors'})"
    h = hashlib.sha256()
    for f in shards:
        h.update(f.read_bytes())
    want = json.loads((dest / "start_meta.json").read_text())["sha256"]
    return None if h.hexdigest() == want else "safetensors hash differs from start_meta.json (sync in progress?)"


while True:
    why = attempt()
    if why is None:
        break
    if time.time() - t0 > a.timeout:
        raise SystemExit(f"timed out: {why}")
    print(f"waiting: {why}", flush=True)
    time.sleep(90)
print(f"start downloaded and verified in {dest}", flush=True)
