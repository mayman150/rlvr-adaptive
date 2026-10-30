"""Copy a public Hugging Face model (pinned revision) into the project's S3 model store, for use as a SageMaker model
channel. Usage (inside a job): python import_model.py --repo R --revision SHA --bucket B --prefix models/NAME
[--out --model ignored]"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import boto3

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--revision", required=True)
ap.add_argument("--bucket", required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--region", default="us-west-2")
ap.add_argument("--out", default=None)
ap.add_argument("--model", default=None)
a = ap.parse_args()
dest = Path("/opt/ml/import") / a.repo.replace("/", "__")
env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
code = ("import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2], "
        "local_dir=sys.argv[3], allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', '*.py'])")
subprocess.run([sys.executable, "-c", code, a.repo, a.revision, str(dest)], env=env, check=True)
s3 = boto3.client("s3", region_name=a.region)
n = 0
for f in sorted(dest.rglob("*")):
    if f.is_file() and ".cache" not in f.parts:
        s3.upload_file(str(f), a.bucket, f"{a.prefix.rstrip('/')}/{f.relative_to(dest)}")
        n += 1
print(f"uploaded {n} files from {a.repo}@{a.revision} to s3://{a.bucket}/{a.prefix}", flush=True)
