"""SageMaker entry point.

Without plan.json: runs the paired training comparison; results go to /opt/ml/model.
With plan.json ({"common": [args], "gpus": [[[name, [args]], ...], ...]}): runs each GPU's
list of runs in sequence, all GPUs in parallel. Results are written under
/opt/ml/checkpoints/<name> (synced to S3 while the job runs) and copied to /opt/ml/model.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "training" / "train_grpo.py"
LIVE = Path("/opt/ml/checkpoints")
FINAL = Path("/opt/ml/model")


def run_one(name, cmd, gpu, status):
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUDA_VISIBLE_DEVICES": str(gpu)}
    print(f"[{name}] gpu {gpu}: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(f"[{name}] {line.rstrip()}", flush=True)
    status[name] = proc.wait()
    print(f"[{name}] exit {status[name]}", flush=True)


def run_plan(plan):
    from huggingface_hub import snapshot_download
    snapshot_download(plan.get("model", "Qwen/Qwen2.5-Math-1.5B"))
    status = {}

    def lane(gpu, runs):
        for name, extra in runs:
            out = LIVE / name
            cmd = [sys.executable, str(TRAIN), "--device", "cuda", "--out", str(out), *plan["common"], *extra]
            run_one(name, cmd, gpu, status)

    threads = [threading.Thread(target=lane, args=(g, runs)) for g, runs in enumerate(plan["gpus"])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    LIVE.mkdir(parents=True, exist_ok=True)
    (LIVE / "status.json").write_text(json.dumps(status, indent=1))
    shutil.copytree(LIVE, FINAL / "runs", dirs_exist_ok=True)
    print("status:", status, flush=True)
    if any(code != 0 for code in status.values()):
        sys.exit(1)


def main():
    plan_path = HERE / "plan.json"
    if plan_path.exists():
        run_plan(json.loads(plan_path.read_text()))
        return
    parser = argparse.ArgumentParser()
    for name, default in (("seed", "1"), ("steps", "150"), ("prompts-per-step", "16"), ("group", "8"),
                          ("max-new", "512"), ("lr", "1e-6"), ("eval-every", "50"),
                          ("arms", "baseline,scaled")):
        parser.add_argument(f"--{name}", default=default)
    args, _ = parser.parse_known_args()
    command = [sys.executable, str(TRAIN), "--device", "cuda", "--out", "/opt/ml/model/training"]
    for key, value in vars(args).items():
        command += [f"--{key.replace('_', '-')}", value]
    print("running:", " ".join(command), flush=True)
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
