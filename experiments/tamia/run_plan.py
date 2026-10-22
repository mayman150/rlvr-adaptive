"""Run a SageMaker-style plan (training "gpus" lanes or "eval_lanes") on one Tamia node, one lane per GPU.

Usage: python run_plan.py PLAN OUT [--ckpts DIR] [--model DIR]
Training lanes write to OUT/<name>; evaluation lanes read OUT-independent checkpoints from --ckpts/<sub> and
write to OUT/<name>/eval. Mirrors experiments/sagemaker/entry.py.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "training"


def guarded(fn, gpu, specs, status):
    try:
        fn(gpu, specs)
    except Exception as e:  # a lane that crashes must not look successful
        status[f"lane{gpu}"] = f"exception: {e!r}"
        print(f"lane {gpu} exception: {e!r}", flush=True)


def run_one(name, cmd, gpu, status, log):
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1"}
    with open(log / f"{name}.log", "a") as f:
        f.write(f"gpu {gpu}: {' '.join(cmd)}\n")
        f.flush()
        status[name] = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"[{name}] exit {status[name]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("out")
    ap.add_argument("--ckpts", default=None)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    plan, out = json.loads(Path(args.plan).read_text()), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "logs"
    log.mkdir(exist_ok=True)
    status, threads = {}, []
    if "eval_lanes" in plan:
        def lane(gpu, specs):
            for spec in specs:
                base = ["--base", args.model] if spec.get("include_base") else []
                ckpts = Path(args.ckpts) / spec["sub"] if spec.get("sub") else Path(args.ckpts or out) / "none"
                cmd = [sys.executable, str(CODE / "eval_ckpt.py"), "--ckpts", str(ckpts), *base,
                       "--out", str(out / spec["name"] / "eval"), *spec["args"]]
                run_one(f"{spec['name']}#{spec.get('part', 0)}", cmd, gpu, status, log)
        lanes = plan["eval_lanes"]
    else:
        def lane(gpu, runs):
            for name, extra in runs:
                cmd = [sys.executable, str(CODE / "train_grpo.py"), "--device", "cuda", "--out", str(out / name),
                       *plan["common"], *extra, "--model", args.model]
                run_one(name, cmd, gpu, status, log)
        lanes = plan["gpus"]
    for g, specs in enumerate(lanes):
        threads.append(threading.Thread(target=guarded, args=(lane, g, specs, status)))
        threads[-1].start()
    for t in threads:
        t.join()
    (out / "status.json").write_text(json.dumps(status, indent=1))
    print("status:", status, flush=True)
    sys.exit(1 if any(v != 0 for v in status.values()) else 0)  # includes lane exceptions (non-zero strings)


if __name__ == "__main__":
    main()
