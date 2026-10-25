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
EVAL = HERE / "training" / "eval_ckpt.py"
CKPT_CHANNEL = Path("/opt/ml/input/data/ckpts")
LIVE = Path("/opt/ml/checkpoints")
MODEL_CHANNEL = Path("/opt/ml/input/data/model")
FINAL = Path("/opt/ml/model")


def local_model_args():
    """Load the model from the S3 input channel when present (no Hugging Face Hub calls)."""
    if (MODEL_CHANNEL / "config.json").exists():
        os.environ["HF_HUB_OFFLINE"] = "1"
        return ["--model", str(MODEL_CHANNEL)]
    return []


def run_one(name, cmd, gpu, status):
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUDA_VISIBLE_DEVICES": str(gpu)}
    print(f"[{name}] gpu {gpu}: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(f"[{name}] {line.rstrip()}", flush=True)
    status[name] = proc.wait()
    print(f"[{name}] exit {status[name]}", flush=True)


def run_plan(plan):
    local = local_model_args()
    if local:
        expected = plan.get("architecture")
        found = json.loads((MODEL_CHANNEL / "config.json").read_text()).get("architectures", ["?"])[0]
        if expected and found != expected:
            raise SystemExit(f"model channel holds {found}, plan expects {expected}")
        print(f"model channel: {found}", flush=True)
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(plan.get("model", "Qwen/Qwen2.5-Math-1.5B"))
        local = ["--model", plan.get("model", "Qwen/Qwen2.5-Math-1.5B")]
    status = {}
    if "eval_lanes" in plan:
        # One lane per GPU; a lane is a list of evaluations run in sequence. Each evaluation reads the
        # checkpoints of one run from a subfolder of the ckpts channel (a training job's S3 prefix)
        # and writes to /opt/ml/checkpoints/<name>/eval.
        def eval_lane(gpu, specs):
            for spec in specs:
                if spec.get("script") == "discover.py":
                    cmd = [sys.executable, str(HERE / "training" / "discover.py"), "--model", local[1],
                           "--out", str(LIVE / spec["name"]), *spec["args"]]
                    run_one(f"{spec['name']}#{spec.get('part', 0)}", cmd, gpu, status)
                    continue
                if spec.get("script") == "transfer_probe.py":
                    cmd = [sys.executable, str(HERE / "training" / "transfer_probe.py"), "--ckpts", str(CKPT_CHANNEL / spec["sub"]),
                           "--base", local[1], "--out", str(LIVE / spec["name"] / "probe"), *spec["args"]]
                    run_one(f"{spec['name']}#{spec.get('part', 0)}", cmd, gpu, status)
                    continue
                base = ["--base", local[1]] if spec.get("include_base") else []
                ckpts = CKPT_CHANNEL / spec["sub"] if spec.get("sub") else CKPT_CHANNEL / "none"
                cmd = [sys.executable, str(EVAL), "--ckpts", str(ckpts), *base, "--out", str(LIVE / spec["name"] / "eval"),
                       *spec["args"]]
                run_one(f"{spec['name']}#{spec.get('part', 0)}", cmd, gpu, status)
        threads = [threading.Thread(target=eval_lane, args=(g, specs)) for g, specs in enumerate(plan["eval_lanes"])]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        LIVE.mkdir(parents=True, exist_ok=True)
        (LIVE / "status.json").write_text(json.dumps(status, indent=1))
        shutil.copytree(LIVE, FINAL / "runs", dirs_exist_ok=True, ignore=shutil.ignore_patterns("ckpt_step*"))
        if any(code != 0 for code in status.values()):
            sys.exit(1)
        return
    if "torchrun" in plan:
        # Multi-GPU runs in sequence (train_dist.py, dist_parity.py), each a torchrun over spec["nproc"] GPUs;
        # spec["out"] = "dir" passes --out /opt/ml/checkpoints/<name>, "file" passes --out .../<name>.json.
        for spec in plan["torchrun"]:
            n = int(spec["nproc"])
            out = {"dir": ["--out", str(LIVE / spec["name"])], "file": ["--out", str(LIVE / f"{spec['name']}.json")]}.get(spec.get("out"), [])
            LIVE.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={n}",
                   str(HERE / "training" / spec["script"]), *spec["args"], *out, *local]
            run_one(spec["name"], cmd, ",".join(str(g) for g in range(n)), status)
        (LIVE / "status.json").write_text(json.dumps(status, indent=1))
        shutil.copytree(LIVE, FINAL / "runs", dirs_exist_ok=True, ignore=shutil.ignore_patterns("ckpt_step*"))
        if any(code != 0 for code in status.values()):
            sys.exit(1)
        return
    if "eval" in plan:
        base = ["--base", local[1]] if plan["eval"].get("include_base") else []
        cmd = [sys.executable, str(EVAL), "--ckpts", str(CKPT_CHANNEL), *base, "--out", str(LIVE / "eval"), *plan["eval"]["args"]]
        run_one("eval", cmd, 0, status)
        LIVE.mkdir(parents=True, exist_ok=True)
        (LIVE / "status.json").write_text(json.dumps(status, indent=1))
        shutil.copytree(LIVE, FINAL / "runs", dirs_exist_ok=True)
        if any(code != 0 for code in status.values()):
            sys.exit(1)
        return

    def lane(gpu, runs):
        for name, extra in runs:
            out = LIVE / name
            cmd = [sys.executable, str(TRAIN), "--device", "cuda", "--out", str(out), *plan["common"], *extra, *local]
            run_one(name, cmd, gpu, status)

    threads = [threading.Thread(target=lane, args=(g, runs)) for g, runs in enumerate(plan["gpus"])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    LIVE.mkdir(parents=True, exist_ok=True)
    (LIVE / "status.json").write_text(json.dumps(status, indent=1))
    # Checkpoints stay in the live S3 sync only; copying them here would double the disk use
    # and the model.tar.gz upload.
    shutil.copytree(LIVE, FINAL / "runs", dirs_exist_ok=True, ignore=shutil.ignore_patterns("ckpt_step*"))
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
    command += local_model_args()
    print("running:", " ".join(command), flush=True)
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
