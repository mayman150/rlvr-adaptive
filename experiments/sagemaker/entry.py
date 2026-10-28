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
EVAL_VLLM = HERE / "training" / "eval_vllm.py"
VLLM_ENV = Path("/opt/vllm-env")
VLLM_VERSION = "0.11.0"
_vllm_lock = threading.Lock()
CKPT_CHANNEL = Path("/opt/ml/input/data/ckpts")
LIVE = Path("/opt/ml/checkpoints")
MODEL_CHANNEL = Path("/opt/ml/input/data/model")
FINAL = Path("/opt/ml/model")


def clean_env():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["PYTHONNOUSERSITE"] = "1"
    return env


def vllm_python():
    """A separate environment with vLLM (it pins its own torch and transformers); installed once per job, isolated
    from the container's packages, and checked (versions, origins, CUDA) before use."""
    with _vllm_lock:
        py = VLLM_ENV / "bin" / "python"
        if not (VLLM_ENV / ".ready").exists():
            env = clean_env()
            subprocess.run([sys.executable, "-I", "-m", "venv", str(VLLM_ENV)], check=True, env=env)
            subprocess.run([str(py), "-I", "-m", "pip", "install", "-q", f"vllm=={VLLM_VERSION}", "transformers>=4.56,<5"],
                           check=True, env=env)
            check = ("import torch, transformers, vllm; print('vllm', vllm.__version__, vllm.__file__); "
                     "print('transformers', transformers.__version__, transformers.__file__); "
                     "print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); "
                     "assert transformers.__file__.startswith('%s') and torch.cuda.is_available()" % VLLM_ENV)
            subprocess.run([str(py), "-I", "-c", check], check=True, env=env)
            subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"], env=env)
            (VLLM_ENV / ".ready").write_text(VLLM_VERSION)
            print(f"vLLM {VLLM_VERSION} installed in {VLLM_ENV}", flush=True)
        return str(py)


def local_model_args():
    """Load the model from the S3 input channel when present (no Hugging Face Hub calls)."""
    if (MODEL_CHANNEL / "config.json").exists():
        os.environ["HF_HUB_OFFLINE"] = "1"
        return ["--model", str(MODEL_CHANNEL)]
    return []


def run_one(name, cmd, gpu, status, isolated=False):
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUDA_VISIBLE_DEVICES": str(gpu)}
    if isolated:  # a separate virtual environment: do not let the container's site-packages shadow its own
        env = {**clean_env(), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUDA_VISIBLE_DEVICES": str(gpu)}
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
    if "lanes" in plan:
        # Mixed lanes, one per GPU, each a sequence of steps run in order (a step after a failed one is skipped):
        # {"kind": "train", "name", "args"}: train_grpo.py with plan["common"] + args, output /opt/ml/checkpoints/<name>;
        # {"kind": "script", "name", "script", "args"}: training/<script> with --out /opt/ml/checkpoints/<name>;
        # {"kind": "eval", ...}: an eval_lanes spec whose checkpoints come from the ckpts channel ("sub") or from an
        # earlier step of this job ("live": a path under /opt/ml/checkpoints); output /opt/ml/checkpoints/<name>/eval.
        LIVE.mkdir(parents=True, exist_ok=True)

        def mixed_lane(gpu, steps):
            for spec in steps:
                key = f"{spec['name']}#{spec['kind']}#{spec.get('part', 0)}#lane{gpu}"
                if any(code != 0 for k, code in status.items() if k.endswith(f"#lane{gpu}")):
                    status[key] = "skipped"
                    print(f"[{key}] skipped (an earlier step of lane {gpu} failed)", flush=True)
                    continue
                if spec["kind"] == "train":
                    cmd = [sys.executable, str(TRAIN), "--device", "cuda", "--out", str(LIVE / spec["name"]),
                           *plan["common"], *spec["args"], *local]
                    run_one(key, cmd, gpu, status)
                elif spec["kind"] == "script":
                    cmd = [sys.executable, str(HERE / "training" / spec["script"]), *spec["args"],
                           "--out", str(LIVE / spec["name"]), *local]
                    run_one(key, cmd, gpu, status)
                elif spec["kind"] == "eval":
                    if spec.get("engine", "hf") not in ("hf", "vllm"):
                        raise ValueError(f"unknown engine {spec['engine']}")
                    ckpts = LIVE / spec["live"] if spec.get("live") else CKPT_CHANNEL / spec["sub"]
                    base = ["--base", local[1]] if spec.get("include_base") else []
                    engine = [vllm_python(), "-I", str(EVAL_VLLM)] if spec.get("engine") == "vllm" else [sys.executable, str(EVAL)]
                    cmd = [*engine, "--ckpts", str(ckpts), *base, "--out", str(LIVE / spec["name"] / "eval"), *spec["args"]]
                    run_one(key, cmd, gpu, status, isolated=spec.get("engine") == "vllm")
                else:
                    raise ValueError(f"unknown step kind {spec['kind']}")

        def guarded_mixed(g, steps):
            try:
                mixed_lane(g, steps)
            except Exception as e:
                status[f"lane{g}"] = f"exception: {e!r}"
                print(f"lane {g} exception: {e!r}", flush=True)
        threads = [threading.Thread(target=guarded_mixed, args=(g, steps)) for g, steps in enumerate(plan["lanes"])]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        (LIVE / "status.json").write_text(json.dumps(status, indent=1))
        shutil.copytree(LIVE, FINAL / "runs", dirs_exist_ok=True, ignore=shutil.ignore_patterns("ckpt_step*"))
        print("status:", status, flush=True)
        if any(code != 0 for code in status.values()):
            sys.exit(1)
        return
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
                if spec.get("engine", "hf") not in ("hf", "vllm"):
                    raise ValueError(f"unknown engine {spec['engine']}")
                engine = [vllm_python(), "-I", str(EVAL_VLLM)] if spec.get("engine") == "vllm" else [sys.executable, str(EVAL)]
                cmd = [*engine, "--ckpts", str(ckpts), *base, "--out", str(LIVE / spec["name"] / "eval"), *spec["args"]]
                run_one(f"{spec['name']}#{spec.get('part', 0)}#lane{gpu}", cmd, gpu, status, isolated=spec.get("engine") == "vllm")
        def guarded_lane(g, specs):
            try:
                eval_lane(g, specs)
            except Exception as e:  # a lane that crashes (e.g. the vLLM install) must not look successful
                status[f"lane{g}"] = f"exception: {e!r}"
                print(f"lane {g} exception: {e!r}", flush=True)
        threads = [threading.Thread(target=guarded_lane, args=(g, specs)) for g, specs in enumerate(plan["eval_lanes"])]
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
        # Multi-GPU runs (train_dist.py, dist_parity.py), each a torchrun over spec["nproc"] GPUs: in sequence, or,
        # with plan["parallel"], concurrently on the disjoint GPU lists spec["gpus"].
        # spec["out"] = "dir" passes --out /opt/ml/checkpoints/<name>, "file" passes --out .../<name>.json.
        LIVE.mkdir(parents=True, exist_ok=True)

        def evaluate_after(spec, gpus):
            # spec["then"]: evaluations of this run's checkpoints (or others under /opt/ml/checkpoints, "live"), run
            # after it succeeds; each one's "split_steps" are dealt round-robin to the run's GPUs, one vLLM process per GPU.
            for ev in spec.get("then", []):
                parts = [ev["split_steps"][i::len(gpus)] for i in range(len(gpus))]
                threads = []
                for gpu, steps in zip(gpus, parts):
                    if not steps:
                        continue
                    cmd = [vllm_python(), "-I", str(EVAL_VLLM), "--ckpts", str(LIVE / ev["live"]),
                           "--out", str(LIVE / ev["name"] / "eval"), *ev["args"], "--sampled-steps", ",".join(str(x) for x in steps)]
                    key = f"{ev['name']}#eval#{'-'.join(str(x) for x in steps)}#gpu{gpu}"
                    status[key] = "not finished"  # every expected evaluation must report

                    def guarded(key=key, cmd=cmd, gpu=gpu):
                        try:
                            run_one(key, cmd, gpu, status, True)
                        except Exception as e:
                            status[key] = f"exception: {e!r}"
                            print(f"[{key}] exception: {e!r}", flush=True)
                    threads.append(threading.Thread(target=guarded))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        def launch(spec):
            status[spec["name"]] = "not finished"  # every expected run must report
            try:
                launch_one(spec)
            except Exception as e:
                status[spec["name"]] = f"exception: {e!r}"
                print(f"[{spec['name']}] exception: {e!r}", flush=True)

        def launch_one(spec):
            gpus = spec.get("gpus") or list(range(int(spec["nproc"])))
            out = {"dir": ["--out", str(LIVE / spec["name"])], "file": ["--out", str(LIVE / f"{spec['name']}.json")]}.get(spec.get("out"), [])
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={len(gpus)}",
                   str(HERE / "training" / spec["script"]), *spec["args"], *out, *local]
            run_one(spec["name"], cmd, ",".join(str(g) for g in gpus), status)
            if status.get(spec["name"]) == 0:
                try:
                    evaluate_after(spec, gpus)
                except Exception as e:  # a failed evaluation must not look successful
                    status[f"{spec['name']}#then"] = f"exception: {e!r}"
                    print(f"[{spec['name']}] evaluation exception: {e!r}", flush=True)

        if plan.get("parallel"):
            used = [g for spec in plan["torchrun"] for g in spec["gpus"]]
            assert len(used) == len(set(used)), "parallel torchrun specs need disjoint GPU lists"
            threads = [threading.Thread(target=launch, args=(spec,)) for spec in plan["torchrun"]]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            for spec in plan["torchrun"]:
                launch(spec)
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
