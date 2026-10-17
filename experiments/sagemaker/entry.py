"""SageMaker entry point: runs the paired training comparison; results go to /opt/ml/model."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    for name, default in (("seed", "1"), ("steps", "150"), ("prompts-per-step", "16"), ("group", "8"),
                          ("max-new", "512"), ("lr", "1e-6"), ("eval-every", "50"),
                          ("arms", "baseline,scaled")):
        parser.add_argument(f"--{name}", default=default)
    args, _ = parser.parse_known_args()
    command = [sys.executable, str(HERE / "training" / "train_grpo.py"), "--device", "cuda",
               "--out", "/opt/ml/model/training"]
    for key, value in vars(args).items():
        command += [f"--{key.replace('_', '-')}", value]
    print("running:", " ".join(command), flush=True)
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
