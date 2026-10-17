"""Parse training curves of a running single-arm-per-line job from its CloudWatch log.

Usage: python fetch_logs.py region job [profile]
Writes experiments/training/results/<job>/from_logs.json with, per arm, the training
reward by step and the greedy evaluation accuracy by step.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

STEP = re.compile(r"\[(\w+)\] step (\d+) reward ([\d.]+) mixed (\d+)/(\d+)(?: answers (\d+))?")
EVAL = re.compile(r"\[(\w+)\] step (\d+) eval accuracy ([\d.]+)")


def main():
    region, job = sys.argv[1], sys.argv[2]
    profile = sys.argv[3] if len(sys.argv) > 3 else "stop-ro"
    out = subprocess.run(["aws", "logs", "filter-log-events", "--log-group-name", "/aws/sagemaker/TrainingJobs",
                          "--log-stream-name-prefix", job, "--filter-pattern", '"step"', "--profile", profile,
                          "--region", region, "--query", "events[].message", "--output", "json"],
                         capture_output=True, text=True, check=True).stdout
    curves = {}
    for msg in json.loads(out):
        for line in msg.splitlines():
            m = STEP.search(line)
            if m:
                arm = curves.setdefault(m.group(1), {"train": {}, "eval": {}})
                arm["train"][int(m.group(2))] = {"step": int(m.group(2)), "mean_reward": float(m.group(3)),
                                                 "answers": int(m.group(6)) if m.group(6) else None}
                continue
            m = EVAL.search(line)
            if m:
                arm = curves.setdefault(m.group(1), {"train": {}, "eval": {}})
                arm["eval"][int(m.group(2))] = {"step": int(m.group(2)), "accuracy": float(m.group(3))}
    result = {a: {k: [v[s] for s in sorted(v)] for k, v in c.items()} for a, c in curves.items()}
    dest = Path(__file__).resolve().parents[1] / "training" / "results" / job
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "from_logs.json").write_text(json.dumps(result))
    print(job, {a: (len(c["train"]), len(c["eval"])) for a, c in result.items()})


if __name__ == "__main__":
    main()
