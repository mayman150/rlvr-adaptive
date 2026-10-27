"""Plans that re-evaluate every 1.5B run of Pre-registrations 21, 22 and 23 with eval_vllm.py on one A100 machine
(8 lanes), from the merged checkpoint prefix ckpt-sets/v15/<arm><seed>/ in us-west-2. One engine for every arm of a
comparison; the registered steps, temperatures and greedy evaluations of each pre-registration.
Usage: python make_v15_vllm_plans.py RUN [RUN ...]   (runs like B1k8 PSR1k9 HG1k10 MH8; writes plans/v15v_*.json)
"""

import json
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parent / "plans"
META = {"model": "Qwen/Qwen2.5-Math-1.5B", "model_dir": "Qwen2.5-Math-1.5B", "architecture": "Qwen2ForCausalLM", "common": [], "gpus": []}
TEST, VAL = "eval_frontier_512.json", "val_frontier_256.json"
GREEDY = ["--greedy-problems", "eval_math_test_numeric.json", "--greedy-steps", "150"]


def spec(name, sub, problems, steps, expect, temps="1.0", greedy=False):
    a = ["--problems", problems, "--n", "32", "--max-new", "1024", "--expect-steps", expect, "--sampled-steps", steps, "--temperatures", temps]
    return {"name": name, "sub": sub, "part": 0, "engine": "vllm", "args": a + (GREEDY if greedy else [])}


def specs(run):
    arm, seed = re.fullmatch(r"([A-Za-z0-9]+?k|MH|AVH)(\d+)", run).groups()
    if arm == "B1k":
        out = [spec(run, run, TEST, "25,50,100,150", "25,50,100,150", greedy=True),
               spec(run, run, TEST, "150", "25,50,100,150", "1.15,1.3")]
        if seed == "8":  # Pre-registration 23's reference on the validation problems
            out += [spec(run + "val", run, VAL, "25,50,100,150", "25,50,100,150"), spec(run + "val", run, VAL, "150", "25,50,100,150", "1.15,1.3")]
        return out
    if arm in ("PSR1k", "PG1k"):
        return [spec(run, run, TEST, "25,50,100,150", "25,50,100,150", greedy=True)]
    if arm in ("MH", "AVH"):
        return [spec(run + "val", run, VAL, "50,100,150", "50,100,150")]
    return [spec(run, run, TEST, "50,100,150", "50,100,150", greedy=True)]


def main():
    runs = sys.argv[1:]
    items = [s for r in runs for s in specs(r)]
    base = [{"name": "BASE15", "sub": None, "part": 0, "engine": "vllm", "include_base": True,
             "args": ["--problems", TEST, "--n", "32", "--max-new", "1024", "--temperatures", "1.0", "--base-temperatures", "1.0",
                      "--sampled-steps", "0", "--expect-steps", "0"]},
            {"name": "BASE15val", "sub": None, "part": 0, "engine": "vllm", "include_base": True,
             "args": ["--problems", VAL, "--n", "32", "--max-new", "1024", "--temperatures", "1.0", "--base-temperatures", "1.0",
                      "--sampled-steps", "0", "--expect-steps", "0"]}]
    items = base + items
    cost = lambda s: len(s["args"][s["args"].index("--sampled-steps") + 1].split(",")) * len(s["args"][s["args"].index("--temperatures") + 1].split(","))
    lanes = [[] for _ in range(8)]
    for s in sorted(items, key=cost, reverse=True):  # longest first onto the lightest lane
        min(lanes, key=lambda l: sum(cost(x) for x in l)).append(s)
    (P / "v15v_all.json").write_text(json.dumps({**META, "eval_lanes": lanes}, indent=1))
    print(f"{len(items)} specs, lane loads {[sum(cost(x) for x in l) for l in lanes]}")


if __name__ == "__main__":
    main()
