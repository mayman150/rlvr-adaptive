"""Pre-registration 14 plans. Usage: python make_v14_plans.py train | eval SEED
train: plans/v14_s{8,9,10}_ALL.json (lanes D32, U32, S32, C05).
eval SEED: plans/v14e_s<SEED>_E1.json and _E2.json (4 lanes each; ckpts channel = the seed's training-job prefix).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
META = {"model": "Qwen/Qwen2.5-Math-1.5B", "model_dir": "Qwen2.5-Math-1.5B", "architecture": "Qwen2ForCausalLM"}
DISC = "discovery_qwen_train32.json"


def train():
    for seed in (8, 9, 10):
        common = ["--seed", str(seed), "--steps", "150", "--group", "8", "--prompts-per-step", "16", "--max-new", "512",
                  "--lr", "1e-6", "--gen-batch", "192", "--calib-prompts", "16", "--eval-steps", "150", "--entropy-every", "5",
                  "--save-steps", "50,100,150", "--train", "train.json"]
        lanes = [[["D32", ["--arms", "baseline", "--memory-lambda", "0.5", "--memory-weighting", "coverage", "--seed-archive", DISC]]],
                 [["U32", ["--arms", "baseline", "--memory-lambda", "0.5", "--seed-archive", DISC]]],
                 [["S32", ["--arms", "baseline", "--sft-archive", DISC, "--save-steps", "0,50,100,150"]]],
                 [["C05", ["--arms", "baseline", "--memory-lambda", "0.5", "--memory-weighting", "coverage"]]]]
        (HERE / "plans" / f"v14_s{seed}_ALL.json").write_text(json.dumps({**META, "common": common, "gpus": lanes}, indent=1))
    print("wrote plans/v14_s{8,9,10}_ALL.json")


def ev(name, steps, temps, part, expect="50,100,150"):
    greedy = ["--greedy-problems", "eval_math_test_numeric.json", "--greedy-steps", "150"] if "150" in steps.split(",") else []
    return {"name": name, "sub": name, "part": part,
            "args": ["--problems", "eval_frontier_512.json", "--n", "32", "--max-new", "512", "--gen-batch", "256",
                     "--expect-steps", expect, "--sampled-steps", steps, "--temperatures", temps] + greedy}


def evals(seed):
    s0 = "0,50,100,150"
    e1 = [[ev("D32", "50,100", "1.0", 1)], [ev("D32", "150", "1.0", 2), ev("U32", "150", "1.0", 2)],
          [ev("U32", "50,100", "1.0", 1)], [ev("C05", "50,100", "1.0", 1)]]
    e2 = [[ev("C05", "150", "1.0", 2)], [ev("S32", "0,50", "1.0", 1, s0)], [ev("S32", "100,150", "1.0", 2, s0)],
          [ev("S32", "150", "1.15,1.3", 3, s0)]]
    for name, lanes in (("E1", e1), ("E2", e2)):
        (HERE / "plans" / f"v14e_s{seed}_{name}.json").write_text(json.dumps({**META, "common": [], "gpus": [], "eval_lanes": lanes}, indent=1))
    print(f"wrote plans/v14e_s{seed}_E1.json and _E2.json")


if __name__ == "__main__":
    train() if sys.argv[1] == "train" else evals(sys.argv[2])
