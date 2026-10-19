"""Write the two 4-lane evaluation plans of one Pre-registration 12 seed.

Usage: python make_v12_eval_plans.py SEED B ARM1 ARM2 ARM3 [--base]
The ckpts channel is the seed's training-job prefix (subfolders B, ARM1, ARM2, ARM3 with ckpt_step50/100/150).
Writes plans/v12e_s<SEED>_E1.json and plans/v12e_s<SEED>_E2.json.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
seed, b, x1, x2, x3 = sys.argv[1:6]
with_base = "--base" in sys.argv
common = ["--problems", "eval_frontier_512.json", "--n", "32", "--max-new", "512", "--gen-batch", "256",
          "--expect-steps", "50,100,150"]


def ev(name, steps, temps, part):
    return {"name": name, "sub": name, "part": part,
            "args": common + ["--sampled-steps", steps, "--temperatures", temps]}


base = {"name": "BASE", "sub": None, "part": 9, "include_base": True,
        "args": ["--problems", "eval_frontier_512.json", "--n", "32", "--max-new", "512", "--gen-batch", "256",
                 "--temperatures", "1.0", "--base-temperatures", "1.0", "--sampled-steps", "0", "--expect-steps", "0"]}
e1 = [[ev(b, "50,100", "1.0", 1)],
      [ev(b, "150", "1.0,1.15", 2)],
      [ev(b, "150", "1.3", 3)] + ([base] if with_base else []),
      [ev(x1, "50,100", "1.0", 1)]]
e2 = [[ev(x1, "150", "1.0", 2), ev(x2, "150", "1.0", 2)],
      [ev(x2, "50,100", "1.0", 1)],
      [ev(x3, "50,100", "1.0", 1)],
      [ev(x3, "150", "1.0", 2)]]
meta = {"model": "Qwen/Qwen2.5-Math-1.5B", "model_dir": "Qwen2.5-Math-1.5B", "architecture": "Qwen2ForCausalLM",
        "common": [], "gpus": []}
for name, lanes in (("E1", e1), ("E2", e2)):
    (HERE / "plans" / f"v12e_s{seed}_{name}.json").write_text(json.dumps({**meta, "eval_lanes": lanes}, indent=1))
print(f"wrote plans/v12e_s{seed}_E1.json and _E2.json")
