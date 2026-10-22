"""Check 16 plans (1,024-token re-evaluation). Usage: python make_v16t_plans.py
Writes plans/v16t_s<seed>_L<i>.json for seeds 9 and 10: L0 B 50,100; L1 B 150 (+ base for seed 9); L2 S32 0,50; L3 S32 100,150.
"""

import json
from pathlib import Path

from make_v14_plans import META

HERE = Path(__file__).resolve().parent
ARGS = ["--problems", "eval_frontier_512.json", "--n", "32", "--max-new", "1024", "--gen-batch", "192", "--temperatures", "1.0"]


def spec(name, steps, expect, base=False):
    s = {"name": name, "sub": name, "part": 0, "args": ARGS + ["--expect-steps", expect, "--sampled-steps", steps]}
    return s


for seed in (9, 10):
    lanes = [[spec("B", "50,100", "50,100,150")], [spec("B", "150", "50,100,150")], [spec("S32", "0,50", "0,50,100,150")],
             [spec("S32", "100,150", "0,50,100,150")]]
    if seed == 9:
        lanes[1].append({"name": "BASE", "sub": None, "part": 9, "include_base": True,
                         "args": ARGS + ["--base-temperatures", "1.0", "--sampled-steps", "0", "--expect-steps", "0"]})
    for i, lane in enumerate(lanes):
        (HERE / "plans" / f"v16t_s{seed}_L{i}.json").write_text(json.dumps({**META, "common": [], "gpus": [], "eval_lanes": [lane]}, indent=1))
print("wrote plans/v16t_s{9,10}_L{0..3}.json")
