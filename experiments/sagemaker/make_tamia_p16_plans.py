"""Pre-registration 16 pilot plans for Tamia. Usage: python make_tamia_p16_plans.py
Training: plans/tamia_p16_A.json (B, S32, AR, GC) and _B.json (PG, MXGC, SD, MXR), seed 12.
Evaluation: plans/tamia_p16e_A.json and _B.json (ckpts = the matching training output).
"""

import json
from pathlib import Path

from make_v14_plans import DISC, META

HERE = Path(__file__).resolve().parent
COMMON = ["--seed", "12", "--steps", "150", "--group", "8", "--prompts-per-step", "16", "--max-new", "512", "--lr", "1e-6",
          "--gen-batch", "192", "--calib-prompts", "16", "--eval-steps", "150", "--entropy-every", "5",
          "--save-steps", "50,100,150", "--train", "train.json"]
S0 = ["--save-steps", "0,50,100,150"]
ARMS = {"B": ["--arms", "baseline"],
        "S32": ["--arms", "baseline", "--sft-archive", DISC] + S0,
        "AR": ["--arms", "baseline", "--seed-archive", DISC, "--memory-lambda", "4", "--memory-weighting", "coverage",
               "--memory-questions", "16", "--memory-anneal-steps", "100"],
        "GC": ["--arms", "baseline", "--rescue-archive", DISC, "--complete-groups"],
        "PG": ["--arms", "baseline", "--rescue-archive", DISC, "--prefix-rollouts", "2", "--prefix-frac", "0.5",
               "--prefix-anneal-steps", "100", "--prefix-max-c", "4"],
        "MXGC": ["--arms", "maxrl", "--rescue-archive", DISC, "--complete-groups"],
        "SD": ["--arms", "baseline", "--sft-archive", DISC, "--seed-archive", DISC, "--memory-lambda", "0.5",
               "--memory-weighting", "coverage"] + S0,
        "MXR": ["--arms", "maxrl"]}
EV = ["--problems", "val_frontier_256.json", "--n", "32", "--max-new", "512", "--gen-batch", "256"]


def ev(name, steps, expect, temps="1.0", part=1):
    s = {"name": name, "sub": name, "part": part, "args": EV + ["--expect-steps", expect, "--sampled-steps", steps, "--temperatures", temps]}
    return s


BASE = {"name": "BASE", "sub": None, "part": 9, "include_base": True,
        "args": EV + ["--temperatures", "1.0", "--base-temperatures", "1.0", "--sampled-steps", "0", "--expect-steps", "0"]}
for tag, names in (("A", ("B", "S32", "AR", "GC")), ("B", ("PG", "MXGC", "SD", "MXR"))):
    (HERE / "plans" / f"tamia_p16_{tag}.json").write_text(json.dumps({**META, "common": COMMON, "gpus": [[[n, ARMS[n]]] for n in names]}, indent=1))
e_a = [[ev("B", "50,100", "50,100,150"), BASE], [ev("B", "150", "50,100,150", "1.0,1.15,1.3", part=2)],
       [ev("S32", "0,50,100,150", "0,50,100,150")], [ev("AR", "50,100,150", "50,100,150"), ev("GC", "50,100,150", "50,100,150")]]
e_b = [[ev("PG", "50,100,150", "50,100,150")], [ev("MXGC", "50,100,150", "50,100,150")],
       [ev("SD", "0,50,100,150", "0,50,100,150")], [ev("MXR", "50,100,150", "50,100,150")]]
for tag, lanes in (("A", e_a), ("B", e_b)):
    (HERE / "plans" / f"tamia_p16e_{tag}.json").write_text(json.dumps({**META, "common": [], "gpus": [], "eval_lanes": lanes}, indent=1))
print("wrote plans/tamia_p16_{A,B}.json and tamia_p16e_{A,B}.json")
