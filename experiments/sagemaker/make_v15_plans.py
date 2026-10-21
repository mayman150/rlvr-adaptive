"""Pre-registration 15 plans (matched-accuracy audit of 2026 methods). Usage: python make_v15_plans.py smoke | train | eval SEED
smoke: plans/v15smoke_s8_<ARM>.json (6 steps; BBG, MXR, and Pre-registration 14's T32).
train: plans/v15_s<seed>_<ARM>.json, one lane each: BBG (W-REINFORCE, positive weight 0.3, Bayesian boundary gating
at k = 16), MXR (MaxRL) and WRF (ungated W-REINFORCE, positive weight 0.3; BBG's own base).
eval SEED: plans/v15e_s<SEED>_E1.json (4 lanes) and _E2.json (WRF, 2 lanes); ckpts channel = a prefix holding the arms.
"""

import json
import sys
from pathlib import Path

from make_v14_plans import DISC, META, ev

HERE = Path(__file__).resolve().parent
ARMS = {"BBG": ["--arms", "wreinforce", "--pos-weight", "0.3", "--bbg-k", "16", "--bbg-tau", "0.01"],
        "MXR": ["--arms", "maxrl"],
        "WRF": ["--arms", "wreinforce", "--pos-weight", "0.3"]}


def common(seed, steps="150", save="50,100,150"):
    return ["--seed", str(seed), "--steps", steps, "--group", "8", "--prompts-per-step", "16", "--max-new", "512",
            "--lr", "1e-6", "--gen-batch", "192", "--calib-prompts", "16", "--eval-steps", steps, "--entropy-every", "5",
            "--save-steps", save, "--train", "train.json"]


def main():
    if sys.argv[1] == "smoke":
        arms = {k: v for k, v in ARMS.items() if k != "WRF"} | {"T32": ["--arms", "baseline", "--tar-archive", DISC]}
        for name, extra in arms.items():
            (HERE / "plans" / f"v15smoke_s8_{name}.json").write_text(
                json.dumps({**META, "common": common(8, "6", "6"), "gpus": [[[name, extra]]]}, indent=1))
        print("wrote plans/v15smoke_s8_{BBG,MXR,T32}.json")
    elif sys.argv[1] == "train":
        for seed in (8, 9, 10):
            for name, extra in ARMS.items():
                (HERE / "plans" / f"v15_s{seed}_{name}.json").write_text(
                    json.dumps({**META, "common": common(seed), "gpus": [[[name, extra]]]}, indent=1))
        print("wrote plans/v15_s{8,9,10}_{BBG,MXR,WRF}.json")
    else:
        seed = sys.argv[2]
        lanes = [[ev("BBG", "50,100", "1.0", 1)], [ev("BBG", "150", "1.0", 2)], [ev("MXR", "50,100", "1.0", 1)],
                 [ev("MXR", "150", "1.0", 2)]]
        (HERE / "plans" / f"v15e_s{seed}_E1.json").write_text(json.dumps({**META, "common": [], "gpus": [], "eval_lanes": lanes}, indent=1))
        e2 = [[ev("WRF", "50,100", "1.0", 1)], [ev("WRF", "150", "1.0", 2)]]
        (HERE / "plans" / f"v15e_s{seed}_E2.json").write_text(json.dumps({**META, "common": [], "gpus": [], "eval_lanes": e2}, indent=1))
        print(f"wrote plans/v15e_s{seed}_E1.json and _E2.json")


if __name__ == "__main__":
    main()
