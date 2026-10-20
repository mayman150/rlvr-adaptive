"""Write Pre-registration 12's primary training plans for branch M.

Usage: python make_v12_train_plans.py LAMBDA_CM LAMBDA_MB
Seeds 8, 9, 10: a 4-lane job (B, CM, M, U; M uses CM's lambda) and a 1-lane job (MB).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
lam_cm, lam_mb = sys.argv[1], sys.argv[2]
meta = {"model": "Qwen/Qwen2.5-Math-1.5B", "model_dir": "Qwen2.5-Math-1.5B", "architecture": "Qwen2ForCausalLM"}
for seed in (8, 9, 10):
    common = ["--seed", str(seed), "--steps", "150", "--group", "8", "--prompts-per-step", "16", "--max-new", "512",
              "--lr", "1e-6", "--gen-batch", "192", "--calib-prompts", "16", "--eval-steps", "150", "--entropy-every", "5",
              "--save-steps", "50,100,150", "--train", "train.json"]
    lanes = [[["B", ["--arms", "baseline"]]],
             [["CM", ["--arms", "baseline", "--memory-lambda", lam_cm, "--memory-weighting", "coverage"]]],
             [["M", ["--arms", "baseline", "--memory-lambda", lam_cm]]],
             [["U", ["--arms", "baseline", "--unlikely-beta", "0.25"]]]]
    (HERE / "plans" / f"v12_s{seed}_ALL.json").write_text(json.dumps({**meta, "common": common, "gpus": lanes}, indent=1))
    (HERE / "plans" / f"v12_s{seed}_MB.json").write_text(json.dumps(
        {**meta, "common": common, "gpus": [[["MB", ["--arms", "baseline", "--memory-lambda", lam_mb]]]]}, indent=1))
print(f"wrote plans for seeds 8-10 (CM and M lambda {lam_cm}, MB lambda {lam_mb})")
