"""Pre-registration 20's evaluations with eval_vllm.py: one ml.p4de.24xlarge job per arm (8 lanes), checkpoints from the
arm's training-job prefix (us-east-1); the 511 test problems and the frozen hard cohort (data/hard7_cohort.json), 32
samples at T = 1 and 2,048 tokens, B7's step-150 temperatures on both sets, greedy test accuracy at step 150.
B7's step 25-150 (seed 8) and step 150 (seed 9) test-set evaluations exist from the vLLM test job and are not repeated.
Writes plans/v20v_<ARM>.json.
"""

import json
from pathlib import Path

P = Path(__file__).resolve().parent / "plans"
META = {"model": "Qwen/Qwen2.5-Math-7B", "model_dir": "Qwen2.5-Math-7B", "architecture": "Qwen2ForCausalLM", "common": [], "gpus": []}
GREEDY = ["--greedy-problems", "eval_math_test_numeric.json", "--greedy-steps", "150"]


def spec(name, sub, problems, steps, expect, temps="1.0", greedy=False, base=False):
    a = ["--problems", problems, "--n", "32", "--max-new", "2048", "--expect-steps", expect, "--sampled-steps", steps, "--temperatures", temps]
    s = {"name": name, "sub": sub, "part": 0, "engine": "vllm", "args": a + (GREEDY if greedy else [])}
    if base:
        s["include_base"] = True
        s["args"] = s["args"] + ["--base-temperatures", "1.0"]
    return s


def write(arm, lanes):
    assert len(lanes) <= 8
    (P / f"v20v_{arm}.json").write_text(json.dumps({**META, "eval_lanes": lanes}, indent=1))
    print(arm, [[s["name"] + ":" + s["args"][s["args"].index("--sampled-steps") + 1] + "@" + s["args"][s["args"].index("--temperatures") + 1] for s in l] for l in lanes])


T, H, ALL4, ALL3 = "eval_frontier_512.json", "hard7_cohort.json", "25,50,100,150", "50,100,150"
write("B7", [
    [spec("B7s8", "B7s8", T, "150", ALL4, "1.15,1.3")],
    [spec("B7s9", "B7s9", T, "25", ALL4), spec("B7s9", "B7s9", T, "50", ALL4)],
    [spec("B7s9", "B7s9", T, "100", ALL4), spec("BASE7hard", None, H, "0", "0", base=True)],
    [spec("B7s9", "B7s9", T, "150", ALL4, "1.15,1.3")],
    [spec("B7s8hard", "B7s8", H, ALL4, ALL4)],
    [spec("B7s9hard", "B7s9", H, ALL4, ALL4)],
    [spec("B7s8hard", "B7s8", H, "150", ALL4, "1.15,1.3")],
    [spec("B7s9hard", "B7s9", H, "150", ALL4, "1.15,1.3")]])
for arm, steps, expect in (("PSR7", ("25", "50", "100", "150"), ALL4), ("NSR7", ("50", "100", "150"), ALL3), ("AC7", ("50", "100", "150"), ALL3)):
    test = [spec(f"{arm}s{s}", f"{arm}s{s}", T, st, expect, greedy=(st == "150")) for s in (8, 9) for st in steps]
    hard = [spec(f"{arm}s{s}hard", f"{arm}s{s}", H, ",".join(steps), expect) for s in (8, 9)]
    lanes = [[x] for x in test] + [[h] for h in hard]
    while len(lanes) > 8:  # fold the shortest lanes together
        lanes.sort(key=len)
        lanes = [lanes[0] + lanes[1]] + lanes[2:]
    write(arm, lanes)
