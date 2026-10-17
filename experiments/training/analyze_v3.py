"""Analysis of Pre-registration 4: positive versus negative reinforcement.

Usage: python analyze_v3.py [results_dir]
Reads results/worth-v3-<RUN>-s<seed>-*/<RUN>/ (synced by fetch_live.sh).

Per arm and seed (steps 0 to 49 unless stated):
  q        projection of the per-prompt mean update on the reference direction (rho);
           the theorem predicts q(psr) = q(nsr) = 8/7 q(drgrpo) and q(wreinforce) = 1.1 q(psr)
  m2       measured per-prompt second moment (split-half), against the theorem's noise
           term sum_i A_i^2 F_{r_i} from calibration F0, F1 and the observed group counts
  entropy  on-policy probe entropy (first 4 answers of every prompt), steps 100 to 150
  pass@k   at step 150
"""

import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_v2 import window_stats  # noqa: E402
from train_grpo import fixed_advantage  # noqa: E402

ARMS = {"DR": "baseline", "PSR": "psr", "NSR": "nsr", "WR": "wreinforce"}
MEAN_FACTOR = {"baseline": 7 / 8, "psr": 1.0, "nsr": 1.0, "wreinforce": 1.1}


def load(root):
    runs = {}
    for job in sorted(root.glob("worth-v3-*")):
        m = re.match(r"worth-v3-(\w+?)-s(\d+)-", job.name)
        if not m or m.group(1) not in ARMS:
            continue
        name, seed = m.group(1), int(m.group(2))
        arm = ARMS[name]
        folder = job / name
        get = lambda f: json.loads((folder / f).read_text()) if (folder / f).exists() else None
        train = get(f"{arm}_train.json") or []
        key = (name, seed)
        if key in runs and len(runs[key]["train"]) >= len(train):
            continue
        runs[key] = {"arm": arm, "train": train, "eval": get(f"{arm}_eval.json") or [],
                     "passk": get(f"{arm}_passk.json") or [], "final": get(f"{arm}_final_eval.json"),
                     "calib": get("calibration.json")}
    return runs


def predicted_noise(arm, steps, calib):
    F1, F0 = calib["F1"], calib["F0"]
    tot, cnt = 0.0, 0
    for x in steps:
        for k, n in x["group_counts"]:
            m = k / n
            a1 = fixed_advantage(arm, 1, m, calib.get("w", 1.0), 0.1)
            a0 = fixed_advantage(arm, 0, m, calib.get("w", 1.0), 0.1)
            tot += k * a1 * a1 * F1 + (n - k) * a0 * a0 * F0
            cnt += 1
    return tot / cnt if cnt else math.nan


def mean_probe(train, key, lo, hi):
    xs = [x["probe"][key] for x in train if lo <= x["step"] <= hi and x.get("probe") and x["probe"].get(key) is not None]
    return sum(xs) / len(xs) if xs else math.nan


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    runs = load(root)
    if not runs:
        print("no v3 runs under", root)
        return
    print("run  seed steps  q/q(PSR)  (theory)   m2 measured / theorem   probe entropy 0-10 | 100-150   greedy@150  pass@1  pass@64")
    for seed in sorted({s for _, s in runs}):
        ref = runs.get(("PSR", seed))
        ref_st = window_stats([x for x in ref["train"] if x["step"] < 50]) if ref else None
        for name in ARMS:
            r = runs.get((name, seed))
            if not r or not r["train"]:
                continue
            steps = [x for x in r["train"] if x["step"] < 50]
            st = window_stats(steps)
            qrel = st["q"] / ref_st["q"] if st and ref_st and ref_st["q"] else math.nan
            pred = predicted_noise(r["arm"], steps, r["calib"]) if r["calib"] else math.nan
            ratio = st["m2"] / pred if st and pred else math.nan
            e0, e1 = mean_probe(r["train"], "entropy", 0, 10), mean_probe(r["train"], "entropy", 100, 150)
            g = next((100 * e["accuracy"] for e in r["eval"] if e["step"] == 150), math.nan)
            pk = next((p["pass_at_k"] for p in r["passk"] if p["step"] == 150), {})
            p1 = 100 * pk.get("1", pk.get(1, math.nan)) if pk else math.nan
            p64 = 100 * pk.get("64", pk.get(64, math.nan)) if pk else math.nan
            print(f"{name:4s} {seed:3d} {len(r['train']):5d}  {qrel:7.2f}   ({MEAN_FACTOR[r['arm']]:.2f})   {ratio:10.2f}"
                  f"              {e0:.3f} | {e1:.3f}            {g:6.1f}   {p1:6.1f}  {p64:6.1f}")
    print("\nPre-registered primary: pass@64 at step 150, NSR minus PSR, both seeds (>= 3 points), or entropy 100-150 higher by >= 20%.")
    for seed in sorted({s for _, s in runs}):
        a, b = runs.get(("NSR", seed)), runs.get(("PSR", seed))
        if not a or not b:
            continue
        pa = next((p for p in a["passk"] if p["step"] == 150), None)
        pb = next((p for p in b["passk"] if p["step"] == 150), None)
        if pa and pb:
            d = 100 * (pa["pass_at_k"]["64"] - pb["pass_at_k"]["64"])
            print(f"  seed {seed}: pass@64 NSR - PSR = {d:+.1f} points")
        ea, eb = mean_probe(a["train"], "entropy", 100, 150), mean_probe(b["train"], "entropy", 100, 150)
        if math.isfinite(ea) and math.isfinite(eb):
            print(f"  seed {seed}: entropy 100-150 NSR / PSR = {ea / eb:.2f}")


if __name__ == "__main__":
    main()
