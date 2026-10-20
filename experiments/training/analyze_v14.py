"""Pre-registration 14 analysis. See PREREGISTRATION_14.md.

Usage: python analyze_v14.py [--draws 2000]
References (B, BASE, CM) from Pre-registration 12's evaluation jobs (worth-v12e-*); arms D32, U32, S32, C05
from worth-v14e-*.
"""

import argparse
import json
import re

import numpy as np

import analyze_v12 as A


def load():
    out = {}
    for prefix in ("worth-v12e-", "worth-v14e-"):
        for job in sorted(A.RES.glob(f"{prefix}E*-s*-*"), key=lambda j: j.name[-15:]):
            seed = int(re.match(rf"{prefix}E\w*-s(\d+)-", job.name).group(1))
            for arm_dir in job.iterdir():
                if not (arm_dir / "eval").exists():
                    continue
                for f in (arm_dir / "eval").glob("step*_T*_eval.json"):
                    step, t = re.match(r"step(\d+)_T([\d.]+)_eval", f.name).groups()
                    d = json.loads(f.read_text())
                    if "counts" in d and d.get("n") == A.N:
                        key = ("BASE", 0, 0, 1.0) if arm_dir.name == "BASE" else (arm_dir.name, seed, int(step), round(float(t), 2))
                        out[key] = np.array(d["counts"])
                        if "greedy" in d:
                            out[("greedy",) + key] = np.array(d["greedy"]["correct"])
    return out


def traj_ref(arm):
    def ref(r, s, w):
        px, py = A.curve([(r[("BASE", 0, 0, 1.0)], 0)] + [(r[(arm, s, st, 1.0)], st) for st in A.STEPS], w)
        value = lambda x: float(np.interp(x, px, py)) if px.min() <= x <= px.max() else None
        return value, px.min(), px.max()
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    res = load()
    arms = ["B", "D32", "U32", "S32", "C05"]
    need = [("BASE", 0, 0, 1.0)] + [(a, s, st, 1.0) for a in arms for s in A.SEEDS for st in A.STEPS] + \
           [("B", s, 150, t) for s in A.SEEDS for t in (1.15, 1.3)]
    missing = [k for k in need if k not in res]
    if missing:
        print(f"incomplete evaluation; missing {missing}")
        return
    n = len(res[("BASE", 0, 0, 1.0)])
    envelope = lambda r, s, w: A.reference(r, s, w, True)

    print(f"1. Joint win: D32 against B's envelope ({n} problems, seeds {A.SEEDS})")
    prim = A.compare(res, "D32", envelope, A.SEEDS, args.draws, rng)
    verdict = A.three_way(prim)
    A.show("D32", prim, verdict)
    d = np.mean([res[("D32", s, 150, 1.0)] / A.N - res[("B", s, 150, 1.0)] / A.N for s in A.SEEDS], axis=0) * 100
    boot = [d[rng.integers(0, n, n)].mean() for _ in range(args.draws)]
    lo, hi = np.percentile(boot, [5, 95])
    equal = lo >= -1 and hi <= 1
    print(f"   step-150 pass@1, D32 minus B: {d.mean():+.2f}, 90% [{lo:+.2f}, {hi:+.2f}] -> " + ("equal accuracy" if equal else "not equal"))
    if verdict == "moves the curve" and equal:
        print("   VERDICT: joint win")
    elif verdict == "moves the curve":
        print("   VERDICT: moves the curve without equal accuracy (accuracy trade-off)")
    else:
        print(f"   VERDICT: {verdict}")

    print("2. Mechanism contrasts (D32 against each arm's trajectory; claim needs >= +1 and 95% > 0)")
    tests = {"D32 vs B envelope": prim}
    for other in ("U32", "S32", "C05"):
        r = A.compare(res, "D32", traj_ref(other), A.SEEDS, args.draws, rng)
        ok = "estimate" in r and r["estimate"] >= 1 and r["ci95"][0] > 0
        A.show(f"D32 vs {other}", r, "supported" if ok else "not shown")
        tests[f"D32 vs {other}"] = r
    for other in ("U32", "S32", "C05"):
        r = A.compare(res, other, envelope, A.SEEDS, args.draws, rng)
        A.show(f"{other} vs B envelope (reported)", r, A.three_way(r))
    ps = sorted((r.get("p_two_sided", 1.0), k) for k, r in tests.items())
    m, stop = len(ps), False
    print("   Holm over the directional tests (CM's Pre-registration 12 test is added when its p-value is supplied):")
    for i, (p, k) in enumerate(ps):
        ok = not stop and p <= 0.05 / (m - i)
        stop = stop or not ok
        print(f"      {k}: p = {p:.4f} -> {'significant' if ok else 'not significant'}")

    print("Greedy test accuracy at step 150 (3,382 problems), mean over seeds")
    for a in arms:
        g = [res.get(("greedy", a, s, 150, 1.0)) for s in A.SEEDS]
        if all(x is not None for x in g):
            print(f"   {a}: {100 * np.mean([x.mean() for x in g]):.1f}")


if __name__ == "__main__":
    main()
