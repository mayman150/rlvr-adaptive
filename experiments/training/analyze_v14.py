"""Pre-registration 14 analysis. See PREREGISTRATION_14.md.

Usage: python analyze_v14.py [--draws 2000]
References (B, BASE, CM) from Pre-registration 12's evaluation jobs (worth-v12e-*); arms D32, U32, S32, C05
from worth-v14e-* (G32 added by amendment 2). Every comparison uses the same bootstrap draws over problems (a fresh generator with
seed 0 per comparison), so CM's test reproduces Pre-registration 12's primary test exactly.
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
    """An arm's own trajectory at T = 1; it starts from the shared base model, except S32, which starts
    from its own post-fine-tuning checkpoint (step 0)."""
    def ref(r, s, w):
        start = r[("S32", s, 0, 1.0)] if arm == "S32" else r[("BASE", 0, 0, 1.0)]
        px, py = A.curve([(start, 0)] + [(r[(arm, s, st, 1.0)], st) for st in A.STEPS], w)
        value = lambda x: float(np.interp(x, px, py)) if px.min() <= x <= px.max() else None
        return value, px.min(), px.max()
    return ref


def holm(tests, alpha=0.05):
    ps = sorted((r.get("p_two_sided", 1.0), k) for k, r in tests.items())
    out, stop = {}, False
    for i, (p, k) in enumerate(ps):
        ok = not stop and p <= alpha / (len(ps) - i)
        stop = stop or not ok
        out[k] = (p, ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    fresh = lambda: np.random.default_rng(0)
    print(f"problem alignment: {A.check_alignment(('worth-v12e-', 'worth-v14e-'))} evaluation files share one problem order")
    res = load()
    arms = ["B", "D32", "U32", "S32", "C05", "G32"]
    need = [("BASE", 0, 0, 1.0)] + [(a, s, st, 1.0) for a in arms + ["CM"] for s in A.SEEDS for st in A.STEPS] + \
           [(a, s, 150, t) for a in ("B", "S32") for s in A.SEEDS for t in (1.15, 1.3)] + \
           [("S32", s, 0, 1.0) for s in A.SEEDS]
    missing = [k for k in need if k not in res]
    if missing:
        print(f"incomplete evaluation; missing {missing}")
        return
    n = len(res[("BASE", 0, 0, 1.0)])
    envelope = lambda r, s, w: A.reference(r, s, w, True)
    tests, clamps = {}, {}

    def run(name, arm, ref):
        clamps[name] = {}
        tests[name] = A.compare(res, arm, ref, A.SEEDS, args.draws, fresh(), stats=clamps[name])
        return tests[name]

    prim = run("D32 vs B envelope", "D32", envelope)
    run("CM vs B envelope (Pre-registration 12 primary)", "CM", envelope)
    contrasts = ("U32", "S32", "C05", "G32")
    for other in contrasts:
        run(f"D32 vs {other} trajectory", "D32", traj_ref(other))
    h = holm(tests)

    print(f"1. Joint win: D32 against B's envelope ({n} problems, seeds {A.SEEDS})")
    verdict = A.three_way(prim)
    A.show("D32", prim, verdict)
    d = np.mean([res[("D32", s, 150, 1.0)] / A.N - res[("B", s, 150, 1.0)] / A.N for s in A.SEEDS], axis=0) * 100
    r = fresh()
    boot = [d[r.integers(0, n, n)].mean() for _ in range(args.draws)]
    lo, hi = np.percentile(boot, [5, 95])
    equal = lo >= -1 and hi <= 1
    print(f"   step-150 pass@1, D32 minus B: {d.mean():+.2f}, 90% [{lo:+.2f}, {hi:+.2f}] -> "
          + ("equal accuracy" if equal else "equivalence not demonstrated"))
    survives = h["D32 vs B envelope"][1]
    if verdict == "moves the curve" and not survives:
        print("   VERDICT: moves the curve before correction, but not significant under Holm -> not claimed")
    elif verdict == "moves the curve" and equal:
        print("   VERDICT: joint win (Holm-significant)")
    elif verdict == "moves the curve":
        print("   VERDICT: moves the curve without demonstrated equal accuracy (accuracy trade-off)")
    else:
        print(f"   VERDICT: {verdict}")

    print("2. Mechanism contrasts (claim needs >= +1, 95% interval above 0, and Holm significance)")
    for other in contrasts:
        k = f"D32 vs {other} trajectory"
        t = tests[k]
        ok = "estimate" in t and t["estimate"] >= 1 and t["ci95"][0] > 0 and h[k][1]
        A.show(k, t, "supported" if ok else "not shown")
    print(f"   Holm over the {len(h)} registered tests:")
    for k, (p, ok) in sorted(h.items(), key=lambda x: x[1][0]):
        print(f"      {k}: p = {p:.4f} -> {'significant' if ok else 'not significant'}")
    print("   Pre-registration 12's own verdict for CM is reported unchanged in RESULTS_12; its p-value enters here only.")

    print("3. Reported without a rule")
    for other in contrasts:
        t = A.compare(res, other, envelope, A.SEEDS, args.draws, fresh())
        A.show(f"{other} vs B envelope", t, A.three_way(t))
    print("   Interpolation endpoint clamping (bootstrap points outside the reference's pass@1 range):")
    for k, c in clamps.items():
        if c.get("points"):
            print(f"      {k}: {c.get('clamped', 0)} of {c['points']} ({100 * c.get('clamped', 0) / c['points']:.1f}%)")

    print("Greedy test accuracy at step 150 (3,382 problems), mean over seeds")
    for a in arms:
        g = [res.get(("greedy", a, s, 150, 1.0)) for s in A.SEEDS]
        if all(x is not None for x in g):
            print(f"   {a}: {100 * np.mean([x.mean() for x in g]):.1f}")


if __name__ == "__main__":
    main()
