"""Pre-registration 18 analysis (rollout allocation at an equal budget). See PREREGISTRATION_18.md.

Usage: python analyze_v18.py [--draws 2000]
References (B, BASE) from Pre-registration 12's evaluation jobs, G32 from Pre-registration 14's; arms AC, AV,
HR from worth-v18e-* (SageMaker) or from a whole-node run's evaluation directory, results/<dir>/<ARM><seed>/eval
(--node-dirs). Shared bootstrap draws (a fresh generator with seed 0 per comparison).
"""

import argparse
import json
import re
from collections import defaultdict

import numpy as np

import analyze_v12 as A
import analyze_v14 as V14

ARMS = ("AC", "AV", "HR")
COORDS = (37.0, 40.0, 44.0)
FORECAST = {  # pass@16 gain over B at matched pass@1 0.37 / 0.40 / 0.44 (PREREGISTRATION_18_forecast_output.txt)
    "frozen law": {"AC": (3.91, 4.32, 4.79), "AV": (-1.09, -1.14, -1.17), "HR": (0.12, 0.16, 0.17)},
    "deep-sample counts": {"AC": (2.36, 2.55, 2.78), "AV": (-0.70, -0.72, -0.73), "HR": (0.06, 0.06, 0.09)},
}


def check_alignment(node_dirs):
    """Problem order of every saved evaluation text file (references, v18e jobs, node runs) must agree."""
    import gzip
    import hashlib
    files = [f for prefix in ("worth-v12e-E", "worth-v14e-E") for f in A.RES.glob(f"{prefix}*-s*-*/*/eval/*_texts.json.gz")]
    files += list(A.RES.glob("worth-v18e-*-s*-*/*/eval/*_texts.json.gz"))
    files += [f for d in node_dirs for f in (A.RES / d).glob("*/eval/*_texts.json.gz")]
    ref = None
    for f in files:
        qs = json.load(gzip.open(f))["problems"]
        h = hashlib.sha256(json.dumps([(q["question"], q["answer"]) for q in qs]).encode()).hexdigest()
        ref = ref or h
        if h != ref:
            raise ValueError(f"problem order differs in {f}")
    return len(files)


def read_eval(arm_dir, arm, seed, out):
    for f in (arm_dir / "eval").glob("step*_T*_eval.json") if (arm_dir / "eval").exists() else []:
        step, t = re.match(r"step(\d+)_T([\d.]+)_eval", f.name).groups()
        d = json.loads(f.read_text())
        if "counts" in d and d.get("n") == A.N:
            key = (arm, seed, int(step), round(float(t), 2))
            if key in out and not np.array_equal(out[key], np.array(d["counts"])):
                print(f"note: two evaluations of {key}; keeping the first ({f} ignored)")
                continue
            out[key] = np.array(d["counts"])
            if "greedy" in d:
                out[("greedy",) + key] = np.array(d["greedy"]["correct"])


def load(node_dirs):
    out = V14.load()
    for job in sorted(A.RES.glob("worth-v18e-*-s*-*"), key=lambda j: j.name[-15:]):
        seed = int(re.match(r"worth-v18e-\w+-s(\d+)-", job.name).group(1))
        for arm_dir in job.iterdir():
            if arm_dir.name in ARMS:
                read_eval(arm_dir, arm_dir.name, seed, out)
    for d in node_dirs:
        for arm_dir in (A.RES / d).iterdir() if (A.RES / d).exists() else []:
            m = re.fullmatch(r"([A-Z]+)(\d+)", arm_dir.name)
            if m and m.group(1) in ARMS:
                read_eval(arm_dir, m.group(1), int(m.group(2)), out)
    return out


def seed_mean_curve(res, arm, w, temps=False):
    """Seed-mean (pass@1, pass@16) along the trajectory (base model first), or along B's step-150 temperatures."""
    pts = []
    for st in ((150,) if temps else (0,) + A.STEPS):
        for t in ((1.0, 1.15, 1.3) if temps else (1.0,)):
            ms = [A.met(res[("BASE", 0, 0, 1.0)] if st == 0 else res[(arm, s, st, t)], w) for s in A.SEEDS]
            pts.append((np.mean([m[0] for m in ms]), np.mean([m[1] for m in ms]), st + t))
    pts.sort(key=lambda x: (x[0], x[2]))
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def forecast_check(res, arm):
    """Rule 4: the arm's seed-mean pass@16 minus B's seed-mean envelope at pass@1 = 37, 40, 44."""
    w = np.ones(len(res[("BASE", 0, 0, 1.0)]))
    ax, ay = seed_mean_curve(res, arm, w)
    curves = [seed_mean_curve(res, "B", w), seed_mean_curve(res, "B", w, temps=True)]
    gaps = {}
    for x in COORDS:
        ref = [float(np.interp(x, bx, by)) for bx, by in curves if bx.min() <= x <= bx.max()]
        if ax.min() <= x <= ax.max() and ref:
            gaps[x] = float(np.interp(x, ax, ay)) - max(ref)
    return gaps


def allocation_report(arm):
    """Mean group size by pilot count, from the training logs."""
    sizes = defaultdict(list)
    for f in list(A.RES.glob(f"worth-v18-{arm}-s*-*/{arm}/baseline_train.json")) + list(A.RES.glob(f"*/{arm}[0-9]*/baseline_train.json")):
        for rec in json.loads(f.read_text()):
            a = rec.get("alloc")
            if a:
                for c, n in zip(a["first_counts"], a["sizes"]):
                    sizes[c].append(n)
    return {c: (np.mean(v), len(v)) for c, v in sorted(sizes.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--node-dirs", default="", help="comma-separated results/ subdirectories of whole-node evaluations")
    args = ap.parse_args()
    fresh = lambda: np.random.default_rng(0)
    node_dirs = [d for d in args.node_dirs.split(",") if d]
    print(f"problem alignment: {check_alignment(node_dirs)} evaluation files share one problem order")
    res = load(node_dirs)
    need = [("BASE", 0, 0, 1.0)] + [(a, s, st, 1.0) for a in ("B",) + ARMS for s in A.SEEDS for st in A.STEPS] + \
           [("B", s, 150, t) for s in A.SEEDS for t in (1.15, 1.3)]
    missing = [k for k in need if k not in res]
    if missing:
        print(f"incomplete evaluation; missing {missing}")
        return
    envelope = lambda r, s, w: A.reference(r, s, w, True)
    tests, clamps = {}, {}

    def run(name, arm, ref):
        clamps[name] = {}
        tests[name] = A.compare(res, arm, ref, A.SEEDS, args.draws, fresh(), stats=clamps[name])
        return tests[name]

    prim = run("1. AC vs B envelope", "AC", envelope)
    run("2. AC vs AV trajectory", "AC", V14.traj_ref("AV"))
    run("3. AC vs HR trajectory", "AC", V14.traj_ref("HR"))
    h = V14.holm(tests)

    print(f"Rule 1: AC against B's trajectory-and-temperature envelope ({len(res[('BASE', 0, 0, 1.0)])} problems, seeds {A.SEEDS})")
    v = A.three_way(prim)
    if v in ("moves the curve", "worse") and not h["1. AC vs B envelope"][1]:
        v = f"inconclusive ({v} before correction, not significant under Holm)"
    A.show("AC", prim, v + ("; fresh confirmation required" if v == "moves the curve" else ""))
    n = len(res[("BASE", 0, 0, 1.0)])
    d = np.mean([res[("AC", s, 150, 1.0)] / A.N - res[("B", s, 150, 1.0)] / A.N for s in A.SEEDS], axis=0) * 100
    r = fresh()
    boot = [d[r.integers(0, n, n)].mean() for _ in range(args.draws)]
    lo, hi = np.percentile(boot, [5, 95])
    print(f"   step-150 pass@1, AC minus B: {d.mean():+.2f}, 90% [{lo:+.2f}, {hi:+.2f}]")
    for k, label in (("2. AC vs AV trajectory", "Rule 2: allocation target matters"), ("3. AC vs HR trajectory", "Rule 3: beyond HORA")):
        t = tests[k]
        ok = "estimate" in t and t["estimate"] >= 1 and t["ci95"][0] > 0 and h[k][1]
        print(label)
        A.show(k[3:], t, "supported" if ok else "not shown")
    print(f"Holm over rules 1-3:")
    for k, (p, ok) in sorted(h.items(), key=lambda x: x[1][0]):
        print(f"   {k}: p = {p:.4f} -> {'significant' if ok else 'not significant'}")
    print("   Interpolation endpoint clamping:")
    for k, c in clamps.items():
        if c.get("points"):
            print(f"      {k}: {c.get('clamped', 0)} of {c['points']} ({100 * c.get('clamped', 0) / c['points']:.1f}%)")

    print("Rule 4 (reported): forecast accuracy at matched pass@1 37 / 40 / 44, seed-mean curves")
    observed = {}
    for a in ARMS:
        gaps = forecast_check(res, a)
        observed[a] = np.mean(list(gaps.values())) if gaps else None
        if not gaps:
            print(f"   {a}: insufficient overlap")
            continue
        print(f"   {a}: observed " + ", ".join(f"{x:.0f}: {g:+.2f}" for x, g in gaps.items()) + f" (mean {observed[a]:+.2f})")
        for variant, fc in FORECAST.items():
            f_mean = np.mean([fc[a][COORDS.index(x)] for x in gaps])
            ok = abs(observed[a] - f_mean) <= 1.5
            print(f"      {variant}: forecast mean {f_mean:+.2f} -> {'confirmed' if ok else 'not confirmed'} (|diff| {abs(observed[a] - f_mean):.2f})")
    if all(observed[a] is not None for a in ARMS):
        order = sorted(ARMS, key=lambda a: -observed[a])
        print(f"   observed ordering {' > '.join(order)} (forecast AC > HR > AV)")

    print("Reported without a rule")
    for a in ARMS:
        alloc = allocation_report(a)
        if alloc:
            print(f"   {a} mean group size by pilot count: " + ", ".join(f"c={c}: {m:.1f} ({k} groups)" for c, (m, k) in alloc.items()))
    for a in ("AV", "HR"):
        t = A.compare(res, a, envelope, A.SEEDS, args.draws, fresh())
        A.show(f"{a} vs B envelope", t, A.three_way(t))
    if all(("G32", s, st, 1.0) in res for s in A.SEEDS for st in A.STEPS):
        for a in ARMS:
            t = A.compare(res, a, V14.traj_ref("G32"), A.SEEDS, args.draws, fresh())
            A.show(f"{a} vs G32 trajectory", t, "reported")
    base = res[("BASE", 0, 0, 1.0)]
    strata = {"c=0": base == 0, "c=1-4": (base >= 1) & (base <= 4), "c=5-16": (base >= 5) & (base <= 16), "c>16": base > 16}
    print("   step-150 pass@16 by base-count stratum (32 base samples), mean over seeds:")
    for a in ("B",) + ARMS:
        row = []
        for name, m in strata.items():
            v = np.mean([A.met(res[(a, s, 150, 1.0)], m.astype(float))[1] for s in A.SEEDS])
            row.append(f"{name} {v:.1f}")
        print(f"      {a}: " + ", ".join(row) + f" ({', '.join(f'{k}: {int(m.sum())}' for k, m in strata.items())} problems)")
    print("Greedy test accuracy at step 150 (3,382 problems), mean over seeds")
    for a in ("B",) + ARMS:
        g = [res.get(("greedy", a, s, 150, 1.0)) for s in A.SEEDS]
        if all(x is not None for x in g):
            print(f"   {a}: {100 * np.mean([x.mean() for x in g]):.1f}")


if __name__ == "__main__":
    main()
