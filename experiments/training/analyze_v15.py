"""Pre-registration 15 analysis (matched-accuracy audit of BBG and MaxRL). See PREREGISTRATION_15.md.

Usage: python analyze_v15.py [--draws 2000]
References (B, BASE) from Pre-registration 12's evaluation jobs; arms BBG and MXR from worth-v15e-*; D32 from
worth-v14e-* when present. Shared bootstrap draws (a fresh generator with seed 0 per comparison).
"""

import argparse
import re

import numpy as np

import analyze_v12 as A
import analyze_v14 as V14


def load():
    out = V14.load()
    for job in sorted(A.RES.glob("worth-v15e-E*-s*-*"), key=lambda j: j.name[-15:]):
        seed = int(re.match(r"worth-v15e-E\w*-s(\d+)-", job.name).group(1))
        for arm_dir in job.iterdir():
            for f in (arm_dir / "eval").glob("step*_T*_eval.json") if (arm_dir / "eval").exists() else []:
                step, t = re.match(r"step(\d+)_T([\d.]+)_eval", f.name).groups()
                d = A.json.loads(f.read_text())
                if "counts" in d and d.get("n") == A.N:
                    key = (arm_dir.name, seed, int(step), round(float(t), 2))
                    out[key] = np.array(d["counts"])
                    if "greedy" in d:
                        out[("greedy",) + key] = np.array(d["greedy"]["correct"])
    return out


def paired(res, arm, metric, draws):
    """Step-150 metric of the arm minus B's, pooled over seeds, with a 90% interval over problems."""
    def per_problem(c):
        return c / A.N if metric == "pass@1" else np.array([1 - A.comb(A.N - int(x), A.K) / A.comb(A.N, A.K) for x in c])
    d = np.mean([per_problem(res[(arm, s, 150, 1.0)]) - per_problem(res[("B", s, 150, 1.0)]) for s in A.SEEDS], axis=0) * 100
    r, n = np.random.default_rng(0), len(d)
    boot = [d[r.integers(0, n, n)].mean() for _ in range(draws)]
    return d.mean(), np.percentile(boot, [5, 95])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    print(f"problem alignment: {A.check_alignment(('worth-v12e-', 'worth-v14e-', 'worth-v15e-'))} evaluation files share one problem order")
    res = load()
    arms = ["BBG", "MXR", "WRF"]
    need = [("BASE", 0, 0, 1.0)] + [(a, s, st, 1.0) for a in ["B"] + arms for s in A.SEEDS for st in A.STEPS] + \
           [("B", s, 150, t) for s in A.SEEDS for t in (1.15, 1.3)]
    missing = [k for k in need if k not in res]
    if missing:
        print(f"incomplete evaluation; missing {missing}")
        return
    envelope = lambda r, s, w: A.reference(r, s, w, True)
    tests, clamps = {}, {}
    for a in arms:
        clamps[a] = {}
        tests[a] = A.compare(res, a, envelope, A.SEEDS, args.draws, np.random.default_rng(0), stats=clamps[a])
    h = V14.holm(tests)
    print("1. Against B's trajectory-and-temperature envelope at matched pass@1 (directional verdicts need Holm)")
    for a in arms:
        v = A.three_way(tests[a])
        if v in ("moves the curve", "worse") and not h[a][1]:
            v = f"inconclusive ({v} before correction, not significant under Holm)"
        A.show(a, tests[a], v + ("; fresh confirmation required" if v == "moves the curve" else ""))
        c = clamps[a]
        print(f"      clamping: {c.get('clamped', 0)} of {c.get('points', 0)}")
    print("2. Fixed-step reading next to the matched-accuracy reading (step 150, minus B, pooled over seeds)")
    for a in arms:
        (p16, ci16), (p1, ci1) = paired(res, a, "pass@16", args.draws), paired(res, a, "pass@1", args.draws)
        print(f"   {a}: pass@16 {p16:+.2f} [{ci16[0]:+.2f}, {ci16[1]:+.2f}]; pass@1 {p1:+.2f} [{ci1[0]:+.2f}, {ci1[1]:+.2f}]"
              f" -> at matched accuracy {tests[a].get('estimate', float('nan')):+.2f}")
    print("Reported: BBG against W-REINFORCE's trajectory (the gate's own contribution)")
    r = A.compare(res, "BBG", V14.traj_ref("WRF"), A.SEEDS, args.draws, np.random.default_rng(0))
    A.show("BBG vs WRF trajectory", r, "reported")
    print("Reported: D32 against each arm's trajectory (if Pre-registration 14 is evaluated)")
    if all((("D32", s, st, 1.0) in res) for s in A.SEEDS for st in A.STEPS):
        for a in arms:
            r = A.compare(res, "D32", V14.traj_ref(a), A.SEEDS, args.draws, np.random.default_rng(0))
            A.show(f"D32 vs {a} trajectory", r, "reported")
    print("Greedy test accuracy at step 150, mean over seeds")
    for a in ["B"] + arms:
        g = [res.get(("greedy", a, s, 150, 1.0)) for s in A.SEEDS]
        if all(x is not None for x in g):
            print(f"   {a}: {100 * np.mean([x.mean() for x in g]):.1f}")


if __name__ == "__main__":
    main()
