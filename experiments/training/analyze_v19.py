"""Pre-registration 19 analysis (human solutions as external information). See PREREGISTRATION_19.md.

Usage: python analyze_v19.py [--draws 2000] [--node-dirs DIR,...]
Arms HP, HQ, HS from worth-v19e-* (SageMaker) or whole-node evaluation directories results/<dir>/<ARM><seed>/eval;
S32 and the references from Pre-registrations 12 and 14; the long Dr. GRPO runs (B13, B14, B15) from
worth-v17Le-* and Pre-registration 17's Pre-registration-12-step evaluations. Shared bootstrap draws.
"""

import argparse
import json
import re

import numpy as np

import analyze_v12 as A
import analyze_v14 as V14
from analyze_v18 import read_eval

ARMS = ("HP", "HQ", "HS")


def load(node_dirs):
    out = V14.load()
    for job in sorted(A.RES.glob("worth-v19e-*-s*-*"), key=lambda j: j.name[-15:]):
        seed = int(re.match(r"worth-v19e-\w+-s(\d+)-", job.name).group(1))
        for arm_dir in job.iterdir():
            if arm_dir.name in ARMS:
                read_eval(arm_dir, arm_dir.name, seed, out)
    for d in node_dirs:
        for arm_dir in (A.RES / d).iterdir() if (A.RES / d).exists() else []:
            m = re.fullmatch(r"([A-Z]+)(\d+)", arm_dir.name)
            if m and m.group(1) in ARMS:
                read_eval(arm_dir, m.group(1), int(m.group(2)), out)
    for job in A.RES.glob("worth-v17Le-*"):  # long runs: steps beyond 150 of seeds 13 (AWS), 14 and 15
        for arm_dir in job.iterdir():
            m = re.fullmatch(r"B(\d+)", arm_dir.name)
            if m:
                read_eval(arm_dir, "LONG", int(m.group(1)), out)
    for job in A.RES.glob("worth-v17L-*"):
        for arm_dir in job.iterdir():
            m = re.fullmatch(r"B(\d+)", arm_dir.name)
            if m:
                read_eval(arm_dir, "LONG", int(m.group(1)), out)
    return out


def own_traj(arm):
    """An arm's trajectory starting from its own post-fine-tuning checkpoint (step 0)."""
    def ref(r, s, w):
        px, py = A.curve([(r[(arm, s, st, 1.0)], st) for st in (0,) + A.STEPS], w)
        value = lambda x: float(np.interp(x, px, py)) if px.min() <= x <= px.max() else None
        return value, px.min(), px.max()
    return ref


def long_traj(res, seed):
    steps = sorted(st for (a, s, st, t) in [k for k in res if len(k) == 4] if a == "LONG" and s == seed and t == 1.0)

    def ref(r, s, w):
        px, py = A.curve([(r[("BASE", 0, 0, 1.0)], 0)] + [(r[("LONG", seed, st, 1.0)], st) for st in steps], w)
        value = lambda x: float(np.interp(x, px, py)) if px.min() <= x <= px.max() else None
        return value, px.min(), px.max()
    return ref, steps


def stratum_gap(res, a, b, mask, draws):
    """Step-150 pass@16 of a minus b on the masked problems, pooled over seeds, 90% bootstrap interval."""
    pk = lambda c: np.array([1 - A.comb(A.N - int(x), A.K) / A.comb(A.N, A.K) for x in c])
    d = np.mean([pk(res[(a, s, 150, 1.0)]) - pk(res[(b, s, 150, 1.0)]) for s in A.SEEDS], axis=0)[mask] * 100
    r = np.random.default_rng(0)
    boot = [d[r.integers(0, len(d), len(d))].mean() for _ in range(draws)]
    return d.mean(), np.percentile(boot, [5, 95])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--node-dirs", default="")
    args = ap.parse_args()
    fresh = lambda: np.random.default_rng(0)
    res = load([d for d in args.node_dirs.split(",") if d])
    need = [("BASE", 0, 0, 1.0)] + [(a, s, st, 1.0) for a in ARMS + ("S32",) for s in A.SEEDS for st in (0,) + A.STEPS]
    missing = [k for k in need if k not in res]
    if missing:
        print(f"incomplete evaluation; missing {missing}")
        return
    retained = json.loads((A.RES / "p19_retained.json").read_text())
    c0_kept = min(retained[f"human_swap s{s}"]["c0"][1] for s in A.SEEDS)

    print("Rule 1: HQ against HP's trajectory (from HP's post-fine-tuning step 0) at matched pass@1")
    clamps = {}
    t = A.compare(res, "HQ", own_traj("HP"), A.SEEDS, args.draws, fresh(), stats=clamps)
    frac = clamps.get("clamped", 0) / clamps["points"] if clamps.get("points") else 0.0
    problems = []
    if "estimate" not in t:
        problems.append("fewer than two HQ checkpoints in HP's pass@1 range for some seed")
    if frac > 0.10:
        problems.append(f"{100 * frac:.1f}% of bootstrap points clamped")
    if c0_kept < 400:
        problems.append(f"only {c0_kept} of 487 c = 0 solutions retained")
    if problems:
        verdict = "not interpretable (" + "; ".join(problems) + ")"
    else:
        verdict = "supported" if t["estimate"] >= 1 and t["ci95"][0] > 0 else "not shown"
    A.show("HQ vs HP", t, verdict)
    print(f"   clamping {clamps.get('clamped', 0)} of {clamps.get('points', 0)}; c = 0 solutions retained (min over seeds): {c0_kept} of 487")
    base = res[("BASE", 0, 0, 1.0)]
    g, ci = stratum_gap(res, "HQ", "HP", base == 0, args.draws)
    print(f"   step-150 pass@16 on the {int((base == 0).sum())} evaluation problems with no base success in 32, HQ minus HP: "
          f"{g:+.2f}, 90% [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    for a in ARMS:
        ep = np.mean([retained[f"human_{dict(HP='positive', HQ='swap', HS='solvable')[a]} s{s}"]["epochs_at_2.46M"] for s in A.SEEDS])
        print(f"   {a}: {ep:.2f} epochs over its retained solutions")

    print("Rule 2 (reported): HS against S32's trajectory (from S32's post-fine-tuning step 0)")
    t = A.compare(res, "HS", own_traj("S32"), A.SEEDS, args.draws, fresh())
    A.show("HS vs S32", t, A.three_way(t))

    print("Rule 3 (descriptive): against the long Dr. GRPO trajectories at matched pass@1")
    long_seeds = sorted({k[1] for k in res if len(k) == 4 and k[0] == "LONG"})
    for ls in long_seeds:
        ref, steps = long_traj(res, ls)
        if not all(("LONG", ls, st, 1.0) in res for st in A.STEPS):
            print(f"   long seed {ls}: steps {steps} only (steps 50-150 missing); skipped")
            continue
        w = np.ones(len(res[("BASE", 0, 0, 1.0)]))
        value, lo, hi = ref(res, ls, w)
        for a in ARMS + ("S32",):
            # pooled over every checkpoint in range, post-fine-tuning step 0 included (as in the interim report)
            g = [A.met(res[(a, s, st, 1.0)], w) for s in A.SEEDS for st in (0,) + A.STEPS]
            gi = [p16 - value(p1) for p1, p16 in g if lo <= p1 <= hi]
            pooled = f"pooled {np.mean(gi):+.2f} ({len(gi)}/{len(g)} checkpoints in range)" if gi else "no checkpoint in range"
            print(f"   {a} vs long seed {ls} (steps {steps[0]}-{steps[-1]}): {pooled}")
            t = A.compare(res, a, ref, A.SEEDS, args.draws, fresh())
            A.show(f"   {a}, steps 50-150, seed-averaged", t, "descriptive")

    print("Reported: pass@1 / pass@16 at step 0 (after fine-tuning) and step 150, mean over seeds")
    w = np.ones(len(base))
    for a in ARMS + ("S32",):
        row = []
        for st in (0, 150):
            ms = [A.met(res[(a, s, st, 1.0)], w) for s in A.SEEDS]
            row.append(f"step {st}: {np.mean([m[0] for m in ms]):.1f} / {np.mean([m[1] for m in ms]):.1f}")
        print(f"   {a}: " + "; ".join(row))
    print("Greedy test accuracy at step 150 (3,382 problems), mean over seeds")
    for a in ("B",) + ARMS + ("S32",):
        gr = [res.get(("greedy", a, s, 150, 1.0)) for s in A.SEEDS]
        if all(x is not None for x in gr):
            print(f"   {a}: {100 * np.mean([x.mean() for x in gr]):.1f}")


if __name__ == "__main__":
    main()
