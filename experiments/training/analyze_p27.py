"""Registered analysis of Pre-registration 27 (longer and stronger training: LH1, LH3; 1.5B, 1,024 tokens).

A (primary): pass@256 at step 1,000 minus the base's pass@256 (256 samples each; per problem 1[c > 0]); met if the
  estimate is >= +1 and the Holm-adjusted 95% interval lies above 0 (one-sided p < 0.025, Holm over the two arms).
  Reported: step 150, the change 150 -> 1,000, problems solved by the arm but not the base and the reverse.
B (reported): the frozen law (tau = -0.20) from BASE15's 32-sample counts predicts pass@16 at each 32-sample checkpoint's
  pass@1 (strict reachability); departure at step 500 or 1,000: >= +2 with a Holm-adjusted 95% interval above 0.
  The bootstrap recomputes the prior, the initial state and the trajectory in every draw.
C (reported): Spearman correlation between the base's and the arm's 256-sample success rates; success on base-zero
  problems (0 of 256).
Verdict per arm: supported if A is met; not supported if A is not met and step-1,000 pass@1 exceeds B1k8 step 150's by
  at least 3 points; otherwise inconclusive.
Usage: python analyze_p27.py [--draws 2000]
"""

import argparse
import glob
import json
import math
from math import comb

import numpy as np

import law


def load(pattern):
    f = glob.glob(pattern)
    assert len(f) == 1, (pattern, f)
    d = json.load(open(f[0]))
    return np.array(d["counts"]), d["n"], d


def pk(c, n, k):
    return np.array([1 - comb(n - int(x), k) / comb(n, k) for x in c])


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def law_curve(c0, n0):
    prior = law.fit_prior(c0, n0)
    return law.trajectory(law.initial_theta(c0, n0, prior), -0.20)


def law_pred(traj, p1):
    try:
        return law.pass_k_true(law.at_pass1(traj, p1, strict=True), 16)
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(27)
    cb, nb, _ = load("results/worth-p28-public-*/BASE15n256/eval/step0_T1.00_eval.json")
    c32, n32, _ = load("results/worth-v15v-all-*/BASE15/eval/step0_T1.00_eval.json")
    b150, _, _ = load("results/worth-v15v-all-*/B1k8/eval/step150_T1.00_eval.json")
    P = len(cb)
    idx = [rng.integers(0, P, P) for _ in range(a.draws)]
    print("Pre-registration 27 (base pass@256 %.2f, 256 samples)" % (100 * np.mean(cb > 0)))
    pA, rows = [], {}
    for arm in ("LH1", "LH3"):
        c1k, n1k, _ = load(f"results/worth-p27-long-*/{arm}n256/eval/step1000_T1.00_eval.json")
        c150, _, _ = load(f"results/worth-p27-long-*/{arm}n256/eval/step150_T1.00_eval.json")
        s_arm, s_base, s150 = (c1k > 0).astype(float), (cb > 0).astype(float), (c150 > 0).astype(float)
        d = s_arm - s_base
        est = 100 * d.mean()
        boots = np.array([100 * d[i].mean() for i in idx])
        p_one = float(np.mean(boots <= 0))
        pA.append(p_one)
        rows[arm] = {"est": est, "ci95": np.percentile(boots, [2.5, 97.5]), "p": p_one,
                     "s150": 100 * (s150 - s_base).mean(), "chg": 100 * (s_arm - s150).mean(),
                     "arm_only": int(((c1k > 0) & (cb == 0)).sum()), "base_only": int(((c1k == 0) & (cb > 0)).sum()),
                     "rho150": spearman(cb / nb, c150 / 256), "rho1k": spearman(cb / nb, c1k / 256),
                     "zero": (100 * (c150[cb == 0] / 256).mean(), 100 * (c1k[cb == 0] / 256).mean(), int((cb == 0).sum()))}
    adj = holm(np.array(pA))
    traj0 = law_curve(c32, n32)
    trajs = [law_curve(c32[i], n32) for i in idx[: a.draws]]
    pB, keyB = [], []
    for arm, r, ap_ in zip(("LH1", "LH3"), rows.values(), adj):
        met = r["est"] >= 1 and ap_ < 0.025
        print(f"   {arm} A: pass@256 step 1,000 minus base {r['est']:+.2f}, 95% [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}], "
              f"one-sided p {r['p']:.4f} (Holm {ap_:.4f}) -> {'met' if met else 'not met'}; step 150 minus base {r['s150']:+.2f}; "
              f"change 150 -> 1,000 {r['chg']:+.2f}; solved by arm not base {r['arm_only']}, base not arm {r['base_only']}")
        print(f"      C: Spearman(base, arm) step 150 {r['rho150']:.3f}, step 1,000 {r['rho1k']:.3f}; success on the "
              f"{r['zero'][2]} base-zero problems {r['zero'][0]:.2f}% (step 150) -> {r['zero'][1]:.2f}% (step 1,000)")
        r["met"] = met
        for step in (50, 100, 150, 300, 500, 700, 1000):
            c, n, _ = load(f"results/worth-p27-long-*/{arm}/eval/step{step}_T1.00_eval.json")
            p1, p16 = c.mean() / n, pk(c, n, 16).mean()
            pred = law_pred(traj0, p1)
            line = f"      B: step {step:4d} pass@1 {100 * p1:5.1f} pass@16 {100 * p16:5.1f} law {100 * pred:5.1f} diff {100 * (p16 - pred):+5.2f}"
            if step in (500, 1000) and math.isfinite(pred):
                bd = []
                for i, tr in zip(idx, trajs):
                    q = law_pred(tr, c[i].mean() / n)
                    if math.isfinite(q):
                        bd.append(100 * (pk(c[i], n, 16).mean() - q))
                bd = np.array(bd)
                pB.append(float(np.mean(bd <= 0)))
                keyB.append((arm, step, 100 * (p16 - pred), np.percentile(bd, [2.5, 97.5]), len(bd)))
                line += f", 95% [{np.percentile(bd, 2.5):+.2f}, {np.percentile(bd, 97.5):+.2f}] ({len(bd)} scorable draws)"
            print(line)
        p1k = load(f"results/worth-p27-long-*/{arm}/eval/step1000_T1.00_eval.json")
        moved = 100 * (p1k[0].mean() / p1k[1] - b150.mean() / 32)
        r["moved"] = moved
    if pB:
        adjB = holm(np.array(pB))
        for (arm, step, est, ci, nd), q in zip(keyB, adjB):
            print(f"   B departure {arm} step {step}: {est:+.2f} -> {'departure' if est >= 2 and q < 0.025 else 'no departure'} (Holm {q:.4f})")
    for arm, r in rows.items():
        v = "supported" if r["met"] else ("not supported" if r["moved"] >= 3 else "inconclusive")
        print(f"   VERDICT {arm}: {v} (step-1,000 pass@1 minus B1k8 step 150: {r['moved']:+.2f})")


if __name__ == "__main__":
    main()
