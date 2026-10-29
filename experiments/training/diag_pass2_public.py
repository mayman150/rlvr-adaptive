"""Diagnostic (not registered): pass@2 (spread of per-problem success) of the published checkpoints of
Pre-registration 28 at matched pass@1, against our Dr. GRPO seeds' trajectories (base and saved steps, T = 1; mean over
seeds whose trajectory covers the pass@1) at the same budget: 1.5B at 1,024 tokens (B1k 8-10), 7B at 2,048 (B7 8-9);
and against the frozen law at 3,072 tokens (1.5B; no Dr. GRPO run there). At matched mean success the pass@2 change
equals minus the change in Var_x(p_x). Control: each Dr. GRPO seed against the other seeds' trajectories.
Usage: python diag_pass2_public.py [--draws 1000]
"""
import argparse
import glob
from math import comb

import numpy as np

import analyze_p28 as P
import law


def pk(c, n, k):
    return float(np.mean([1 - comb(n - int(x), k) / comb(n, k) for x in c]))


def ref_traj(seeds, u, idx, k):
    vals = []
    for traj, _ in seeds:
        pts = sorted((c[idx].mean() / n, pk(c[idx], n, k)) for c, n in traj)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if xs[0] <= u <= xs[-1]:
            vals.append(float(np.interp(u, xs, ys)))
    return float(np.mean(vals)) if vals else None


def ref_law(c0, n0, u, idx, k):
    c = c0[idx]
    tr = law.trajectory(law.initial_theta(c, n0, law.fit_prior(c, n0)), -0.20)
    try:
        return law.pass_k_true(law.at_pass1(tr, u, strict=True), k)
    except ValueError:
        return None


def gain(c, n, ref, P_, draws, rng, k=2):
    full = np.arange(P_)
    r0 = ref(c[full].mean() / n, full, k)
    if r0 is None:
        return None
    est = 100 * (pk(c, n, k) - r0)
    boots = []
    for _ in range(draws):
        i = rng.integers(0, P_, P_)
        r = ref(c[i].mean() / n, i, k)
        if r is not None:
            boots.append(100 * (pk(c[i], n, k) - r))
    lo, hi = np.percentile(boots, [5, 95])
    return est, lo, hi, 1 - len(boots) / draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1000)
    a = ap.parse_args()
    s15, s7 = P.seeds_15(), P.seeds_7()
    groups = [("1.5B, 1,024 tokens", "PUB_*_1k", s15, None), ("7B, 2,048 tokens", "PUB_*_2k", s7, None),
              ("1.5B, 3,072 tokens (vs frozen law)", "PUB_*_3k", None,
               P.counts(glob.glob("results/worth-p28-public-*/BASE15_3k/eval/step0_T1.00_eval.json")[0])[:2])]
    for title, pat, seeds, base in groups:
        print(f"\n== {title}: pass@2 minus reference at matched pass@1 (pp); implied change in Var_x(p) = -gain/100")
        if seeds is not None:
            ref = lambda u, idx, k, s=seeds: ref_traj(s, u, idx, k)
            for j in range(len(seeds)):  # control: one seed's saved steps against the other seeds' trajectories
                others = [s for i, s in enumerate(seeds) if i != j]
                rng = np.random.default_rng(2)
                res = []
                for c, n in seeds[j][0][1:]:
                    g = gain(c, n, lambda u, idx, k, o=others: ref_traj(o, u, idx, k), len(c), 200, rng)
                    if g:
                        res.append(g[0])
                print(f"   control: Dr. GRPO seed {j} steps vs other seeds: mean {np.mean(res):+.2f} over {len(res)} checkpoints")
        else:
            c0, n0 = base
            ref = lambda u, idx, k, c0=c0, n0=n0: ref_law(c0, n0, u, idx, k)
        for f in sorted(glob.glob(f"results/worth-p28*-public-*/{pat}/eval/step0_T1.00_eval.json")):
            name = f.split("/")[-3][4:]
            c, n, d = P.counts(f)
            if d.get("truncated_rate", 0) > 0.9:
                print(f"   {name:58s} protocol failure")
                continue
            g = gain(c, n, ref, len(c), a.draws, np.random.default_rng(28))
            lab = P.label(name)
            if g is None:
                print(f"   {name:58s} pass@1 {100 * c.mean() / n:5.1f}  outside the reference's range [{lab}]")
                continue
            est, lo, hi, drop = g
            print(f"   {name:58s} pass@1 {100 * c.mean() / n:5.1f}  pass@2 gain {est:+.2f} 90% [{lo:+.2f}, {hi:+.2f}]"
                  f"{' (dropped %.0f%%)' % (100 * drop) if drop > 0.1 else ''}  [{lab}]")


if __name__ == "__main__":
    main()
