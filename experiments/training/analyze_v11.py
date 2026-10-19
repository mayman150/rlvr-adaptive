"""Pre-registration 11 analysis: is the shared curve a sharpening curve? See PREREGISTRATION_11.md.

Usage: python analyze_v11.py [--draws 2000]
Results: results/worth-v11-BASE-s0-*/eval/step0_T<t>_eval.json and
results/worth-v11-PSR{a,b}-s<seed>-*/eval/step<N>_T<t>_eval.json (fetched by fetch_live.sh).
"""

import argparse
import json
import math
import re
from math import comb
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"
SEEDS = (5, 6, 7)
N, K = 32, 16
BASE_T = (0.5, 0.7, 0.85, 1.0)
HIGH_T = (1.15, 1.3, 1.45, 1.6)


def load():
    """{(seed, step, T): result}; the base model is seed 0, step 0. Later jobs win on overlap."""
    out = {}
    jobs = sorted((j for j in RES.glob("worth-v11-*") if not j.name.startswith("worth-v11smoke")), key=lambda j: j.name[-15:])
    for job in jobs:
        m = re.match(r"worth-v11-(\w+?)-s(\d+)-", job.name)
        if not m:
            continue
        seed = int(m.group(2))
        for f in (job / "eval").glob("step*_T*_eval.json"):
            step, t = re.match(r"step(\d+)_T([\d.]+)_eval", f.name).groups()
            d = json.loads(f.read_text())
            if "counts" in d:
                out[(seed, int(step), round(float(t), 2))] = d
    return out


def metrics(counts, w):
    c = np.asarray(counts)
    p1 = c / N
    pk = np.array([1.0 - comb(N - int(x), K) / comb(N, K) for x in c])
    return 100 * float(w @ p1) / w.sum(), 100 * float(w @ pk) / w.sum()


def trajectory(res, seed, w):
    """(pass@1, pass@16, step) of the T = 1 trajectory: base, steps 50, 100, 150."""
    pts = [(*metrics(res[(0, 0, 1.0)]["counts"], w), 0)]
    pts += [(*metrics(res[(seed, s, 1.0)]["counts"], w), s) for s in (50, 100, 150)]
    pts.sort(key=lambda x: (x[0], x[2]))
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def diffs(res, w, qualifying):
    out = {}
    for seed in SEEDS:
        px, py = trajectory(res, seed, w)
        out[seed] = [metrics(res[(seed, 150, t)]["counts"], w)[1] - float(np.interp(metrics(res[(seed, 150, t)]["counts"], w)[0], px, py))
                     for t in qualifying[seed]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    res = load()
    expected = [(0, 0, t) for t in BASE_T] + [(s, st, 1.0) for s in SEEDS for st in (50, 100, 150)] + \
               [(s, 150, t) for s in SEEDS for t in HIGH_T]
    missing = [e for e in expected if e not in res]
    print("Points (pass@1, pass@16, prefix entropy, boxed rate, mean length):")
    w1 = None
    for key in sorted(res):
        d = res[key]
        w1 = np.ones(len(d["counts"]))
        p1, p16 = metrics(d["counts"], w1)
        print(f"   seed {key[0]} step {key[1]:3d} T {key[2]:.2f}: {p1:5.1f} {p16:5.1f}  H {d.get('prefix_entropy', math.nan):.3f}  "
              f"boxed {100 * d.get('boxed_rate', math.nan):.0f}%  len {d.get('mean_length_chars', math.nan):.0f}")
    if missing:
        print(f"\nverdict: incomplete evaluation; missing {missing}")
        return
    n_prob = len(res[(0, 0, 1.0)]["counts"])
    ones = np.ones(n_prob)
    qualifying = {}
    for seed in SEEDS:
        px, _ = trajectory(res, seed, ones)
        qualifying[seed] = [t for t in HIGH_T if px.min() <= metrics(res[(seed, 150, t)]["counts"], ones)[0] <= px.max()]
    print("\nqualifying temperatures:", qualifying)
    if any(not q for q in qualifying.values()):
        print("verdict: insufficient overlap")
        return
    full = diffs(res, ones, qualifying)
    per_seed = {s: float(np.mean(v)) for s, v in full.items()}
    est = float(np.mean(list(per_seed.values())))
    boot, boot_pts = [], []
    for _ in range(args.draws):
        w = np.bincount(rng.integers(0, n_prob, n_prob), minlength=n_prob).astype(float)
        d = diffs(res, w, qualifying)
        boot.append(np.mean([np.mean(v) for v in d.values()]))
        boot_pts.append(np.concatenate([d[s] for s in SEEDS]))
    lo90, hi90 = np.percentile(boot, [5, 95])
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    if est >= 2 and lo95 > 0:
        verdict = "separable (a late checkpoint at a higher temperature beats the training trajectory)"
    elif lo90 >= -2 and hi90 <= 2:
        verdict = "same curve"
    elif est <= -2 and hi95 < 0:
        verdict = "below (raising the temperature costs more coverage than training less)"
    else:
        verdict = "inconclusive"
    print(f"pooled D {est:+.2f}, 90% [{lo90:+.2f}, {hi90:+.2f}], 95% [{lo95:+.2f}, {hi95:+.2f}]; per seed "
          + ", ".join(f"{s}: {v:+.2f}" for s, v in per_seed.items()))
    flat, bp = np.concatenate([full[s] for s in SEEDS]), np.array(boot_pts)
    se = bp.std(axis=0, ddof=1)
    q = np.percentile(np.max(np.abs(bp - flat) / np.where(se > 0, se, np.inf), axis=1), 90)
    labels = [(s, t) for s in SEEDS for t in qualifying[s]]
    print("simultaneous 90% band: " + "; ".join(f"s{s} T{t}: {d:+.1f} [{d - q * e:+.1f}, {d + q * e:+.1f}]"
                                                for (s, t), d, e in zip(labels, flat, se)))
    print(f"verdict: {verdict}")

    print("\nReported without a rule")
    h_base = res[(0, 0, 1.0)].get("prefix_entropy")
    for seed in SEEDS:
        ts = (1.0,) + HIGH_T
        hs = [res[(seed, 150, t)].get("prefix_entropy") for t in ts]
        if h_base is not None and all(h is not None for h in hs) and hs[0] <= h_base <= max(hs):
            t_star = float(np.interp(h_base, hs, ts))
            print(f"   seed {seed}: entropy-matched temperature T* = {t_star:.2f} (base prefix entropy {h_base:.3f}; step 150 at T=1 {hs[0]:.3f})")
        else:
            print(f"   seed {seed}: T* outside 1.0-1.6 (base {h_base}, step 150 {['%.3f' % h for h in hs if h is not None]})")
    bx = sorted((metrics(res[(0, 0, t)]["counts"], ones) for t in BASE_T))
    bpx, bpy = np.array([b[0] for b in bx]), np.array([b[1] for b in bx])
    for seed in SEEDS:
        for t in (1.0,) + HIGH_T:
            p1, p16 = metrics(res[(seed, 150, t)]["counts"], ones)
            if bpx.min() <= p1 <= bpx.max():
                print(f"   seed {seed} step 150 T {t}: pass@16 minus the base model's temperature curve at pass@1 {p1:.1f}: "
                      f"{p16 - float(np.interp(p1, bpx, bpy)):+.1f}")
    for seed in SEEDS:
        px_pts = [(metrics(res[(seed, s, 1.0)]["counts"], ones)[0], 100 * res[(seed, s, 1.0)]["boxed_rate"]) for s in (50, 100, 150)]
        for t in qualifying[seed]:
            p1, _ = metrics(res[(seed, 150, t)]["counts"], ones)
            bx_t = 100 * res[(seed, 150, t)]["boxed_rate"]
            near = min(px_pts, key=lambda z: abs(z[0] - p1))
            if abs(bx_t - near[1]) > 10:
                print(f"   FLAG seed {seed} T {t}: boxed rate {bx_t:.0f}% vs {near[1]:.0f}% on the trajectory near pass@1 {p1:.1f}")


if __name__ == "__main__":
    main()
