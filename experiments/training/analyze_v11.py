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
HIGH_T = (1.07, 1.15, 1.3, 1.45, 1.6)
N_PROBLEMS = 256


def load():
    """{(seed, step, T): result} for PSR checkpoints and the base model (seed 0, step 0); later jobs win
    on overlap. Only jobs of the PSR arm (PSRa, PSRb) and BASE are read; every result is validated."""
    out = {}
    jobs = sorted((j for j in RES.glob("worth-v11-*") if not j.name.startswith("worth-v11smoke")), key=lambda j: j.name[-15:])
    for job in jobs:
        m = re.match(r"worth-v11-(BASE[ab]?|PSR[abc])-s(\d+)-", job.name)
        if not m:
            continue
        seed = int(m.group(2))
        for f in (job / "eval").glob("step*_T*_eval.json"):
            step, t = re.match(r"step(\d+)_T([\d.]+)_eval", f.name).groups()
            d = json.loads(f.read_text())
            num = lambda k: isinstance(d.get(k), (int, float)) and math.isfinite(d[k])
            ok = ("counts" in d and d.get("n") == N and len(d["counts"]) == N_PROBLEMS and d.get("problems") == N_PROBLEMS
                  and all(isinstance(c, int) and 0 <= c <= N for c in d["counts"])
                  and num("temperature") and abs(d["temperature"] - float(t)) < 1e-6
                  and num("prefix_entropy") and all(num(k) for k in ("boxed_rate", "truncated_rate", "mean_length_tokens")))
            if not ok:
                print(f"   invalid result skipped (counts as missing, so the verdict is incomplete evaluation): {job.name}/{f.name}")
                continue
            out[(seed, int(step), round(float(t), 2))] = d
    return out


def metrics(counts, w):
    c = np.asarray(counts)
    p1 = c / N
    pk = np.array([1.0 - comb(N - int(x), K) / comb(N, K) for x in c])
    return 100 * float(w @ p1) / w.sum(), 100 * float(w @ pk) / w.sum()


def trajectory(res, seed, w, value=None):
    """pass@1 and pass@16 (or another per-point value) of the T = 1 trajectory: base, steps 50, 100,
    150, ordered by pass@1 with ties broken by step. Points outside its range take the end value
    (np.interp); how often that happens in the bootstrap is reported."""
    keys = [(0, 0, 1.0)] + [(seed, s, 1.0) for s in (50, 100, 150)]
    pts = []
    for key in keys:
        p1, p16 = metrics(res[key]["counts"], w)
        pts.append((p1, p16 if value is None else value(res[key]), key[1]))
    pts.sort(key=lambda x: (x[0], x[2]))
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def diffs(res, w, qualifying, outside=None):
    out = {}
    for seed in SEEDS:
        px, py = trajectory(res, seed, w)
        vals = []
        for t in qualifying[seed]:
            p1, p16 = metrics(res[(seed, 150, t)]["counts"], w)
            if outside is not None:
                outside.append(not (px.min() <= p1 <= px.max()))
            vals.append(p16 - float(np.interp(p1, px, py)))
        out[seed] = vals
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
    print("Points (pass@1, pass@16, prefix entropy, parseable-box rate, mean tokens, truncation):")
    for key in sorted(res):
        d = res[key]
        p1, p16 = metrics(d["counts"], np.ones(len(d["counts"])))
        print(f"   seed {key[0]} step {key[1]:3d} T {key[2]:.2f}: {p1:5.1f} {p16:5.1f}  H {d.get('prefix_entropy', math.nan):.3f}  "
              f"boxed {100 * d.get('boxed_rate', math.nan):.0f}%  tokens {d.get('mean_length_tokens', math.nan):.0f}  "
              f"truncated {100 * d.get('truncated_rate', math.nan):.0f}%")
    if missing:
        print(f"\nverdict: incomplete evaluation; missing {missing}")
        return
    n_prob = N_PROBLEMS
    ones = np.ones(n_prob)
    qualifying = {}
    for seed in SEEDS:
        px, _ = trajectory(res, seed, ones)
        qualifying[seed] = [t for t in HIGH_T if px.min() <= metrics(res[(seed, 150, t)]["counts"], ones)[0] <= px.max()]
    print("\nqualifying temperatures:", qualifying)
    if any(not q for q in qualifying.values()):
        print("verdict: insufficient overlap")
    else:
        full = diffs(res, ones, qualifying)
        per_seed = {s: float(np.mean(v)) for s, v in full.items()}
        est = float(np.mean(list(per_seed.values())))
        boot, boot_pts, outside = [], [], []
        for _ in range(args.draws):
            w = np.bincount(rng.integers(0, n_prob, n_prob), minlength=n_prob).astype(float)
            d = diffs(res, w, qualifying, outside)
            boot.append(np.mean([np.mean(v) for v in d.values()]))
            boot_pts.append(np.concatenate([d[s] for s in SEEDS]))
        lo90, hi90 = np.percentile(boot, [5, 95])
        lo95, hi95 = np.percentile(boot, [2.5, 97.5])
        flat, bp = np.concatenate([full[s] for s in SEEDS]), np.array(boot_pts)
        se = bp.std(axis=0, ddof=1)
        q = np.percentile(np.max(np.abs(bp - flat) / np.where(se > 0, se, np.inf), axis=1), 90)
        band = [(d - q * e, d + q * e) for d, e in zip(flat, se)]
        crossing = any(lo > 2 for lo, _ in band) and any(hi < -2 for _, hi in band)
        if est >= 2 and lo95 > 0:
            verdict = "separable (a late checkpoint at a higher temperature beats the training trajectory on average)"
        elif lo90 >= -2 and hi90 <= 2:
            verdict = "average agreement (no average separation within ±2; not pointwise equivalence)"
        elif est <= -2 and hi95 < 0:
            verdict = "below (on average, raising the temperature costs more coverage than training less)"
        else:
            verdict = "inconclusive"
        if crossing:
            verdict += "; CROSSING: some points are clearly above +2 and others clearly below -2, so the method choice waits"
        print(f"pooled D {est:+.2f}, 90% [{lo90:+.2f}, {hi90:+.2f}], 95% [{lo95:+.2f}, {hi95:+.2f}]; per seed "
              + ", ".join(f"{s}: {v:+.2f}" for s, v in per_seed.items()))
        labels = [(s, t) for s in SEEDS for t in qualifying[s]]
        print("simultaneous 90% band: " + "; ".join(f"s{s} T{t}: {d:+.1f} [{lo:+.1f}, {hi:+.1f}]"
                                                    for (s, t), d, (lo, hi) in zip(labels, flat, band)))
        print(f"bootstrap points outside the resampled trajectory range (end value used): {100 * np.mean(outside):.1f}%")
        print(f"verdict: {verdict}")

    print("\nReported without a rule")
    h_base = res[(0, 0, 1.0)]["prefix_entropy"]
    ts = (1.0,) + HIGH_T
    for seed in SEEDS:
        hs = [res[(seed, 150, t)]["prefix_entropy"] for t in ts]
        if hs[0] <= h_base <= max(hs):
            t_star = float(np.interp(h_base, hs, ts))
            p1s = [metrics(res[(seed, 150, t)]["counts"], ones)[0] for t in ts]
            p16s = [metrics(res[(seed, 150, t)]["counts"], ones)[1] for t in ts]
            p1, p16 = float(np.interp(t_star, ts, p1s)), float(np.interp(t_star, ts, p16s))
            px, py = trajectory(res, seed, ones)
            inside = px.min() <= p1 <= px.max()
            print(f"   seed {seed}: entropy-matched temperature T* = {t_star:.2f} (base prefix entropy {h_base:.3f}); at T*: pass@1 {p1:.1f}, "
                  f"pass@16 {p16:.1f}, D(T*) " + (f"{p16 - float(np.interp(p1, px, py)):+.1f}" if inside else "outside the trajectory range"))
        else:
            print(f"   seed {seed}: T* outside 1.0-1.6 (base {h_base:.3f}; step 150: " + ", ".join(f"{h:.3f}" for h in hs) + ")")
    bx = sorted((metrics(res[(0, 0, t)]["counts"], ones) for t in BASE_T))
    bpx, bpy = np.array([b[0] for b in bx]), np.array([b[1] for b in bx])
    for seed in SEEDS:
        c1 = np.array([metrics(res[(seed, 150, t)]["counts"], ones)[0] for t in ts])
        c16 = np.array([metrics(res[(seed, 150, t)]["counts"], ones)[1] for t in ts])
        lo, hi = max(bpx.min(), c1.min()), min(bpx.max(), c1.max())
        if lo <= hi:
            grid = np.linspace(lo, hi, 5)
            o = np.argsort(c1)
            gap = [float(np.interp(g, c1[o], c16[o]) - np.interp(g, bpx, bpy)) for g in grid]
            print(f"   seed {seed}: step 150's temperature curve minus the base model's, pass@1 {lo:.1f}-{hi:.1f}: "
                  + ", ".join(f"{x:+.1f}" for x in gap))
        else:
            print(f"   seed {seed}: no pass@1 overlap between the base model's and step 150's temperature curves")
    for seed in SEEDS:
        _, tb = trajectory(res, seed, ones, value=lambda d: 100 * d["boxed_rate"])
        tx, _ = trajectory(res, seed, ones)
        _, tt = trajectory(res, seed, ones, value=lambda d: 100 * d["truncated_rate"])
        for t in qualifying.get(seed, []):
            d = res[(seed, 150, t)]
            p1, _ = metrics(d["counts"], ones)
            for name, curve_v, val in (("boxed", tb, 100 * d["boxed_rate"]), ("truncated", tt, 100 * d["truncated_rate"])):
                ref = float(np.interp(p1, tx, curve_v))
                if abs(val - ref) > 10:
                    print(f"   FLAG seed {seed} T {t}: {name} rate {val:.0f}% vs {ref:.0f}% on the trajectory at pass@1 {p1:.1f}")


if __name__ == "__main__":
    main()
