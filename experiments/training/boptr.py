"""Restricted BOPTR baseline (Sun et al. 2026, arXiv 2609.01274) against Dr. GRPO's curve, on existing evaluations.

BOPTR approximates an RL checkpoint's pass@k by the base model's pass@N with N = alpha k^beta. Restricted here to one
decoder (T = 1 sampling, 32 samples per problem), so N ranges over [1, 32]; the full rule also selects search policies.

Predicting pass@16 from pass@1 for held-out checkpoints, with both predictors fitted on Dr. GRPO's checkpoints only:
- curve: Dr. GRPO's pass@16-vs-pass@1 points (all seeds, steps, temperatures; base included), linearly interpolated;
- BOPTR: beta fitted on Dr. GRPO's checkpoints (least squares over k = 2, 4, 8, 16); for each checkpoint alpha solves
  base pass@alpha = its pass@1; prediction base pass@(alpha 16^beta). Unscorable if alpha < 1 or alpha 16^beta > 32.
pass@N of the base for non-integer N: linear interpolation in log N between the unbiased integer estimates.
Usage: python boptr.py [--setting 1k]
"""

import argparse
import glob
import json
import re
from math import comb

import numpy as np

N = 32


def passk(c, k):
    return float(np.mean([1 - comb(N - int(x), k) / comb(N, k) for x in c]))


def load(pattern):
    out = {}
    for f in glob.glob(pattern):
        name = f.split("/")[-3]
        step, t = re.search(r"step(\d+)_T([\d.]+)_eval", f).groups()
        out[(name, int(step), round(float(t), 2))] = np.array(json.loads(open(f).read())["counts"])
    return out


def base_curve(c0):
    ks = np.arange(1, N + 1)
    vals = np.array([passk(c0, k) for k in ks])
    lk = np.log(ks)

    def at(n):
        return float(np.interp(np.log(n), lk, vals)) if 1 <= n <= N else None

    def inv(p):  # smallest N with base pass@N = p (the base curve is increasing)
        return float(np.exp(np.interp(p, vals, lk))) if vals[0] <= p <= vals[-1] else None
    return at, inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="1k", choices=("1k",))
    a = ap.parse_args()
    res = load("results/worth-v15v-all-*/*/eval/step*_T*_eval.json")
    c0 = res[("BASE15", 0, 1.0)]
    at, inv = base_curve(c0)
    fit = {k: v for k, v in res.items() if re.fullmatch(r"B1k(8|9|10)", k[0])}
    held = {k: v for k, v in res.items() if not re.fullmatch(r"B1k(8|9|10)|BASE15(val)?", k[0]) and not k[0].endswith("val")}
    pts = sorted([(passk(c0, 1), passk(c0, 16))] + [(passk(c, 1), passk(c, 16)) for c in fit.values()])
    px, py = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])

    def curve(p1):
        return float(np.interp(p1, px, py)) if px.min() <= p1 <= px.max() else None

    def loss(beta):
        err = []
        for c in fit.values():
            al = inv(passk(c, 1))
            if al is None or al < 1:
                continue
            for k in (2, 4, 8, 16):
                v = at(al * k ** beta)
                if v is not None:
                    err.append(v - passk(c, k))
        return np.mean(np.square(err)) if err else np.inf, len(err)
    betas = np.linspace(0.05, 2.0, 391)
    losses = [loss(b)[0] for b in betas]
    beta = float(betas[int(np.argmin(losses))])
    print(f"setting {a.setting}: base pass@1 {100 * passk(c0, 1):.2f}, pass@16 {100 * passk(c0, 16):.2f}, pass@32 {100 * passk(c0, 32):.2f}")
    print(f"BOPTR beta fitted on {len(fit)} Dr. GRPO evaluations: {beta:.3f} (fit RMSE {100 * np.sqrt(min(losses)):.2f} pp over k = 2-16)")
    rows = []
    for (name, step, t), c in sorted(held.items()):
        p1, p16 = passk(c, 1), passk(c, 16)
        al = inv(p1)
        b = at(al * 16 ** beta) if al is not None and al >= 1 else None
        cv = curve(p1)
        rows.append((name, step, t, p1, p16, cv, b))
    both = [r for r in rows if r[5] is not None and r[6] is not None]
    e_c = np.array([r[4] - r[5] for r in both])
    e_b = np.array([r[4] - r[6] for r in both])
    print(f"held-out evaluations: {len(rows)}; scorable by both predictors: {len(both)} "
          f"(curve only: {sum(r[5] is not None and r[6] is None for r in rows)}, BOPTR only: {sum(r[5] is None and r[6] is not None for r in rows)})")
    print(f"pass@16 prediction error on the common set (pp): curve RMSE {100 * np.sqrt(np.mean(e_c ** 2)):.2f}, mean {100 * e_c.mean():+.2f}; "
          f"BOPTR RMSE {100 * np.sqrt(np.mean(e_b ** 2)):.2f}, mean {100 * e_b.mean():+.2f}")
    by = {}
    for r, ec, eb in zip(both, e_c, e_b):
        by.setdefault(re.sub(r"\d+$", "", r[0]), []).append((ec, eb))
    for arm, v in sorted(by.items()):
        v = np.array(v)
        print(f"   {arm}: n {len(v)}, curve RMSE {100 * np.sqrt(np.mean(v[:, 0] ** 2)):.2f}, BOPTR RMSE {100 * np.sqrt(np.mean(v[:, 1] ** 2)):.2f}")


if __name__ == "__main__":
    main()
