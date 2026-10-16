"""Shape of |grad p|^2 against difficulty from high-answer-count measurements.

Per prompt (MLP block):
  U    the unbiased pair statistic mean_{i<j} (r_i - c)(r_j - c) Gamma_ij, with c the
       share correct among the other answers (independent of the pair);
  SE   delete-one-answer jackknife standard error of U;
  MoM  median over 8 disjoint answer groups of the within-group statistic
       (robust to a few answers with very large gradients).
Prompts with all answers wrong or all right use c = 1/2, which is also unbiased.
Pooled: weighted least-squares fit of U to each candidate shape, and the ratio of
mean signal on extreme prompts (p <= 0.15 or p >= 0.85) to middle prompts
(0.3 <= p <= 0.7). Predicted ratios: about 0.13 for p^2 (1-p)^2, 0.36 for p (1-p),
and 1 for a flat shape.
"""

import json
import sys
from pathlib import Path

import numpy as np

SHAPES = {"p^2 (1-p)^2": lambda p: (p * (1 - p)) ** 2, "p (1-p)": lambda p: p * (1 - p),
          "constant": lambda p: np.ones_like(p)}


def u_stat(r, G):
    r = np.asarray(r, float)
    n = len(r)
    iu = np.triu_indices(n, 1)
    S = r.sum()
    if S == 0 or S == n:
        c = 0.5
    else:
        c = (S - r[iu[0]] - r[iu[1]]) / (n - 2)
    return float(np.mean((r[iu[0]] - c) * (r[iu[1]] - c) * G[iu]))


def jackknife_se(r, G):
    n = len(r)
    vals = np.array([u_stat(np.delete(r, k), np.delete(np.delete(G, k, 0), k, 1)) for k in range(n)])
    return float(np.sqrt((n - 1) / n * np.sum((vals - vals.mean()) ** 2)))


def median_of_means(r, G, groups=8, seed=0):
    idx = np.random.default_rng(seed).permutation(len(r))
    parts = np.array_split(idx, groups)
    return float(np.median([u_stat(np.asarray(r)[g], G[np.ix_(g, g)]) for g in parts if len(g) > 2]))


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    rows = []
    for rec in data["records"]:
        r = np.asarray(rec["rewards"], float)
        G = np.array(rec["gram_mlp"])
        rows.append({"p": float(r.mean()), "U": u_stat(r, G), "SE": jackknife_se(r, G),
                     "MoM": median_of_means(r, G), "F": float(np.mean(np.diag(G)))})
    rows.sort(key=lambda x: x["p"])
    print(f"{len(rows)} prompts, {data['answers']} answers each (MLP block)\n")
    print(f"{'p':>6s} {'U':>10s} {'SE':>9s} {'U/SE':>6s} {'MoM':>10s}")
    for x in rows:
        print(f"{x['p']:6.3f} {x['U']:10.3f} {x['SE']:9.3f} {(x['U'] / x['SE'] if x['SE'] else float('nan')):6.2f} {x['MoM']:10.3f}")

    p = np.array([x["p"] for x in rows])
    U = np.array([x["U"] for x in rows])
    wts = 1 / np.maximum(np.array([x["SE"] for x in rows]), 1e-12) ** 2
    print("\nWeighted fit to each shape (chi-square per prompt; lower is better):")
    for name, f in SHAPES.items():
        s = f(np.clip(p, 0.01, 0.99))
        k = float(np.sum(wts * s * U) / np.sum(wts * s * s))
        chi2 = float(np.sum(wts * (U - k * s) ** 2) / len(U))
        print(f"   {name:12s} scale {k:10.4g}   chi-square {chi2:.3f}")
    print(f"   zero signal   chi-square {float(np.sum(wts * U ** 2) / len(U)):.3f}")

    ext = (p <= 0.15) | (p >= 0.85)
    mid = (p >= 0.3) & (p <= 0.7)
    if ext.any() and mid.any():
        def wmean(m):
            return float(np.sum(wts[m] * U[m]) / np.sum(wts[m])), float(1 / np.sqrt(np.sum(wts[m])))
        e, se_e = wmean(ext)
        m, se_m = wmean(mid)
        print(f"\nExtreme prompts (n={ext.sum()}): {e:.3f} +/- {se_e:.3f};  middle prompts (n={mid.sum()}): {m:.3f} +/- {se_m:.3f}")
        if m > 2 * se_m:
            print(f"Ratio extreme / middle = {e / m:.2f} (predicted 0.13 for p^2(1-p)^2, 0.36 for p(1-p), 1 for flat)")
        else:
            print("Middle-prompt signal is not clearly above zero; the ratio is not identified.")


if __name__ == "__main__":
    main()
