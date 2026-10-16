"""A practical approximation of the optimal rule, and how close it gets.

Band rule (two parameters plus the failure weight):
  sample at least n_min answers; continue while the minority outcome is at
  least a fraction theta of the answers so far and n < n_max; then stop.
Advantages on stopping, with a posterior-mean baseline from a Beta(a0, b0)
prior on difficulty and failures weighted by w:
  success  a = 1 - p_hat,   failure  b = -w * p_hat,   p_hat = (k + a0) / (n + a0 + b0).
"""

import math

import numpy as np

import catalog
import lattice
import optimal
import sensitivity


def band_rule(theta, n_min, n_max, w=1.0, prior=(1.0, 1.0), baseline="posterior"):
    def stop(n0, n1):
        n = n0 + n1
        if n < n_min:
            return False
        return min(n0, n1) < math.ceil(theta * n) or n >= n_max

    def adv(n0, n1):
        n = n0 + n1
        if baseline == "posterior":
            ph = (n1 + prior[0]) / (n + prior[0] + prior[1])
        else:
            ph = n1 / n
        return (1 - ph) / n, -w * ph / n

    return lattice.Rule(f"band theta={theta} n_min={n_min} cap={n_max} w={w} {baseline}", stop, adv, n_max)


def evaluate(shape_name, ratio, R=1e4):
    shape = sensitivity.SHAPES[shape_name]
    optimal.model = lambda p, R, shape=shape: (shape(p), np.full_like(p, R * ratio), np.full_like(p, R))
    w = 1 / ratio
    candidates = [
        lattice.fixed(8),
        catalog.RULES[5],  # probe 4, fill to 16
        lattice.fixed(8, lambda n0, n1: (lambda a, b: (a, b * w))(*lattice.mean_centered(n0, n1)), "fixed 8, failures scaled by w"),
        band_rule(0.25, 3, 16, 1.0, baseline="mean"),
        band_rule(0.25, 3, 16, 1.0),
        band_rule(0.25, 3, 16, w),
        band_rule(0.25, 3, 32, w),
        band_rule(0.2, 3, 16, w),
        band_rule(0.3, 3, 16, w),
    ]
    rows = {c.name: [] for c in candidates}
    for a, b in catalog.PRIORS.values():
        p, wt = optimal.grid(a, b, 400)
        opt, _ = optimal.optimal_efficiency(p, wt, R, catalog.N_MAX)
        for c in candidates:
            rows[c.name].append(optimal.rule_efficiency(c, p, wt, R)[0] / opt)
    print(f"=== |grad p|^2 ~ {shape_name}, F0 = {ratio:g} x F1 (efficiency relative to optimal)")
    print(f"{'rule':52s}" + "".join(f"{k:>13s}" for k in catalog.PRIORS))
    for name, vals in rows.items():
        print(f"{name:52s}" + "".join(f"{v:13.3f}" for v in vals))
    print()


if __name__ == "__main__":
    for shape_name in ("p^2 (1-p)^2", "p (1-p)", "constant"):
        evaluate(shape_name, 1 / 0.395)
