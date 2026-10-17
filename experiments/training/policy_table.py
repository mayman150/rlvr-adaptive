"""Optimal reward-adapted collection policy as a lookup table (numpy only).

Same criterion and solver as theory/optimal.py, restricted to decisions at
block boundaries (every `block` answers, cap `n_max`), with the measured model:
|grad p|^2 proportional to p^2 (1-p)^2, F1 = R, F0 = ratio * R. The difficulty
distribution is a Beta(a, b) prior fitted to recent groups by beta-binomial
maximum likelihood (adaptive stopping does not change the likelihood in p).
"""

import math

import numpy as np


def grid(a, b, n=400):
    p = (np.arange(n) + 0.5) / n
    w = p ** (a - 1) * (1 - p) ** (b - 1)
    return p, w / w.sum()


def _solve(p, w, R, ratio, n_max, block, theta):
    G2 = (p * (1 - p)) ** 2
    F1, F0 = R, R * ratio
    kap1, kap0 = 1 / p, -1 / (1 - p)
    V, policy = {}, {}
    for n in range(n_max, -1, -1):
        for n1 in range(n + 1):
            n0 = n - n1
            mass = w * p ** n1 * (1 - p) ** n0
            u = np.array([np.sum(mass * G2 * n1 * kap1), np.sum(mass * G2 * n0 * kap0)])
            q11 = np.sum(mass * (F1 * n1 + G2 * (n1 * kap1) ** 2 - G2 * n1 * kap1 ** 2))
            q00 = np.sum(mass * (F0 * n0 + G2 * (n0 * kap0) ** 2 - G2 * n0 * kap0 ** 2))
            q10 = np.sum(mass * G2 * (n1 * kap1) * (n0 * kap0))
            Q = np.array([[q11, q10], [q10, q00]])
            idx = [i for i, k in enumerate((n1, n0)) if k > 0]
            adv = np.zeros(2)
            stop_value = 0.0
            if idx:
                sol = np.linalg.solve(Q[np.ix_(idx, idx)], u[idx])
                adv[idx] = sol
                stop_value = 0.5 * float(u[idx] @ sol)
            entry = ("stop", float(adv[0]), float(adv[1]))
            if n == n_max:
                V[(n0, n1)], policy[(n0, n1)] = stop_value, entry
                continue
            cont = -theta * mass.sum() + V[(n0, n1 + 1)] + V[(n0 + 1, n1)]
            forced = n == 0 or n % block != 0
            if forced or cont > stop_value:
                V[(n0, n1)], policy[(n0, n1)] = cont, ("continue", 0.0, 0.0)
            else:
                V[(n0, n1)], policy[(n0, n1)] = stop_value, entry
    return V[(0, 0)], policy


def solve_table(a, b, ratio, R=130.0, n_max=16, block=4, grid_n=400):
    """Optimal blocked policy for a Beta(a, b) difficulty prior; returns (efficiency, table)."""
    p, w = grid(a, b, grid_n)
    lo, hi = 0.0, 1.0
    while _solve(p, w, R, ratio, n_max, block, hi)[0] > 0:
        hi *= 2
    for _ in range(50):
        mid = (lo + hi) / 2
        if _solve(p, w, R, ratio, n_max, block, mid)[0] > 0:
            lo = mid
        else:
            hi = mid
    _, table = _solve(p, w, R, ratio, n_max, block, lo)
    return 2 * lo, table


def fit_beta(groups):
    """Beta-binomial maximum likelihood over a log grid; groups are (successes, answers)."""
    best, best_ll = (1.0, 1.0), -math.inf
    for a in np.exp(np.linspace(math.log(0.1), math.log(20), 40)):
        for b in np.exp(np.linspace(math.log(0.1), math.log(20), 40)):
            ll = sum(math.lgamma(k + a) + math.lgamma(n - k + b) - math.lgamma(n + a + b)
                     - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b) for k, n in groups)
            if ll > best_ll:
                best, best_ll = (float(a), float(b)), ll
    return best
