"""The frontier law of Pre-registration 17: per-question log-odds learning from base-model counts.

theta_x(0) = log((c_x + a) / (n_x - c_x + b)), (a, b) a Beta prior fitted by moments to the base counts;
d theta_x / dt = p_x (1 - p_x) + tau * mean_x' p_x' (1 - p_x'), p = sigmoid(theta).
A checkpoint is placed at the time t where the predicted mean success equals its observed pass@1; the law
then predicts the rest of the distribution (pass@k for k > 1, difficulty strata, per-question counts).
Alternatives of equal or larger complexity: a uniform log-odds shift (tau-free) and an affine log-odds map
theta' = alpha + beta * theta (alpha matched to pass@1, beta fitted once on the same data as tau).
"""

from math import comb, lgamma

import numpy as np

DT, T_MAX, EVERY = 0.01, 200.0, 5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_prior(counts, n):
    """Beta(a, b) by the method of moments on per-question success rates, removing binomial noise."""
    p = np.asarray(counts, float) / n
    m, v = p.mean(), p.var()
    v_true = max(v - np.mean(p * (1 - p)) / n, 1e-5)
    k = max(m * (1 - m) / v_true - 1, 1e-3)
    return m * k, (1 - m) * k


def initial_theta(counts, n, prior, extra=None):
    """extra: optional (counts, trials) of additional base samples per question (0 where none)."""
    c, t = np.asarray(counts, float), np.full(len(counts), float(n))
    if extra is not None:
        c, t = c + extra[0], t + extra[1]
    a, b = prior
    return np.log((c + a) / (t - c + b))


def trajectory(theta0, tau):
    th, out, means = theta0.copy(), [], []
    for i in range(int(T_MAX / DT)):
        p = sigmoid(th)
        if i % EVERY == 0:
            out.append(p.copy())
            means.append(p.mean())
            if p.mean() > 0.995:
                break
        s = p * (1 - p)
        th = th + DT * (s + tau * s.mean())
    return out, np.array(means)


def at_pass1(traj, target):
    states, means = traj
    return states[int(np.argmin(np.abs(means - target)))]


def uniform_shift(theta0, target, beta=1.0):
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sigmoid(beta * theta0 + mid).mean() < target else (lo, mid)
    return sigmoid(beta * theta0 + (lo + hi) / 2)


def pass_k_true(p, k):
    return float(np.mean(1 - (1 - p) ** k))


def pass_k_observed(counts, n, k):
    return float(np.mean([1 - comb(n - int(c), k) / comb(n, k) for c in counts]))


def loglik(counts, n, p):
    """Binomial log-likelihood of observed per-question counts under predicted success rates."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    c = np.asarray(counts, float)
    const = sum(lgamma(n + 1) - lgamma(x + 1) - lgamma(n - x + 1) for x in c)
    return float(const + np.sum(c * np.log(p) + (n - c) * np.log(1 - p)))


def fit_tau(theta0, runs, n, k=16, grid=np.round(np.arange(-1.0, 2.0001, 0.05), 2)):
    """Least squares on pass@k over the given runs (lists of per-question counts)."""
    best = None
    for tau in grid:
        tr = trajectory(theta0, tau)
        err = sum((pass_k_true(at_pass1(tr, c.mean() / n), k) - pass_k_observed(c, n, k)) ** 2 for c in runs)
        if best is None or err < best[1]:
            best = (float(tau), err)
    return best[0]


def fit_beta(theta0, runs, n, k=16, grid=np.round(np.arange(0.5, 3.0001, 0.05), 2)):
    best = None
    for beta in grid:
        err = sum((pass_k_true(uniform_shift(theta0, c.mean() / n, beta), k) - pass_k_observed(c, n, k)) ** 2 for c in runs)
        if best is None or err < best[1]:
            best = (float(beta), err)
    return best[0]


def evaluate(theta0, tau, beta, runs, n, strata=None, ks=(2, 4, 8, 16, 32)):
    """Per run: observed and predicted pass@k (law, uniform shift, affine), per-stratum pass@16, log-likelihoods."""
    tr = trajectory(theta0, tau)
    rows = []
    for name, c in runs:
        target = c.mean() / n
        law, uni, aff = at_pass1(tr, target), uniform_shift(theta0, target), uniform_shift(theta0, target, beta)
        row = {"run": name, "pass@1": 100 * target}
        for k in ks:
            if k <= n:
                row[f"obs@{k}"] = 100 * pass_k_observed(c, n, k)
                row[f"law@{k}"], row[f"uni@{k}"], row[f"aff@{k}"] = (100 * pass_k_true(q, k) for q in (law, uni, aff))
        for sname, m in (strata or {}).items():
            row[f"{sname}:obs@16"] = 100 * pass_k_observed(c[m], n, 16)
            row[f"{sname}:law@16"] = 100 * pass_k_true(law[m], 16)
        row["ll_law"], row["ll_uni"], row["ll_aff"] = (loglik(c, n, q) for q in (law, uni, aff))
        rows.append(row)
    return rows


def rmse(rows, key_obs="obs@16", key_pred="law@16"):
    return float(np.sqrt(np.mean([(r[key_obs] - r[key_pred]) ** 2 for r in rows])))
