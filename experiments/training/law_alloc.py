"""Pre-registration 18 forecast: the frontier law under two-stage rollout allocation.

Each problem's expected step scales with (n_x - 1) under Dr. GRPO with group size n_x, so relative to
B (n = 8) its growth rate is multiplied by m_x = (n_x - 1) / 7. The allocation rule's expected m(p) is
estimated by Monte Carlo on the current state (batches of 16 problems drawn from the evaluation problems,
stage-1 counts ~ Binomial(4, p), extras by allocation_value), refreshed during integration.
d theta_x / dt = m(p_x) p_x (1 - p_x) + tau * mean m(p) p (1 - p).
"""

import math

import numpy as np

import law


def allocation_value(c, n, k=16):
    a, b = c + 0.5, n - c + 0.5
    lb = lambda x, y: math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)
    return k * math.exp(lb(a + 1, b + k) - lb(a, b))


def allocate_extra(values, extra, cap_each):
    m = len(values); out = [0] * m; left, active = extra, list(range(m))
    while left > 0 and active:
        tot = sum(values[i] for i in active)
        share = {i: (values[i] / tot if tot > 0 else 1.0 / len(active)) * left for i in active}
        base = {i: min(int(share[i]), cap_each - out[i]) for i in active}
        for i in active:
            out[i] += base[i]
        left -= sum(base.values())
        for i in sorted(active, key=lambda i: -(share[i] - int(share[i]))):
            if left == 0:
                break
            if out[i] < cap_each:
                out[i] += 1; left -= 1
        active = [i for i in active if out[i] < cap_each]
    return out


VALUES = {"coverage": [allocation_value(c, 4) for c in range(5)],
          "variance": [(c + 0.5) * (4 - c + 0.5) / (5.0 * 6.0) for c in range(5)]}


def multiplier(p, rng, batches=1500, P=16, n0=4, total=128, cap=32, rule="coverage"):
    """Expected (n_x - 1) / 7 for each problem, given current success probabilities p."""
    acc, cnt = np.zeros(len(p)), np.zeros(len(p))
    for _ in range(batches):
        idx = rng.choice(len(p), P, replace=False)
        c = rng.binomial(n0, p[idx])
        extra = allocate_extra([VALUES[rule][x] for x in c], total - P * n0, cap - n0)
        acc[idx] += (n0 + np.array(extra) - 1) / 7.0
        cnt[idx] += 1
    m = np.where(cnt > 0, acc / np.maximum(cnt, 1), 1.0)
    # smooth over problems with similar p (expected multiplier depends on p and the batch distribution)
    order = np.argsort(p); sm = np.convolve(m[order], np.ones(25) / 25, mode="same")
    out = np.empty_like(m); out[order] = sm
    return out


def trajectory_alloc(theta0, tau, refresh=200, seed=0, rule="coverage"):
    rng = np.random.default_rng(seed)
    th, out, means = theta0.copy(), [], []
    m = multiplier(law.sigmoid(th), rng, rule=rule)
    for i in range(int(law.T_MAX / law.DT)):
        p = law.sigmoid(th)
        if i % law.EVERY == 0:
            out.append(p.copy()); means.append(p.mean())
            if p.mean() > 0.9:
                break
        if i % refresh == 0 and i:
            m = multiplier(p, rng, rule=rule)
        s = m * p * (1 - p)
        th = th + law.DT * (s + tau * s.mean())
    return out, np.array(means)
