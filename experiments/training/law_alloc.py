"""Pre-registration 18 forecasts: the frontier law under two-stage rollout allocation at a fixed rollout budget.

For a Bernoulli log-odds parameter, the expected Dr. GRPO step for a problem whose pooled group has C successes
of N is E[C (N - C) / N] (for a fixed N this is (N - 1) p (1 - p)). Under two-stage allocation N depends on the
stage-1 count c1 (reused in the group) and on the other problems in the batch, so the growth multiplier
relative to B (N = 8) is m(p) = E[w C (N - C) / N] / (7 p (1 - p)), with w = 1 for Dr. GRPO's fixed normalizer
and w = 8 / N for HORA's per-problem normalization. E[C (N - C) | c1, N] is exact (C = c1 + Binomial(N - 4, p));
(c1, N) are simulated with the batch competition on the current state; m is smoothed over problems of similar p
with a normalized window. d theta_x / dt = m(p_x) p_x (1 - p_x) + tau * mean m(p) p (1 - p).
"""

import math

import numpy as np

import law

N0, TOTAL, CAP, G = 4, 128, 32, 8
lb = lambda x, y: math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)


def law_value(c, n=N0, k=16, lam=0.37):
    a, b = c + 0.5, n - c + 0.5
    return max(k * math.exp(lb(a + 2, b + k + 1) - lb(a, b)) - lam * math.exp(lb(a + 2, b + 2) - lb(a, b)), 0.0)


VALUES = {"law": [law_value(c) for c in range(N0 + 1)],
          "variance": [(c + 0.5) * (N0 - c + 0.5) / ((N0 + 1.0) * (N0 + 2.0)) for c in range(N0 + 1)]}


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


def hora_allocation(counts, extra, cap_each):
    a = [1.0 + c for c in counts]; b = [1.0 + N0 - c for c in counts]
    m = [ai / (ai + bi) for ai, bi in zip(a, b)]; out = [0] * len(counts)
    for _ in range(extra):
        i = max((j for j in range(len(counts)) if out[j] < cap_each), key=lambda j: m[j], default=None)
        if i is None:
            break
        out[i] += 1
        m[i] *= (b[i] + out[i] - 1) / (a[i] + b[i] + out[i])
    return out


def smooth(values, p, width=25):
    order = np.argsort(p)
    v = values[order]
    num = np.convolve(v, np.ones(width), mode="same"); den = np.convolve(np.ones_like(v), np.ones(width), mode="same")
    out = np.empty_like(values); out[order] = num / den
    return out


def multiplier(p, rng, rule, batches=3000, P=16):
    num, cnt = np.zeros(len(p)), np.zeros(len(p))
    for _ in range(batches):
        idx = rng.choice(len(p), P, replace=False)
        c1 = rng.binomial(N0, p[idx])
        if rule == "hora":
            extra = np.array(hora_allocation(list(c1), TOTAL - P * N0, CAP - N0))
        else:
            extra = np.array(allocate_extra([VALUES[rule][x] for x in c1], TOTAL - P * N0, CAP - N0))
        N = N0 + extra
        q = p[idx]
        EC = c1 + extra * q
        EC2 = extra * q * (1 - q) + EC ** 2
        step = (N * EC - EC2) / N  # E[C (N - C) / N | c1, N]
        if rule == "hora":
            step = step * G / N
        num[idx] += step; cnt[idx] += 1
    mean_step = np.where(cnt > 0, num / np.maximum(cnt, 1), 7 * p * (1 - p))
    return smooth(mean_step, p) / np.maximum(smooth(7 * p * (1 - p), p), 1e-12)


def trajectory_alloc(theta0, tau, rule, refresh=200, seed=0):
    rng = np.random.default_rng(seed)
    th, out, means = theta0.copy(), [], []
    m = multiplier(law.sigmoid(th), rng, rule)
    for i in range(int(law.T_MAX / law.DT)):
        p = law.sigmoid(th)
        if i % law.EVERY == 0:
            out.append(p.copy()); means.append(p.mean())
            if p.mean() > 0.9:
                break
        if i % refresh == 0 and i:
            m = multiplier(p, rng, rule)
        s = m * p * (1 - p)
        th = th + law.DT * (s + tau * s.mean())
    return out, np.array(means)
