"""The optimal reward-adapted collection rule, by dynamic programming.

Criterion. A batch is built from prompts drawn from a difficulty prior pi(p);
each prompt is processed by the same rule. With gradients of different prompts
orthogonal and the step size chosen optimally, the improvement of the target
J = E[p] per generated answer is proportional to

    Eff = Sig^2 / (Cost * M2),
    Sig  = E_pi[c(p) G2(p)],  M2 = E_pi[second moment],  Cost = E_pi[N].

Scaling all advantages leaves Eff unchanged, so the rule is optimized by
maximizing Sig - M2/2 - theta * Cost over stopping and advantages and choosing
theta with optimum 0 (Dinkelbach); then theta = max Eff / 2. For fixed theta the
objective is additive over lattice states. At each (n0, n1) the value of
stopping is max over (a, b) of u.(a, b) - (a, b) Q (a, b)/2 = u Q^-1 u / 2, with u
and Q integrals over the posterior mass of that state.

Model of the four numbers per prompt (declared, varied in the report):
G2(p) = p^2 (1-p)^2 (a single consequential decision), F0 = F1 = R (constant
per-answer Fisher; R >> 1 is the high-dimensional regime of real models).
"""

import numpy as np
import lattice


def grid(a, b, n=2000):
    p = (np.arange(n) + 0.5) / n
    w = p ** (a - 1) * (1 - p) ** (b - 1)
    return p, w / w.sum()


def model(p, R):
    G2 = (p * (1 - p)) ** 2
    return G2, np.full_like(p, R), np.full_like(p, R)


def rule_efficiency(rule, p, w, R):
    G2, F0, F1 = model(p, R)
    sig = m2 = cost = 0.0
    for pk, wk, g2, f0, f1 in zip(p, w, G2, F0, F1):
        m = lattice.moments(rule, float(pk))
        sig += wk * m["c"] * g2
        m2 += wk * lattice.second_moment(m, f0, f1, g2)
        cost += wk * m["N"]
    return sig * sig / (cost * m2), cost


def solve(p, w, R, n_max=32, theta=0.0, force_until=None):
    """Value of the best rule for a given theta; returns (value, policy)."""
    G2, F0, F1 = model(p, R)
    kap1, kap0 = 1 / p, -1 / (1 - p)
    V, policy = {}, {}
    for n in range(n_max, -1, -1):
        for n1 in range(n + 1):
            n0 = n - n1
            mass = w * p ** n1 * (1 - p) ** n0
            # u.(a,b): signal; Q: second moment, from the lattice formulas.
            q11 = np.sum(mass * (F1 * n1 + G2 * (n1 * kap1) ** 2 - G2 * n1 * kap1 ** 2))
            q00 = np.sum(mass * (F0 * n0 + G2 * (n0 * kap0) ** 2 - G2 * n0 * kap0 ** 2))
            q10 = np.sum(mass * G2 * (n1 * kap1) * (n0 * kap0))
            # Signal s = n1 a kap1 + n0 b kap0, since -n0 b / (1-p) = n0 b kap0.
            u = np.array([np.sum(mass * G2 * n1 * kap1), np.sum(mass * G2 * n0 * kap0)])
            Q = np.array([[q11, q10], [q10, q00]])
            idx = [i for i, k in enumerate((n1, n0)) if k > 0]
            adv = np.zeros(2)
            if idx:
                Qs, us = Q[np.ix_(idx, idx)], u[idx]
                sol = np.linalg.solve(Qs, us)
                adv[idx] = sol
                stop_value = 0.5 * float(us @ sol)
            else:
                stop_value = 0.0
            stop_entry = ("stop", float(adv[0]), float(adv[1]))
            if n == n_max:
                V[(n0, n1)], policy[(n0, n1)] = stop_value, stop_entry
                continue
            cont = -theta * mass.sum() + V[(n0, n1 + 1)] + V[(n0 + 1, n1)]
            must_continue = force_until is not None and n < force_until
            if must_continue or cont > stop_value:
                V[(n0, n1)], policy[(n0, n1)] = cont, ("continue", 0.0, 0.0)
            else:
                V[(n0, n1)], policy[(n0, n1)] = stop_value, stop_entry
    return V[(0, 0)], policy


def optimal_efficiency(p, w, R, n_max=32, force_until=None):
    """Largest Eff = 2 theta with value(theta) = 0, by bisection."""
    lo, hi = 0.0, 1.0
    while solve(p, w, R, n_max, hi, force_until)[0] > 0:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if solve(p, w, R, n_max, mid, force_until)[0] > 0:
            lo = mid
        else:
            hi = mid
    value, policy = solve(p, w, R, n_max, lo, force_until)
    return 2 * lo, policy


def expected_cost(policy, p, w):
    cost, reach = 0.0, {(0, 0): w.copy()}
    for n in range(0, 200):
        nxt = {}
        for (n0, n1), mass in reach.items():
            if policy.get((n0, n1), ("stop",))[0] == "stop":
                cost += (n0 + n1) * mass.sum()
            else:
                nxt[(n0, n1 + 1)] = nxt.get((n0, n1 + 1), 0) + mass * p
                nxt[(n0 + 1, n1)] = nxt.get((n0 + 1, n1), 0) + mass * (1 - p)
        if not nxt:
            break
        reach = nxt
    return cost


def policy_rule(policy, n_max, name="optimal"):
    """A lattice Rule that follows a solved policy."""
    stop = lambda n0, n1: policy[(n0, n1)][0] == "stop"
    adv = lambda n0, n1: policy[(n0, n1)][1:]
    return lattice.Rule(name, stop, adv, n_max)
