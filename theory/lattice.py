"""Exact moments and cost of reward-adapted rollout collection rules.

A rule samples answers one at a time and sees only their binary rewards. With
exchangeable answers it is a map on the lattice of counts (n0 failures, n1
successes): at each state it either continues or stops, and on stopping it
gives every success advantage a and every failure advantage b (a = b = 0 is
abandonment). For a prompt with success probability p, the mean update is
c * grad p and the second moment is F1*D1 + F0*D0 + |grad p|^2 * X, where

    c  = E[n1 a / p - n0 b / (1 - p)]
    D1 = E[n1 a^2],  D0 = E[n0 b^2]
    X  = E[(n1 a / p - n0 b / (1 - p))^2 - n1 a^2 / p^2 - n0 b^2 / (1 - p)^2]

with expectations over the stopped state. Expected cost is E[n0 + n1].
"""

from fractions import Fraction as Fr
from math import comb


class Rule:
    """stop(n0, n1) -> bool; adv(n0, n1) -> (a, b). Must stop by n_max."""
    def __init__(self, name, stop, adv, n_max):
        self.name, self.stop, self.adv, self.n_max = name, stop, adv, n_max


def stopped_states(rule, p):
    """Yield (n0, n1, probability) for every state where the rule stops."""
    reach = {(0, 0): Fr(1) if isinstance(p, Fr) else 1.0}
    for n in range(rule.n_max + 1):
        for n1 in range(n + 1):
            n0 = n - n1
            mass = reach.pop((n0, n1), 0)
            if not mass:
                continue
            if n == rule.n_max or rule.stop(n0, n1):
                yield n0, n1, mass
            else:
                reach[(n0, n1 + 1)] = reach.get((n0, n1 + 1), 0) + mass * p
                reach[(n0 + 1, n1)] = reach.get((n0 + 1, n1), 0) + mass * (1 - p)


def moments(rule, p):
    c = D1 = D0 = X = N = abandon = 0
    for n0, n1, m in stopped_states(rule, p):
        a, b = rule.adv(n0, n1)
        s = n1 * a / p - n0 * b / (1 - p)
        c += m * s
        D1 += m * n1 * a * a
        D0 += m * n0 * b * b
        X += m * (s * s - n1 * a * a / (p * p) - n0 * b * b / ((1 - p) ** 2))
        N += m * (n0 + n1)
        if a == 0 and b == 0:
            abandon += m
    return {"c": c, "D1": D1, "D0": D0, "X": X, "N": N, "abandon": abandon}


def second_moment(m, F0, F1, G2):
    return F1 * m["D1"] + F0 * m["D0"] + G2 * m["X"]


# ---- advantages ------------------------------------------------------------

def mean_centered(n0, n1):
    n = n0 + n1
    if n == 0:
        return 0, 0
    mean = Fr(n1, n)
    return (1 - mean) / n, (0 - mean) / n


def std_normalized(n0, n1):
    n = n0 + n1
    if n0 == 0 or n1 == 0:
        return 0.0, 0.0
    mean = n1 / n
    sd = (mean * (1 - mean)) ** 0.5
    return (1 - mean) / (n * sd), (0 - mean) / (n * sd)


# ---- rules in use ------------------------------------------------------------

def fixed(G, adv=mean_centered, name=None):
    return Rule(name or f"fixed group of {G}", lambda n0, n1: n0 + n1 >= G, adv, G)


def probe_then_fill(k, G):
    """Abandon after k answers if they all agree, else fill to G (SARA-style)."""
    def stop(n0, n1):
        n = n0 + n1
        return (n == k and (n0 == 0 or n1 == 0)) or n >= G
    def adv(n0, n1):
        if n0 + n1 == k and (n0 == 0 or n1 == 0) and k < G:
            return 0, 0
        return mean_centered(n0, n1)
    return Rule(f"probe {k}, abandon if agree, fill to {G}", stop, adv, G)


def until_mixed(n_min, n_max):
    """Sample until both outcomes appear, between n_min and n_max answers."""
    stop = lambda n0, n1: n0 + n1 >= n_min and n0 > 0 and n1 > 0
    return Rule(f"until mixed, {n_min} to {n_max}", stop, mean_centered, n_max)


def until_k_correct(k, n_max):
    """Sample until k correct answers (Never-Give-Up and Reinforce-Ada style)."""
    return Rule(f"until {k} correct, cap {n_max}", lambda n0, n1: n1 >= k, mean_centered, n_max)


def until_balanced(pos, neg, n_max):
    """Sample until pos successes and neg failures, or n_max (Reinforce-Ada-balance style)."""
    return Rule(f"until {pos} correct and {neg} wrong, cap {n_max}",
                lambda n0, n1: n1 >= pos and n0 >= neg, mean_centered, n_max)


if __name__ == "__main__":
    # Cross-check against the brute-force enumeration in exact.py.
    import exact
    checks = 0
    pairs = [
        (fixed(4), exact.Fixed(4)),
        (probe_then_fill(2, 4), exact.ProbeThenFill(2, 4)),
        (until_mixed(2, 6), exact.UntilMixed(2, 6)),
    ]
    prompts = [
        exact.Prompt([Fr(1, 2), Fr(1, 4), Fr(1, 4)], [1, 0, 0]),
        exact.Prompt([Fr(1, 8), Fr(3, 8), Fr(1, 6), Fr(1, 3)], [1, 1, 0, 0]),
        exact.Prompt([Fr(1, 5), Fr(2, 5), Fr(1, 5), Fr(1, 5)], [0, 1, 1, 1]),
    ]
    for lat_rule, brute_rule in pairs:
        for prompt in prompts:
            p = prompt.mean_reward()
            m = moments(lat_rule, p)
            mean, second, _ = exact.moments(prompt, brute_rule)
            assert exact.coefficient(mean, prompt.grad()) == m["c"]
            G2 = exact.dot(prompt.grad(), prompt.grad())
            pred = second_moment(m, prompt.fisher_given(0), prompt.fisher_given(1), G2)
            assert pred == second, (lat_rule.name, pred, second)
            checks += 2
    print(f"lattice engine matches brute-force enumeration: {checks} exact checks")
