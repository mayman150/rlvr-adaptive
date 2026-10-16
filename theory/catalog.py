"""Rules in use against the optimal rule, across difficulty priors and noise regimes."""

import lattice
import optimal

PRIORS = {"uniform": (1, 1), "mostly hard": (0.5, 2), "U-shaped": (0.3, 0.3), "mostly easy": (2, 0.5)}
RULES = [
    lattice.fixed(4), lattice.fixed(8), lattice.fixed(16),
    lattice.fixed(8, lattice.std_normalized, "fixed 8, std-normalized"),
    lattice.probe_then_fill(2, 8), lattice.probe_then_fill(4, 16),
    lattice.until_mixed(2, 16),
    lattice.until_k_correct(1, 16), lattice.until_k_correct(2, 16),
    lattice.until_balanced(2, 2, 16),
]
N_MAX = 32


def main():
    p, w = optimal.grid(1, 1, 400)
    eff, policy = optimal.optimal_efficiency(p, w, 100.0, N_MAX)
    rule = optimal.policy_rule(policy, N_MAX)
    check, _ = optimal.rule_efficiency(rule, p, w, 100.0)
    assert abs(check - eff) / eff < 1e-6, (check, eff)
    print(f"consistency: dynamic-programming optimum {eff:.6g} equals lattice evaluation {check:.6g}\n")

    for R in (1.0, 100.0, 1e4):
        print(f"=== per-answer Fisher R = {R:g} (efficiency relative to the optimal rule)")
        header = f"{'rule':38s}" + "".join(f"{k:>13s}" for k in PRIORS)
        print(header)
        rows = {r.name: [] for r in RULES}
        best_fixed, opt_cost = [], []
        for a, b in PRIORS.values():
            p, w = optimal.grid(a, b, 400)
            opt, pol = optimal.optimal_efficiency(p, w, R, N_MAX)
            opt_cost.append(optimal.expected_cost(pol, p, w))
            fixed_opt, _ = optimal.optimal_efficiency(p, w, R, 8, force_until=8)
            best_fixed.append(fixed_opt / opt)
            for r in RULES:
                e, _ = optimal.rule_efficiency(r, p, w, R)
                rows[r.name].append(e / opt)
        for name, vals in rows.items():
            print(f"{name:38s}" + "".join(f"{v:13.3f}" for v in vals))
        print(f"{'fixed 8 with optimal advantages':38s}" + "".join(f"{v:13.3f}" for v in best_fixed))
        print(f"{'optimal rule: answers per prompt':38s}" + "".join(f"{v:13.2f}" for v in opt_cost))
        print()


if __name__ == "__main__":
    main()
