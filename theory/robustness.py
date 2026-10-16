"""Is the optimal lookup-table rule robust to the unmeasured shape of |grad p|^2?

Solve the optimal policy assuming one shape, evaluate it with the exact lattice
engine under every true shape, and report efficiency relative to the true
optimum. F0 / F1 is the measured full-parameter ratio (1 / 0.395).
"""

import numpy as np

import catalog
import lattice
import optimal
import sensitivity

RATIO, R = 1 / 0.395, 1e4


def set_shape(shape):
    optimal.model = lambda p, R, shape=shape: (shape(p), np.full_like(p, R * RATIO), np.full_like(p, R))


def main():
    shapes = sensitivity.SHAPES
    for prior_name in ("uniform", "mostly hard"):
        a, b = catalog.PRIORS[prior_name]
        p, w = optimal.grid(a, b, 400)
        print(f"=== prior {prior_name}: rows assumed shape, columns true shape")
        print(f"{'policy solved under':22s}" + "".join(f"{s:>14s}" for s in shapes) + f"{'answers/prompt':>16s}")
        truth_opt = {}
        for t_name, t_shape in shapes.items():
            set_shape(t_shape)
            truth_opt[t_name], _ = optimal.optimal_efficiency(p, w, R, catalog.N_MAX)
        for a_name, a_shape in shapes.items():
            set_shape(a_shape)
            _, policy = optimal.optimal_efficiency(p, w, R, catalog.N_MAX)
            cost = optimal.expected_cost(policy, p, w)
            rule = optimal.policy_rule(policy, catalog.N_MAX, f"solved under {a_name}")
            row = []
            for t_name, t_shape in shapes.items():
                set_shape(t_shape)
                row.append(optimal.rule_efficiency(rule, p, w, R)[0] / truth_opt[t_name])
            print(f"{a_name:22s}" + "".join(f"{v:14.3f}" for v in row) + f"{cost:16.2f}")
        best_in_use = []
        for t_name, t_shape in shapes.items():
            set_shape(t_shape)
            best_in_use.append(max(optimal.rule_efficiency(r, p, w, R)[0] for r in catalog.RULES) / truth_opt[t_name])
        print(f"{'best rule in use':22s}" + "".join(f"{v:14.3f}" for v in best_in_use))
        print()


if __name__ == "__main__":
    main()
