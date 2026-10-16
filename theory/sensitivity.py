"""Does the ranking of rules survive other models of gradient signal versus difficulty?"""

import numpy as np

import catalog
import optimal

SHAPES = {
    "p^2 (1-p)^2": lambda p: (p * (1 - p)) ** 2,
    "p (1-p)": lambda p: p * (1 - p),
    "constant": lambda p: np.ones_like(p),
}
R = 1e4


def main():
    for shape_name, shape in SHAPES.items():
        optimal.model = lambda p, R, shape=shape: (shape(p), np.full_like(p, R), np.full_like(p, R))
        print(f"=== |grad p|^2 proportional to {shape_name}, R = {R:g}")
        print(f"{'rule':38s}" + "".join(f"{k:>13s}" for k in catalog.PRIORS))
        rows = {r.name: [] for r in catalog.RULES}
        costs = []
        for a, b in catalog.PRIORS.values():
            p, w = optimal.grid(a, b, 400)
            opt, pol = optimal.optimal_efficiency(p, w, R, catalog.N_MAX)
            costs.append(optimal.expected_cost(pol, p, w))
            for r in catalog.RULES:
                rows[r.name].append(optimal.rule_efficiency(r, p, w, R)[0] / opt)
        for name, vals in rows.items():
            print(f"{name:38s}" + "".join(f"{v:13.3f}" for v in vals))
        print(f"{'optimal rule: answers per prompt':38s}" + "".join(f"{v:13.2f}" for v in costs))
        print()


if __name__ == "__main__":
    main()
