"""Stage A rerun with wrong answers carrying larger gradients than correct ones (F0 = ratio * F1)."""

import numpy as np

import catalog
import optimal
import sensitivity

R, RATIO = 1e4, 4.0


def main():
    for shape_name, shape in sensitivity.SHAPES.items():
        optimal.model = lambda p, R, shape=shape: (shape(p), np.full_like(p, R * RATIO), np.full_like(p, R))
        print(f"=== |grad p|^2 ~ {shape_name}, F1 = {R:g}, F0 = {RATIO:g} x F1")
        print(f"{'rule':38s}" + "".join(f"{k:>13s}" for k in catalog.PRIORS))
        rows = {r.name: [] for r in catalog.RULES}
        for a, b in catalog.PRIORS.values():
            p, w = optimal.grid(a, b, 400)
            opt, _ = optimal.optimal_efficiency(p, w, R, catalog.N_MAX)
            for r in catalog.RULES:
                rows[r.name].append(optimal.rule_efficiency(r, p, w, R)[0] / opt)
        for name, vals in rows.items():
            print(f"{name:38s}" + "".join(f"{v:13.3f}" for v in vals))
        print()


if __name__ == "__main__":
    main()
