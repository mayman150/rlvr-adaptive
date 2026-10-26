"""Pre-registration 20, rule 3: the frontier model's frozen forecast for 7B, computed from the base 7B counts
(32 samples, 2,048 tokens, the 511 problems) before any trained 7B checkpoint is evaluated.

Frozen: law.fit_prior / law.initial_theta on the base counts, tau = -0.20 (Pre-registration 17), comparators
uniform log-odds shift (beta 1) and affine log-odds map (beta 2.10, Pre-registration 17's fit). For each pass@1 on a
grid, the predicted pass@16 overall and by base-count stratum. Rule 3 scores the law at B7's observed pass@1.
Usage: python forecast_20.py  (writes FORECAST_20.json)
"""

import json
from pathlib import Path

import numpy as np

import law

HERE = Path(__file__).resolve().parent
BASE = HERE / "results" / "worth-v20-BASE7s-s0-20261025-193202" / "BASE7" / "eval" / "step0_T1.00_eval.json"
TAU, BETA, N = -0.20, 2.10, 32


def strata(c):
    return {"c=0": c == 0, "c=1-4": (c >= 1) & (c <= 4), "c=5-16": (c >= 5) & (c <= 16), "c>16": c > 16}


def main():
    c = np.array(json.loads(BASE.read_text())["counts"])
    prior = law.fit_prior(c, N)
    th0 = law.initial_theta(c, N, prior)
    traj = law.trajectory(th0, TAU)
    grid = [round(x, 3) for x in np.arange(0.30, 0.901, 0.01)]
    out = {"base": str(BASE.relative_to(HERE)), "prior": prior, "tau": TAU, "beta": BETA, "grid": []}
    for x in grid:
        row = {"pass@1": x}
        for name, p in (("law", law.at_pass1(traj, x)), ("uniform", law.uniform_shift(th0, x)), ("affine", law.uniform_shift(th0, x, BETA))):
            row[name] = {"pass@16": law.pass_k_true(p, 16), **{k: law.pass_k_true(p[m], 16) for k, m in strata(c).items() if m.any()}}
        out["grid"].append(row)
    (HERE / "FORECAST_20.json").write_text(json.dumps(out, indent=1))
    for row in out["grid"][::5]:
        print(f"pass@1 {row['pass@1']:.2f}: law {100*row['law']['pass@16']:.1f}, uniform {100*row['uniform']['pass@16']:.1f}, affine {100*row['affine']['pass@16']:.1f}")


if __name__ == "__main__":
    main()
