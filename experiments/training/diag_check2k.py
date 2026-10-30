"""Pre-registration 30's registered 2,048-token sensitivity check (descriptive, one seed): seed-8 checkpoints of R25
and H25 (steps 25, 50, 100, 150) and the starts (H25, R25, OPD25) against B1k8's trajectory at the same budget
(base and B1k8 steps 25-150, all re-evaluated at 2,048 tokens), next to the same quantities at 1,024 tokens.
Gap = pass@16 minus the reference at the checkpoint's pass@1 (pp); 90% intervals from 1,000 problem-bootstrap draws.
Usage: python diag_check2k.py
"""
import numpy as np

import analyze_p30 as A

C = "results/worth-p30-check2k-s0-20261030-061010"
full = np.arange(A.P)
rng = np.random.default_rng(2048)
draws = [rng.integers(0, A.P, A.P) for _ in range(1000)]
ref2 = [(0, A.one(f"{C}/BASE15_2k/eval/step0_T1.00_eval.json"))] + \
       [(st, A.one(f"{C}/B1k8_2k/eval/step{st}_T1.00_eval.json")) for st in (25, 50, 100, 150)]
ref1 = A.b1k(8)[0]


def gap(p, ref, idx):
    r = A.interp_ref(ref)(A.pk(p[0], p[1], 1, idx), idx)
    return None if r is None else 100 * (A.pk(p[0], p[1], 16, idx) - r)


def show(label, p, ref):
    g = gap(p, ref, full)
    if g is None:
        return f"{label}: outside the reference's support (pass@1 {100 * A.pk(p[0], p[1], 1, full):.1f})"
    b = [gap(p, ref, i) for i in draws]
    b = np.array([x for x in b if x is not None])
    lo, hi = np.percentile(b, [5, 95])
    return (f"{label}: pass@1 {100 * A.pk(p[0], p[1], 1, full):.1f}, pass@16 {100 * A.pk(p[0], p[1], 16, full):.1f}, "
            f"truncated {100 * p[2].get('truncated_rate', np.nan):.1f}%; gap {g:+.2f} [{lo:+.2f}, {hi:+.2f}]")


print("B1k8 reference at 2,048 tokens: " + ", ".join(f"step {st}: {100 * A.pk(p[0], p[1], 1, full):.1f}/{100 * A.pk(p[0], p[1], 16, full):.1f} "
      f"(trunc {100 * p[2].get('truncated_rate', np.nan):.1f}%)" for st, p in ref2))
print("B1k8 reference at 1,024 tokens: " + ", ".join(f"step {st}: {100 * A.pk(p[0], p[1], 1, full):.1f}/{100 * A.pk(p[0], p[1], 16, full):.1f} "
      f"(trunc {100 * p[2].get('truncated_rate', np.nan):.1f}%)" for st, p in ref1))
for name, d1, d2 in (("H25 start", f"{A.S1B}/H025_test", f"{C}/H25_2k"), ("R25 start", f"{A.S1B}/R025_test", f"{C}/R25_2k"),
                     ("OPD25 start", f"{A.S2}/OPD25_test", f"{C}/OPD25_2k")):
    print(f"\n{name}")
    print("   1,024: " + show("", A.one(f"{d1}/eval/step0_T1.00_eval.json"), ref1))
    print("   2,048: " + show("", A.one(f"{d2}/eval/step0_T1.00_eval.json"), ref2))
for run in ("S2H25s8", "S2R25s8"):
    print(f"\n{run}")
    for st in (25, 50, 100, 150):
        print(f"   step {st} 1,024: " + show("", A.one(f"{A.S2}/{run}/eval/step{st}_T1.00_eval.json"), ref1))
        print(f"   step {st} 2,048: " + show("", A.one(f"{C}/{run}_2k/eval/step{st}_T1.00_eval.json"), ref2))
