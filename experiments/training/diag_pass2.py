"""Diagnostic (not registered): pass@2 and pass@16 at matched pass@1 against B1k's trajectory, for the 1.5B/1,024-token
interventions. At exactly matched mean success, the pass@2 change equals minus the change in Var_x(p_x)."""
import sys
sys.path.insert(0, ".")
import numpy as np
import analyze_v12 as A
import analyze_v21 as V
import matched as M

res = V.load()
SEEDS = (8, 9, 10)
n = len(V.counts(res, "BASE15", 0))
arms = [("PSR1k", "PSR1k", (25, 50, 100, 150)), ("NSR1k", "NSR1k", (50, 100, 150)), ("AC1k", "AC1k", (50, 100, 150)),
        ("HG1k", "HG1k", (50, 100, 150)), ("HGm1k", "HGm1k", (50, 100, 150)), ("PG1k", "PG1k", (25, 50, 100, 150)),
        ("FL10", "FL10s", (25, 50, 100, 150)), ("FL5", "FL5s", (25, 50, 100, 150))]
print(f"{'arm':7s} {'matched dpass@2':>26s} {'implied dVar(p)':>16s} {'matched dpass@16':>26s}")
for lab, name, steps in arms:
    row = []
    for k in (2, 16):
        A.K = k
        B = M.reference(V.traj(res, "B1k", (25, 50, 100, 150), ""))
        t = M.compare(V.arm_points(res, name, steps), B, SEEDS, n, int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
        row.append(t)
    A.K = 16
    t2, t16 = row
    if "estimate" not in t2:
        print(lab, t2.get("verdict")); continue
    print(f"{lab:7s} {t2['estimate']:+6.2f} [{t2['ci90'][0]:+.2f}, {t2['ci90'][1]:+.2f}]   {-t2['estimate'] / 100:+.4f}         "
          f"{t16['estimate']:+6.2f} [{t16['ci90'][0]:+.2f}, {t16['ci90'][1]:+.2f}]")
# scale: variance of p across problems along B1k (seed 8), from unbiased pairwise estimates c(c-1)/(n(n-1))
c0 = V.counts(res, "BASE15", 0)
for st in (0, 25, 50, 100, 150):
    c = c0 if st == 0 else V.counts(res, "B1k8", st)
    m = c.mean() / 32
    ep2 = np.mean(c * (c - 1) / (32 * 31))
    print(f"B1k8 step {st:3d}: pass@1 {100 * m:5.1f}, Var_x(p) {ep2 - m * m:.4f}")
