"""Descriptive 2,048-token check, three seeds: H25 and R25 RL checkpoints 25-150 and starts against B1k_s at 2,048
tokens (base: BASE15 at 2,048, 32 samples; B1k8 from check2k, B1k9/10 from ref2k). analyze_p36's estimator."""
import sys

import numpy as np

sys.argv = sys.argv[:1]
import analyze_p36 as A

rng = np.random.default_rng(36)
A.Model.W = np.stack([np.bincount(rng.integers(0, A.P, A.P), minlength=A.P) for _ in range(2000)]).astype(float)
C, R = A.CHECK2K, A.REF2K
b2 = A.one(f"{C}/BASE15_2k/eval/step0_T1.00_eval.json", need=True)
ref = {8: A.traj(b2, A.ckpts(C, "B1k8_2k", (25, 50, 100, 150))), 9: A.traj(b2, A.ckpts(R, "B1k9_2k", (25, 50, 100, 150))),
       10: A.traj(b2, A.ckpts(R, "B1k10_2k", (25, 50, 100, 150)))}
for X in ("H25", "R25"):
    pts = {8: A.ckpts(C, f"S2{X}s8_2k", (25, 50, 100, 150)), 9: A.ckpts(R, f"S2{X}s9_2k", (25, 50, 100, 150)),
           10: A.ckpts(R, f"S2{X}s10_2k", (25, 50, 100, 150))}
    t = A.contrast(f"{X} checkpoints 25-150 vs B1k_s at 2,048 (descriptive)", pts, ref, A.SEEDS)
    print(A.show(t))
    A.per_checkpoint(t)
    st = A.one(f"{C}/{X}_2k/eval/step0_T1.00_eval.json", need=True)
    r = []
    for s in A.SEEDS:
        rx = sorted((m.full[1], k, m.full[16]) for k, m in ref[s])
        xs, ys = np.array([x[0] for x in rx]), np.array([x[2] for x in rx])
        r.append(f"{100 * (st.full[16] - np.interp(st.full[1], xs, ys)):+.2f}" if xs[0] <= st.full[1] <= xs[-1] else "n/a")
    print(f"   {X} start at 2,048: pass@1 {100 * st.full[1]:.1f}, truncated {100 * st.meta['truncated_rate']:.0f}%; vs B1k8/9/10 {', '.join(r)}")
