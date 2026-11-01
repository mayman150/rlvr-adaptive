"""Figure: matched-pass@1 pass@16 gap by RL step for four warm starts, against the 700-step Dr. GRPO replicates
(B1kL seeds 8 and 10; mean over the two seeds; descriptive). Palette: reference categorical slots 1-4 (dataviz)."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.argv = sys.argv[:1]
import analyze_p36 as A

A.Model.W = np.zeros((1, A.P))
base = A.one("results/worth-v15v-all-*/BASE15/eval/step0_T1.00_eval.json", need=True)
L = {s: A.traj(base, A.ckpts(f"results/worth-p30-longref{s}-*", f"B1kL{s}", (25, 50, 100, 150, 200, 300, 400, 500, 600, 700))) for s in (8, 9, 10)}
ARMS = [("TEACHF", "Teacher solutions", "#2a78d6", lambda s: A.ckpts(A.P36, f"P35bTEACHFs{s}")),
        ("CORR", "Teacher, corrupted final answer", "#eb6834", lambda s: A.ckpts(A.P36, f"P35bCORRs{s}")),
        ("SELF", "Base's own solutions", "#1baf7a", lambda s: A.ckpts(A.P36, f"P35bSELFs{s}")),
        ("CWS", "Base's own, coverage-weighted", "#eda100", lambda s: A.ckpts(A.P36, f"P36CWS{A.SFT_SEED[s]}s{s}"))]
fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=200)
for key, label, col, get in ARMS:
    g = {st: [] for st in A.STEPS}
    for s in (8, 9, 10):
        pts = get(s)
        xs = sorted((m.full[1], t, m.full[16]) for t, m in L[s])
        x, y = np.array([v[0] for v in xs]), np.array([v[2] for v in xs])
        for st in A.STEPS:
            u = pts[st].full[1]
            if x[0] <= u <= x[-1]:
                g[st].append(100 * (pts[st].full[16] - np.interp(u, x, y)))
    st = [k for k in A.STEPS if len(g[k]) == 3]
    ax.plot(st, [np.mean(g[k]) for k in st], color=col, lw=1.6, marker="o", ms=3.2, label=label, mec="white", mew=0.6)
ax.axhline(0, color="#9a9994", lw=0.8)
ax.set_xscale("log")
ax.set_xticks(list(A.STEPS))
ax.set_xticklabels([str(k) for k in A.STEPS], fontsize=6.5)
ax.tick_params(axis="y", labelsize=6.5)
ax.set_xlabel("RL step (Dr. GRPO after the warm start)", fontsize=7)
ax.set_ylabel("pass@16 gap at matched pass@1 (points)", fontsize=7)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="#e6e5e0", lw=0.5)
ax.legend(fontsize=5.8, frameon=False, loc="lower left")
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
fig.tight_layout()
fig.savefig("../../paper/icml/figures/start_trajectories.pdf"); fig.savefig("/tmp/p30/start_trajectories.png")
print("saved")
