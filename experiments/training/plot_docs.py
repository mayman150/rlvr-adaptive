"""Figures for docs/: the frontier law's predictions, pass@16 by base difficulty, and the Qwen frontier."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_v12 as A
import analyze_v14 as V
import analyze_v9 as O
import law

OUT = Path(__file__).resolve().parents[2] / "docs" / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
r = V.load()
base = r[("BASE", 0, 0, 1.0)]
n = 32
prior = law.fit_prior(base, n)
th = law.initial_theta(base, n, prior)
fit = [r[("B", s, st, 1.0)] for s in A.SEEDS for st in A.STEPS]
tau = law.fit_tau(th, fit, n)

# 1. law: predicted vs observed pass@16
fig, ax = plt.subplots(figsize=(3.6, 3.4))
groups = {"Dr. GRPO (fit)": [("B", s, st) for s in A.SEEDS for st in A.STEPS],
          "9 other rules (Qwen)": [(a, s, st) for a in ("CM", "M", "U", "D32", "U32", "C05", "G32", "MXR", "WRF") for s in A.SEEDS for st in A.STEPS]}
tr = law.trajectory(th, tau)
for (label, keys), col in zip(groups.items(), ("#1f77b4", "#2ca02c")):
    xs, ys = [], []
    for k in keys:
        if k + (1.0,) in r:
            c = r[k + (1.0,)]
            xs.append(100 * law.pass_k_true(law.at_pass1(tr, c.mean() / n), 16)); ys.append(100 * law.pass_k_observed(c, n, 16))
    ax.scatter(xs, ys, s=12, color=col, label=label, alpha=0.8)
xs, ys = [], []
for arm in ("DR", "PSR", "NSR", "PK4", "O128"):
    for s in (1, 2):
        pts = O.eval_points(arm, s)
        if 0 not in pts:
            continue
        tho = law.initial_theta(pts[0], n, law.fit_prior(pts[0], n)); tro = law.trajectory(tho, tau)
        for st, c in pts.items():
            if st > 0:
                xs.append(100 * law.pass_k_true(law.at_pass1(tro, c.mean() / n), 16)); ys.append(100 * law.pass_k_observed(c, n, 16))
ax.scatter(xs, ys, s=12, color="#ff7f0e", label="OLMo, 5 rules (transferred)", alpha=0.8)
lo, hi = 40, 92
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("predicted pass@16 (model)"); ax.set_ylabel("observed pass@16")
ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "law_fit.pdf")

# 2. pass@16 by base difficulty at step 150 (Qwen, mean over seeds)
strata = {"never\n(0/32)": base == 0, "rare\n(1-4)": (base >= 1) & (base <= 4), "medium\n(5-16)": (base >= 5) & (base <= 16)}
arms = [("BASE", "base model"), ("B", "Dr. GRPO"), ("CM", "coverage memory"), ("D32", "seeded replay"), ("S32", "warm start")]
fig, ax = plt.subplots(figsize=(4.6, 2.8))
w = 0.16
for i, (a, lab) in enumerate(arms):
    vals = []
    for m in strata.values():
        cs = [r[("BASE", 0, 0, 1.0)]] if a == "BASE" else [r[(a, s, 150, 1.0)] for s in A.SEEDS]
        vals.append(np.mean([100 * law.pass_k_observed(c[m], n, 16) for c in cs]))
    ax.bar(np.arange(3) + (i - 2) * w, vals, w, label=lab)
ax.set_xticks(range(3)); ax.set_xticklabels(strata.keys()); ax.set_ylabel("pass@16 at step 150")
ax.legend(frameon=False, fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(OUT / "strata.pdf")

# 3. the Qwen frontier: pass@16 vs pass@1 (511 problems), seed means
fig, ax = plt.subplots(figsize=(3.8, 3.0))
ones = np.ones(len(base))
for a, lab, col, steps in (("B", "Dr. GRPO", "#1f77b4", A.STEPS), ("CM", "coverage memory", "#2ca02c", A.STEPS), ("D32", "seeded replay", "#9467bd", A.STEPS),
                           ("S32", "warm start", "#d62728", (0,) + A.STEPS), ("G32", "groups of 32", "#8c564b", A.STEPS)):
    pts = []
    for st in steps:
        cs = [r[(a, s, st, 1.0)] for s in A.SEEDS if (a, s, st, 1.0) in r]
        if cs:
            pts.append(np.mean([A.met(c, ones) for c in cs], axis=0))
    if a in ("B", "CM", "D32"):
        pts = [A.met(base, ones)] + pts
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], "o-", ms=3, lw=1, color=col, label=lab)
import glob, json
long = {0: base}
for f in glob.glob(str(A.RES / "worth-v17Le-*/B13/eval/step*_T1.00_eval.json")):
    long[int(f.split("step")[-1].split("_")[0])] = np.array(json.load(open(f))["counts"])
if len(long) > 1:
    pts = np.array([A.met(c, ones) for st, c in sorted(long.items())])
    ax.plot(pts[:, 0], pts[:, 1], "s--", ms=3, lw=1, color="#1f77b4", alpha=0.6, label="Dr. GRPO, long run (seed 13)")
ax.set_xlabel("pass@1 (accuracy)"); ax.set_ylabel("pass@16 (diversity)")
ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(OUT / "frontier_qwen.pdf")
print("figures written to", OUT, "tau", tau)
