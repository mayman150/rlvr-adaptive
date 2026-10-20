"""Figures for the ICML draft: OLMo curves (Pre-registration 9), the noise-only test (10), and the
temperature diagnostic (11). Writes paper/figures/{olmo_curves,noise_test,temperature}.pdf."""

import json
import re
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE.parent.parent / "paper" / "figures"
N, K = 32, 16
COL = {"DR": "#1f77b4", "PSR": "#2ca02c", "NSR": "#d62728", "PK4": "#9467bd", "PSRN": "#ff7f0e", "NSR2": "#8c564b"}
NAME = {"DR": "Dr. GRPO", "PSR": "PSR", "NSR": "NSR", "PK4": "pass@4 objective", "PSRN": "PSR + matched noise", "NSR2": "NSR, 2x lr"}


def met(c):
    c = np.asarray(c)
    return 100 * np.mean(c / N), 100 * np.mean([1 - comb(N - int(x), K) / comb(N, K) for x in c])


def olmo_points(arm, seed):
    pts = {}
    for name in (arm, "DR"):
        for job in sorted(j for sfx in ("", "a", "b") for j in RES.glob(f"worth-v9e-{name}{sfx}-s{seed}-*")):
            for f in (job / "eval").glob("step*_eval.json"):
                step = int(re.findall(r"\d+", f.name)[0])
                d = json.loads(f.read_text())
                if "counts" in d and (name == arm or step == 0):
                    pts[step] = met(d["counts"])
    return sorted(pts.items())


def fig_olmo():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, seed in zip(axes, (1, 2)):
        for arm in ("DR", "PSR", "NSR", "PK4"):
            pts = olmo_points(arm, seed)
            x, y = [p[1][0] for p in pts], [p[1][1] for p in pts]
            ax.plot(x, y, "-o", ms=3, color=COL[arm], label=NAME[arm])
        ax.set_title(f"OLMo-2-1B, seed {seed}")
        ax.set_xlabel("pass@1 (%)")
    axes[0].set_ylabel("pass@16 (%)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "olmo_curves.pdf")


def v10(arm, seed):
    name = {"PSR": "psr", "NSR": "nsr", "PSRN": "psrnoise", "NSR2": "nsr"}[arm]
    job = sorted(RES.glob(f"worth-v10-ALL-s{seed}-*"))[-1]
    train = json.loads((job / arm / f"{name}_train.json").read_text())
    passk = json.loads((job / arm / f"{name}_passk.json").read_text())
    return train, {p["step"]: met(p["counts"]) for p in passk}


def fig_noise():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for arm in ("PSR", "NSR", "PSRN", "NSR2"):
        curves, pts = [], {}
        for seed in (5, 6, 7):
            tr, pk = v10(arm, seed)
            fb = np.array([x["first_block_reward"] for x in tr])
            curves.append(np.convolve(fb, np.ones(10) / 10, mode="valid"))
            for s, v in pk.items():
                pts.setdefault(s, []).append(v)
        m = min(len(c) for c in curves)
        axes[0].plot(np.arange(m) + 9, 100 * np.mean([c[:m] for c in curves], axis=0), color=COL[arm], label=NAME[arm])
        base = v10("PSR", 5)[1].get(0)
        steps = sorted(pts)
        xs = ([base[0]] if base else []) + [np.mean([p[0] for p in pts[s]]) for s in steps]
        ys = ([base[1]] if base else []) + [np.mean([p[1] for p in pts[s]]) for s in steps]
        axes[1].plot(xs, ys, "-o", ms=3, color=COL[arm], label=NAME[arm])
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("training accuracy (%, 10-step mean)")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_xlabel("pass@1 (%)")
    axes[1].set_ylabel("pass@16 (%)")
    axes[1].set_title("128 problems, mean of 3 seeds")
    fig.tight_layout()
    fig.savefig(OUT / "noise_test.pdf")


def fig_temperature():
    import analyze_v11 as A
    res = A.load()
    ones = np.ones(256)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for seed, mk in zip((5, 6, 7), ("o", "s", "^")):
        traj = [(0, 0, 1.0)] + [(seed, s, 1.0) for s in (50, 100, 150)]
        xy = [A.metrics(res[k]["counts"], ones) for k in traj]
        ax.plot([p[0] for p in xy], [p[1] for p in xy], "-", marker=mk, ms=4, color="#1f77b4",
                label="training trajectory (T = 1)" if seed == 5 else None)
        temps = [(seed, 150, t) for t in (1.0, 1.07, 1.15, 1.3)]
        xy = [A.metrics(res[k]["counts"], ones) for k in temps]
        ax.plot([p[0] for p in xy], [p[1] for p in xy], "--", marker=mk, ms=4, color="#d62728",
                label="step 150 at T = 1 to 1.3" if seed == 5 else None)
    xy = [A.metrics(res[(0, 0, t)]["counts"], ones) for t in (0.5, 0.7, 0.85, 1.0)]
    ax.plot([p[0] for p in xy], [p[1] for p in xy], ":", marker="x", color="gray", label="base model at T = 0.5 to 1")
    ax.set_xlabel("pass@1 (%)")
    ax.set_ylabel("pass@16 (%)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "temperature.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(HERE))
    fig_olmo()
    fig_noise()
    fig_temperature()
    print("wrote", [p.name for p in OUT.glob("*.pdf")])
