"""Figures for the paper from all results available so far (finished and running jobs).

Usage: python plot_curves.py   -> paper/figures/{pilot,shape,ladder,regime}.{pdf,png}
Run fetch_live.sh and fetch_logs.py first to refresh running jobs.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = HERE / "results"
OUT = ROOT / "paper" / "figures"
sys.path.insert(0, str(ROOT / "experiments" / "stage_b"))
import analyze_shape  # noqa: E402

PILOTS = {1: "worth-train-s1-20261016-213109", 2: "worth-train-s2-20261016-230523"}
PROBE = {1: "worth-probe-s1-20261017-041756"}
COLORS = {"baseline": "#1f4e79", "scaled": "#c55a11", "probe": "#548235", "optimal": "#7030a0"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def smooth(y, k=10):
    y = np.asarray(y, float)
    out = np.convolve(y, np.ones(k) / k, mode="valid")
    return np.arange(k - 1, len(y)), out


def load_arm(job, arm):
    folder = RES / job / "training"
    if (folder / f"{arm}_train.json").exists():
        train = json.loads((folder / f"{arm}_train.json").read_text())
        ev = json.loads((folder / f"{arm}_eval.json").read_text())
        return train, ev
    logs = RES / job / "from_logs.json"
    if logs.exists():
        c = json.loads(logs.read_text()).get(arm)
        if c:
            return c["train"], c["eval"]
    return None, None


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def pilot():
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
    for col, seed in enumerate((1, 2)):
        arms = [(PILOTS[seed], "baseline", "Dr. GRPO"), (PILOTS[seed], "scaled", "failures scaled by w")]
        if seed in PROBE:
            arms.append((PROBE[seed], "probe", "probe 4, fill to 16 (running)"))
        for job, arm, label in arms:
            train, ev = load_arm(job, arm)
            if not train:
                continue
            ax[0, col].plot([e["step"] for e in ev], [100 * e["accuracy"] for e in ev], "o-", ms=3,
                            color=COLORS[arm], label=label)
            x, y = smooth([r["mean_reward"] for r in train])
            ax[1, col].plot(x, y, ":" if arm == "probe" else "-", color=COLORS[arm], lw=1.2, label=label)
        ax[0, col].set_title(f"Seed {seed}")
        ax[0, col].set_ylabel("Greedy accuracy (%)")
        ax[1, col].set_ylabel("Training reward")
        ax[1, col].set_xlabel("Step")
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Pilot: scaled failure advantages against Dr. GRPO (128 answers per step)", fontsize=10)
    fig.tight_layout()
    save(fig, "pilot")


def shape():
    d = json.loads((ROOT / "experiments/stage_b/results_shape/summary_all.json").read_text())
    p, u, se = [], [], []
    for rec in d["records"]:
        r = np.asarray(rec["rewards"], float)
        G = np.asarray(rec["gram_mlp"], float)
        p.append(r.mean())
        u.append(analyze_shape.u_stat(r, G))
        se.append(analyze_shape.jackknife_se(r, G))
    p, u, se = map(np.asarray, (p, u, se))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.errorbar(p, u, yerr=se, fmt="o", ms=3.5, color="#1f4e79", ecolor="#9db3cc", elinewidth=1, capsize=0,
                label="per prompt (jackknife s.e.)")
    grid = np.linspace(0.001, 0.999, 300)
    styles = {"p^2 (1-p)^2": ("-", "#c00000"), "p (1-p)": ("--", "#7f7f7f"), "constant": (":", "#7f7f7f")}
    wts = 1 / np.maximum(se, 1e-12) ** 2
    for name, f in analyze_shape.SHAPES.items():
        fx = f(p)
        alpha = np.sum(wts * fx * u) / np.sum(wts * fx * fx)
        ls, c = styles[name]
        ax.plot(grid, alpha * f(grid), ls, color=c, lw=1.3, label=f"fit: {name.replace('^2', '²')}")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Success probability p (share of 96 answers correct)")
    ax.set_ylabel("|∇p|² estimate, last-layer MLP")
    ax.set_title("Gradient signal against difficulty", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "shape")


def ladder():
    runs = {}
    for job in sorted(RES.glob("worth-v2-*")):
        m = re.match(r"worth-v2-(\w+?)-s(\d+)-", job.name)
        folder = job / m.group(1)
        files = list(folder.glob("*_train.json")) if folder.exists() else []
        if not files:
            continue
        arm = files[0].name[: -len("_train.json")]
        train = json.loads(files[0].read_text())
        ev_path = folder / f"{arm}_eval.json"
        ev = json.loads(ev_path.read_text()) if ev_path.exists() else []
        key = (m.group(1), int(m.group(2)))
        if key not in runs or len(train) > len(runs[key][0]):
            runs[key] = (train, ev)
    if not runs:
        return
    colors = {"L32": "#bdd7ee", "L64": "#6fa8dc", "L128": "#1f4e79", "L256": "#0b2545",
              "O64": "#c9a0dc", "O128": "#7030a0", "S128": "#c55a11"}
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
    for col, seed in enumerate((1, 2)):
        for name in ("L32", "L64", "L128", "L256", "O64", "O128", "S128"):
            if (name, seed) not in runs:
                continue
            train, ev = runs[(name, seed)]
            ls = "--" if name.startswith("O") else "-"
            if ev:
                ax[0, col].plot([e["step"] for e in ev], [100 * e["accuracy"] for e in ev], "o" + ls, ms=3,
                                color=colors[name], label=f"{name} ({len(train)} steps)")
            if len(train) >= 10:
                x, y = smooth([r["mean_reward"] for r in train])
                ax[1, col].plot(x, y, ls, color=colors[name], lw=1.1)
        ax[0, col].set_title(f"Seed {seed} (runs in progress)")
        ax[0, col].set_ylabel("Greedy accuracy (%)")
        ax[1, col].set_ylabel("Training reward")
        ax[1, col].set_xlabel("Step")
        ax[0, col].legend(frameon=False, fontsize=7)
    fig.suptitle("Budget ladder (L = Dr. GRPO, O = optimal rule, S = scaled failures; number = answers per step)", fontsize=9)
    fig.tight_layout()
    save(fig, "ladder")


def regime():
    B = np.logspace(0.8, 3.6, 200)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for k, ls in ((2.0, "-"), (1.3, "--")):
        Bstar = 128.0
        base = 1 / (1 + Bstar / B)
        better = 1 / (1 + Bstar / (k * B))
        ax.plot(B, better / base, ls, color="#7030a0", label=f"rule with {k:g}x efficiency, SGD")
        ax.plot(B, np.sqrt(better / base), ls, color="#c55a11", lw=1, label=f"same, Adam at fixed lr (approx.)")
    ax.axvline(128, color="black", lw=0.6)
    ax.text(135, 1.55, "B*", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Answers per step B")
    ax.set_ylabel("Progress per step relative to Dr. GRPO")
    ax.set_title("Theorem 2: gains vanish above the critical budget", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "regime")


if __name__ == "__main__":
    pilot()
    shape()
    ladder()
    regime()
    print("figures in", OUT)
