"""Figures for the ICML draft v5 (paper/icml/figures/). Static PDFs; palette: the dataviz reference instance's first
three categorical slots (validated all-pairs, light mode), with marker shapes and direct labels as secondary encoding.
Usage: python plot_paper_v5.py
"""

import glob
import gzip
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import law

OUT = Path(__file__).resolve().parents[2] / "paper" / "icml" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "serif", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4})


def ev(f):
    d = json.load(open(f))
    c = np.array(d["counts"])
    return 100 * c.mean() / d["n"], 100 * np.mean([1 - comb(d["n"] - int(x), 16) / comb(d["n"], 16) for x in c]), d


def law_curve(c0, n0):
    tr = law.trajectory(law.initial_theta(c0, n0, law.fit_prior(c0, n0)), -0.20)
    states, means = tr
    return 100 * np.array(means), np.array([100 * law.pass_k_true(s, 16) for s in states])


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("pass@1 (%)")
    ax.set_ylabel("pass@16 (%)")


def fig_curve():
    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    base = glob.glob("results/worth-v15v-all-*/BASE15/eval/step0_T1.00_eval.json")[0]
    b0 = ev(base)
    c0 = np.array(json.load(open(base))["counts"])
    lx, ly = law_curve(c0, 32)
    m = lx <= 72
    ax.plot(lx[m], ly[m], "--", color=MUTED, linewidth=1.0, label="frozen law (from base counts)")
    for i, s in enumerate((8, 9, 10)):
        pts = [b0] + [ev(f) for f in glob.glob(f"results/worth-v15v-all-*/B1k{s}/eval/step*_T1.00_eval.json")]
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=BLUE, linewidth=1.2, alpha=0.9,
                label="Dr. GRPO (3 seeds)" if i == 0 else None)
    others = []
    for arm in ("PSR1k", "NSR1k", "AC1k", "HG1k", "HGm1k", "PG1k"):
        others += [ev(f) for f in glob.glob(f"results/worth-v15v-all-*/{arm}[0-9]*/eval/step*_T1.00_eval.json")]
    ax.scatter([p[0] for p in others], [p[1] for p in others], s=12, marker="o", color=ORANGE, edgecolor="white",
               linewidth=0.5, zorder=3, label="six other interventions (3 seeds)")
    ax.scatter([b0[0]], [b0[1]], s=16, marker="s", color=INK, zorder=4)
    ax.annotate("base", (b0[0], b0[1]), xytext=(4, -6), textcoords="offset points", color=MUTED, fontsize=6.5)
    style(ax)
    ax.set_xlim(18, 62)
    ax.set_ylim(76, 89)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "curve_1k.pdf")
    plt.close(fig)


def label_of(name):
    if "random" in name:
        return "random reward"
    if any(k in name for k in ("LUFFY-Qwen-Math-7B-Zero_", "LUFFY-Qwen-Math-15B-Zero", "ReLIFT", "ExGRPO-LUFFY")):
        return "external solutions or SFT"
    return "correctness-only RL"


def fig_public():
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.9))
    marks = {"correctness-only RL": (BLUE, "o"), "external solutions or SFT": (ORANGE, "^"), "random reward": (AQUA, "s")}
    # 7B at 2,048 tokens: the frozen forecast (FORECAST_20.json) and B7's evaluations
    ax = axes[0]
    fc = json.load(open("FORECAST_20.json"))
    ax.plot([100 * r["pass@1"] for r in fc["grid"]], [100 * r["law"]["pass@16"] for r in fc["grid"]], "--", color=MUTED,
            linewidth=1.0, label="frozen law (from base counts)")
    import analyze_p28 as A
    for i, (traj, temps) in enumerate(A.seeds_7()):
        pts = sorted((100 * c.mean() / n, 100 * A.pk16(c, n)) for c, n in traj)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=INK, linewidth=0.9, alpha=0.55,
                label="our Dr. GRPO 7B (2 seeds)" if i == 0 else None)
    groups = {}
    for f in sorted(glob.glob("results/worth-p28*-public-*/PUB_*_2k/eval/step0_T1.00_eval.json")):
        name = f.split("/")[-3]
        groups.setdefault(label_of(name), []).append((ev(f), name))
    for lab, pts in groups.items():
        col, mk = marks[lab]
        ax.scatter([p[0][0] for p in pts], [p[0][1] for p in pts], s=18, marker=mk, color=col, edgecolor="white",
                   linewidth=0.5, zorder=3, label=lab)
        for (x, y, _), name in pts:
            if "SimpleRL-Zero" in name:
                ax.annotate("SimpleRL-Zero\n(28% truncated)", (x, y), xytext=(6, -4), textcoords="offset points",
                            color=MUTED, fontsize=6, va="center")
            if "Oat-Zero" in name:
                ax.annotate("Oat-Zero", (x, y), xytext=(-8, -11), textcoords="offset points", color=MUTED, fontsize=6)
    style(ax)
    ax.set_title("Qwen2.5-Math-7B, 2,048 tokens", color=INK)
    ax.set_xlim(25, 92)
    ax.set_ylim(84, 97.5)
    # 1.5B at 3,072 tokens: law from the base's 3,072-token counts
    ax = axes[1]
    bf = glob.glob("results/worth-p28-public-*/BASE15_3k/eval/step0_T1.00_eval.json")[0]
    c0 = np.array(json.load(open(bf))["counts"])
    lx, ly = law_curve(c0, 32)
    m = lx <= 80
    ax.plot(lx[m], ly[m], "--", color=MUTED, linewidth=1.0, label="frozen law (from base counts)")
    b0 = ev(bf)
    ax.scatter([b0[0]], [b0[1]], s=16, marker="s", color=INK, zorder=4)
    ax.annotate("base", (b0[0], b0[1]), xytext=(4, -6), textcoords="offset points", color=MUTED, fontsize=6.5)
    groups = {}
    for f in sorted(glob.glob("results/worth-p28*-public-*/PUB_*_3k/eval/step0_T1.00_eval.json")):
        name = f.split("/")[-3]
        x, y, d = ev(f)
        if d.get("truncated_rate", 0) > 0.9 or x < 1:
            continue
        groups.setdefault(label_of(name), []).append(((x, y), name))
    for lab, pts in groups.items():
        col, mk = marks[lab]
        ax.scatter([p[0][0] for p in pts], [p[0][1] for p in pts], s=18, marker=mk, color=col, edgecolor="white",
                   linewidth=0.5, zorder=3, label=lab)
        for (x, y), name in pts:
            if "pi1" in name:
                ax.annotate("one-shot (1 example)", (x, y), xytext=(4, -8), textcoords="offset points", color=MUTED, fontsize=6)
    style(ax)
    ax.set_title("Qwen2.5-Math-1.5B, 3,072 tokens", color=INK)
    ax.set_xlim(15, 80)
    ax.set_ylim(78, 94)
    h, l = [], []
    for a_ in axes:
        for hh, ll in zip(*a_.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.3, w_pad=1.0, rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / "public_checkpoints.pdf")
    plt.close(fig)


def fig_long():
    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    base = glob.glob("results/worth-v15v-all-*/BASE15/eval/step0_T1.00_eval.json")[0]
    c0 = np.array(json.load(open(base))["counts"])
    lx, ly = law_curve(c0, 32)
    m = (lx >= 45) & (lx <= 74)
    ax.plot(lx[m], ly[m], "--", color=MUTED, linewidth=1.0, label="frozen law")
    for i, s in enumerate((8, 9, 10)):
        pts = sorted(ev(f)[:2] for f in glob.glob(f"results/worth-v15v-all-*/B1k{s}/eval/step*_T1.00_eval.json"))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=INK, alpha=0.4, linewidth=0.9,
                label="Dr. GRPO, 150 steps" if i == 0 else None)
    for arm, col, mk, lab in (("LH1", BLUE, "o", "1,000 steps, lr 1e-6"), ("LH3", ORANGE, "^", "1,000 steps, lr 3e-6")):
        pts = []
        for st in (50, 100, 150, 300, 500, 700, 1000):
            x, y, _ = ev(glob.glob(f"results/worth-p27-long-*/{arm}/eval/step{st}_T1.00_eval.json")[0])
            pts.append((x, y, st))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col, linewidth=1.2, marker=mk, markersize=3.5,
                markeredgecolor="white", markeredgewidth=0.4, label=lab)
        ax.annotate("step 1,000", (pts[-1][0], pts[-1][1]), xytext=(-18, 6), textcoords="offset points", color=MUTED, fontsize=6)
    style(ax)
    ax.set_xlim(48, 73)
    ax.set_ylim(84.5, 90.5)
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "long_training.pdf")
    plt.close(fig)


def fig_finish():
    import sys
    sys.path.insert(0, ".")
    from verifier import last_boxed
    rows = []
    for tag, f in (("base", "/tmp/texts27/base.json.gz"), ("step 25", "/tmp/texts27/b25.json.gz"), ("step 150", "/tmp/texts27/b150.json.gz")):
        d = json.load(gzip.open(f, "rt"))
        n = len(d["samples"])
        right = sum(s["reward"] for s in d["samples"])
        unf = sum(1 for s in d["samples"] if not s["reward"] and last_boxed(s["text"]) is None)
        rows.append((tag, 100 * right / n, 100 * (n - right - unf) / n, 100 * unf / n))
    fig, ax = plt.subplots(figsize=(3.25, 1.35))
    y = np.arange(len(rows))[::-1]
    for (tag, r, w, u), yy in zip(rows, y):
        left = 0
        for v, col, lab in ((r, BLUE, "correct"), (w, ORANGE, "boxed but wrong"), (u, MUTED, "no final box")):
            ax.barh(yy, v - 0.4, left=left + 0.2, color=col, height=0.62, label=lab if tag == "base" else None)
            if v > 8:
                ax.text(left + v / 2, yy, f"{v:.0f}", ha="center", va="center", color="white", fontsize=6.5)
            left += v
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of sampled answers (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "finishing.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_curve()
    fig_public()
    fig_long()
    fig_finish()
    print("figures in", OUT)
