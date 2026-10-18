"""Figures and numbers for the research report (report/report.tex) from all results so far.

Usage: python plot_report.py   -> report/figures/*.pdf and report/numbers.json
"""

import json
import math
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
OUT = ROOT / "report" / "figures"
sys.path.insert(0, str(HERE))
from analyze_v2 import window_stats, paired  # noqa: E402
from train_grpo import fixed_advantage  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
ARM_FILE = {"DR": "baseline", "PSR": "psr", "NSR": "nsr", "WR": "wreinforce", "PSRv": "psr", "NSRv": "nsr",
            "B25": "shift", "B50": "shift", "B75": "shift", "PSRlow": "psr", "L128": "baseline", "L32": "baseline",
            "O128": "optimal", "S128": "scaled", "L256": "baseline", "L64": "baseline", "O64": "optimal", "PK4": "passk"}
COL = {"DR": "#1f4e79", "PSR": "#2e75b6", "NSR": "#c00000", "WR": "#ed7d31", "PSRv": "#2e75b6", "NSRv": "#c00000",
       "B25": "#5b9bd5", "B50": "#a9a9a9", "B75": "#f4b183", "PSRlow": "#70ad47", "PK4": "#7030a0"}
LABEL = {"DR": "Dr. GRPO", "PSR": "reward right only (PSR)", "NSR": "penalize wrong only (NSR)",
         "WR": "W-REINFORCE (0.1 / -1)", "PSRv": "PSR, verl-like", "NSRv": "NSR, verl-like",
         "B25": "r - 0.25", "B50": "r - 0.5", "B75": "r - 0.75", "PSRlow": "PSR, lr / 2.5", "PK4": "pass@4 objective"}


def load(version):
    runs = {}
    for job in sorted(RES.glob(f"worth-{version}-*")):
        m = re.match(rf"worth-{version}-(\w+?)-s(\d+)-", job.name)
        if not m:
            continue
        name, seed = m.group(1), int(m.group(2))
        arm = ARM_FILE.get(name)
        folder = job / name
        f = folder / f"{arm}_train.json"
        if not f.exists():
            continue
        get = lambda suffix: json.loads((folder / f"{arm}{suffix}").read_text()) if (folder / f"{arm}{suffix}").exists() else None
        rec = {"train": json.loads(f.read_text()), "eval": get("_eval.json") or [], "passk": get("_passk.json") or [],
               "final": get("_final_eval.json"),
               "calib": json.loads((folder / "calibration.json").read_text()) if (folder / "calibration.json").exists() else None}
        key = (name, seed)
        if key not in runs or len(rec["train"]) > len(runs[key]["train"]):
            runs[key] = rec
    return runs


def smooth(y, k=10):
    y = np.asarray(y, float)
    return np.arange(k - 1, len(y)), np.convolve(y, np.ones(k) / k, mode="valid")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def predicted_m2(arm, steps, calib, shift=0.0):
    F1, F0 = calib["F1"], calib["F0"]
    tot = cnt = 0
    for x in steps:
        for k, n in x["group_counts"]:
            m = k / n
            a1 = fixed_advantage(arm, 1, m, calib.get("w", 1.0), 0.1, shift)
            a0 = fixed_advantage(arm, 0, m, calib.get("w", 1.0), 0.1, shift)
            tot += k * a1 * a1 * F1 + (n - k) * a0 * a0 * F0
            cnt += 1
    return tot / cnt if cnt else math.nan


def main():
    numbers = {}
    v3 = load("v3")
    # 1. learning curves of the four rules (first-block training reward: unbiased accuracy on fresh prompts)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for col, seed in enumerate((1, 2)):
        for name in ("DR", "PSR", "NSR", "WR"):
            r = v3.get((name, seed))
            if not r:
                continue
            fb = [x["first_block_reward"] for x in r["train"]]
            x, y = smooth(fb)
            ax[col].plot(x, 100 * y, color=COL[name], lw=1.3, label=LABEL[name])
        ax[col].set_title(f"Seed {seed}")
        ax[col].set_xlabel("Training step")
    ax[0].set_ylabel("Training accuracy on fresh prompts (%)\n(first 4 answers, 10-step mean)")
    ax[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "pn_curves")

    # 2. pass@k curves at step 150
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    base = next((p for p in v3.get(("DR", 1), {}).get("passk", []) if p["step"] == 0), None)
    numbers["passk"] = {}
    for col, seed in enumerate((1, 2)):
        if base:
            ks = sorted(int(k) for k in base["pass_at_k"])
            ax[col].plot(ks, [100 * base["pass_at_k"][str(k)] for k in ks], "k--", lw=1, label="base model")
        for name in ("DR", "PSR", "NSR", "WR"):
            r = v3.get((name, seed))
            p = next((p for p in (r or {}).get("passk", []) if p["step"] == 150), None)
            if not p:
                continue
            ks = sorted(int(k) for k in p["pass_at_k"])
            ax[col].plot(ks, [100 * p["pass_at_k"][str(k)] for k in ks], "o-", ms=3, color=COL[name], label=LABEL[name])
            numbers["passk"][f"{name}_s{seed}"] = {k: round(100 * p["pass_at_k"][str(k)], 1) for k in ks}
        ax[col].set_xscale("log", base=2)
        ax[col].set_title(f"Seed {seed}, step 150")
        ax[col].set_xlabel("k")
    if base:
        numbers["passk"]["base"] = {k: round(100 * v, 1) for k, v in base["pass_at_k"].items()}
    ax[0].set_ylabel("pass@k on 364 MATH-500 problems (%)")
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "passk")

    # 3. noise: measured per-prompt second moment against the theorem
    rows = []
    for (name, seed), r in sorted(v3.items()):
        steps = [x for x in r["train"] if x["step"] < 50 and x.get("split")]
        st = window_stats(steps)
        if not st or not r["calib"]:
            continue
        rows.append((name, seed, st["m2"], predicted_m2(ARM_FILE[name], steps, r["calib"])))
    v7 = load("v7")
    for (name, seed), r in sorted(v7.items()):
        steps = [x for x in r["train"] if x["step"] < 50 and x.get("split")]
        st = window_stats(steps)
        if not st or not r["calib"] or len(steps) < 15:
            continue
        b = {"B25": 0.25, "B50": 0.5, "B75": 0.75}.get(name, 0.0)
        rows.append((name, seed, st["m2"], predicted_m2(ARM_FILE[name], steps, r["calib"], b)))
    numbers["noise"] = [{"run": n, "seed": s, "measured": m, "theorem": p, "ratio": m / p} for n, s, m, p in rows]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    seen = set()
    for n, s, m, p in rows:
        lab = LABEL.get(n, n) if n not in seen else None
        seen.add(n)
        ax.scatter(p, m, color=COL.get(n, "gray"), s=30, marker="o" if s % 2 else "s", zorder=3, label=lab)
    lim = [min(min(p for *_, p in rows), min(m for _, _, m, _ in rows)) / 1.5,
           max(max(p for *_, p in rows), max(m for _, _, m, _ in rows)) * 1.5]
    ax.plot(lim, lim, "k-", lw=0.8)
    ax.fill_between(lim, [0.7 * v for v in lim], [1.3 * v for v in lim], color="gray", alpha=0.15, label="within 30%")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Theorem: sum of A^2 F per prompt")
    ax.set_ylabel("Measured per-prompt second moment")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "noise_validation")

    # 4. accuracy-diversity frontier (pass@1 vs pass@32 on 128 problems)
    v5, v6, v8 = load("v5"), load("v6"), load("v8")
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
    numbers["frontier"] = {}
    for col, (title, src, names) in enumerate((("Noise: PSR vs NSR (seeds 3, 4)", {**v5, **{k: v for k, v in v7.items() if k[0] == "PSRlow"}}, ("PSR", "NSR", "PSRlow")),
                                               ("NSR paper's implementation (seeds 1, 2)", v6, ("PSRv", "NSRv")),
                                               ("Objective: pass@4 vs Dr. GRPO (seeds 3, 4)", v8, ("DR", "PK4")))):
        for (name, seed), r in sorted(src.items()):
            if name not in names or not r["passk"]:
                continue
            pts = sorted((p["step"], 100 * p["pass_at_k"]["1"], 100 * p["pass_at_k"]["32"]) for p in r["passk"])
            numbers["frontier"][f"{name}_s{seed}"] = pts
            ax[col].plot([p[1] for p in pts], [p[2] for p in pts], "o-" if seed % 2 else "s--", ms=3.5,
                         color=COL[name], label=f"{LABEL[name]}, seed {seed}")
        ax[col].set_title(title, fontsize=9)
        ax[col].set_xlabel("pass@1 (%)")
        ax[col].legend(frameon=False, fontsize=6.5)
    ax[0].set_ylabel("pass@32 (%)")
    fig.tight_layout()
    save(fig, "frontier")

    # 5. rollout rules at matched answers (paired, 3,382-problem set)
    v2 = load("v2")
    numbers["rules"] = {}
    for seed in (1, 2):
        base = v2.get(("L128", seed))
        if not base or not base["final"]:
            continue
        for name in ("O128", "S128", "L32"):
            o = v2.get((name, seed))
            if o and o["final"]:
                d, se = paired(o["final"]["correct"], base["final"]["correct"])
                numbers["rules"][f"{name}_s{seed}"] = (round(d, 2), round(se, 2), round(100 * o["final"]["accuracy"], 1))
        numbers["rules"][f"L128_s{seed}"] = (0.0, 0.0, round(100 * base["final"]["accuracy"], 1))
    # noise scale over training for Dr. GRPO
    numbers["noise_scale"] = {}
    for seed in (1, 2):
        r = v2.get(("L128", seed))
        if r:
            for lo, hi in ((0, 50), (50, 150)):
                st = window_stats([x for x in r["train"] if lo <= x["step"] < hi])
                numbers["noise_scale"][f"L128_s{seed}_{lo}_{hi}"] = round(st["B_noise"]) if st and math.isfinite(st["B_noise"]) else None
    # summary table of v3
    numbers["v3"] = {}
    for (name, seed), r in sorted(v3.items()):
        fb = [x["first_block_reward"] for x in r["train"]]
        ent = [x["probe"]["entropy"] for x in r["train"] if x.get("probe") and x["step"] >= 100]
        g150 = next((100 * e["accuracy"] for e in r["eval"] if e["step"] == 150), None)
        numbers["v3"][f"{name}_s{seed}"] = {"steps": len(r["train"]), "J_25_50": round(100 * np.mean(fb[25:50]), 1),
                                            "J_100_150": round(100 * np.mean(fb[100:150]), 1) if len(fb) >= 150 else None,
                                            "entropy_100_150": round(float(np.mean(ent)), 3) if ent else None,
                                            "greedy150": round(g150, 1) if g150 else None}
    (ROOT / "report").mkdir(exist_ok=True)
    (ROOT / "report" / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1)[:6000])


if __name__ == "__main__":
    main()
