"""Analysis of the Pre-registration 3 runs (budget ladder and in-training gradient diagnostics).

Usage: python analyze_v2.py [results_dir]   (default: results/, one folder per job, as
written by experiments/sagemaker/fetch_live.sh). Runs that are still going are analyzed
over the steps they have finished.

Split-half statistics per step (prompts split into halves A, B; u_i the arm's update from
prompt i, rho the expected Dr. GRPO reference update per prompt):
    E <g_A, g_B>   = P_A P_B s,                         s  = |E u|^2
    E |g_A|^2      = P_A m2 + P_A (P_A - 1) s,          m2 = E |u|^2
    E <g_A, ref_B> = P_A P_B q,                         q  = <E u, rho>
Window estimates are ratios of sums over steps. With n answers per prompt:
    noise scale in answers  B_noise = n (m2 - s) / s
    efficiency along the population gradient  Eff = q^2 / (n (m2 - s)), times Z^2 (Z the
    loss normalizer), which makes it comparable across budgets.
"""

import json
import math
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WINDOWS = ((0, 50), (50, 150), (0, 150))


def load_runs(root):
    runs = {}
    for job in sorted(root.glob("worth-v2-*")):
        m = re.match(r"worth-v2-(\w+?)-s(\d+)-", job.name)
        if not m:
            continue
        name, seed = m.group(1), int(m.group(2))
        folder = job / name
        if not folder.exists():
            continue
        arm = next((p.name[: -len("_train.json")] for p in folder.glob("*_train.json")), None)
        if arm is None:
            continue
        load = lambda suffix: json.loads((folder / f"{arm}{suffix}").read_text()) if (folder / f"{arm}{suffix}").exists() else None
        runs[(name, seed)] = {"arm": arm, "train": load("_train.json") or [], "eval": load("_eval.json") or [],
                              "sampled": load("_sampled_eval.json") or [], "final": load("_final_eval.json")}
    return runs


def window_stats(steps):
    keys = ("ab", "aa", "bb", "ar", "pp", "pa2", "np", "ans", "grp")
    tot = dict.fromkeys(keys, 0.0)
    for r in steps:
        sp = r.get("split")
        if not sp or sp["prompts_a"] == 0 or sp["prompts_b"] == 0:
            continue
        pa, pb = sp["prompts_a"], sp["prompts_b"]
        z2 = sp["norm"] ** 2
        tot["ab"] += sp["ab"] * z2
        tot["aa"] += (sp["aa"] + sp["bb"]) * z2
        tot["ar"] += sp["ar"] * z2
        tot["pp"] += pa * pb
        tot["pa2"] += pa * (pa - 1) + pb * (pb - 1)
        tot["np"] += pa + pb
        tot["ans"] += r["answers"]
        tot["grp"] += r["groups"]
    if tot["pp"] == 0:
        return None
    s = tot["ab"] / tot["pp"]
    m2 = (tot["aa"] - s * tot["pa2"]) / tot["np"]
    q = tot["ar"] / tot["pp"]
    n = tot["ans"] / tot["grp"]
    v = m2 - s
    return {"s": s, "m2": m2, "q": q, "n": n,
            "B_noise": n * v / s if s > 0 else math.inf,
            "eff": q * q / (n * v) if v > 0 else math.nan,
            "signed_eff": math.copysign(q * q, q) / (n * v) if v > 0 else math.nan}


def bootstrap(steps, key, reps=400, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        st = window_stats([rng.choice(steps) for _ in steps])
        if st and math.isfinite(st[key]):
            vals.append(st[key])
    vals.sort()
    if len(vals) < 20:
        return None
    return vals[int(0.05 * len(vals))], vals[int(0.95 * len(vals)) - 1]


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else math.nan
    return 100 * mean, 100 * sd / math.sqrt(n)


def auc(evals, lo=10, hi=150):
    pts = [e["accuracy"] for e in evals if lo <= e["step"] <= hi]
    return 100 * sum(pts) / len(pts) if pts else math.nan


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    runs = load_runs(root)
    if not runs:
        print("no runs found under", root)
        return
    print("run   seed  steps  greedy curve (step: %)                                   AUC10-150  sampled@150  final(3382)  entropy 0-50/50-150")
    for (name, seed), r in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        curve = " ".join(f"{e['step']}:{100 * e['accuracy']:.1f}" for e in r["eval"])
        samp = next((100 * e["accuracy"] for e in r["sampled"] if e["step"] == 150), math.nan)
        fin = 100 * r["final"]["accuracy"] if r["final"] else math.nan
        ent = []
        for lo, hi in WINDOWS[:2]:
            xs = [x["entropy"] for x in r["train"] if lo <= x["step"] < hi and x.get("entropy") is not None]
            ent.append(sum(xs) / len(xs) if xs else math.nan)
        print(f"{name:5s} {seed:4d} {len(r['train']):6d}  {curve:58s} {auc(r['eval']):8.1f}  {samp:10.1f}  {fin:10.1f}   {ent[0]:.3f}/{ent[1]:.3f}")

    print("\nGradient diagnostics (ratio of sums over steps; 90% bootstrap interval over steps)")
    print("run   seed  window    answers/prompt  B_noise (answers)          Eff x Z^2 (relative to L128 same seed and window)")
    for (name, seed), r in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        for lo, hi in WINDOWS:
            steps = [x for x in r["train"] if lo <= x["step"] < hi]
            st = window_stats(steps)
            if not st:
                continue
            ref = runs.get(("L128", seed))
            ref_st = window_stats([x for x in ref["train"] if lo <= x["step"] < hi]) if ref else None
            rel = st["signed_eff"] / ref_st["signed_eff"] if ref_st and ref_st["signed_eff"] else math.nan
            ci = bootstrap(steps, "B_noise")
            ci_txt = f"[{ci[0]:.0f}, {ci[1]:.0f}]" if ci else ""
            print(f"{name:5s} {seed:4d}  {lo:3d}-{hi:3d}   {st['n']:8.2f}      {st['B_noise']:9.0f} {ci_txt:16s}  {st['signed_eff']:.3g} ({rel:.2f}x)")

    print("\nPredicted progress per step relative to L128 (same seed, steps 0-49), from each run's own measurements")
    print("  SGD with tuned step: (q_R/q_0)^2 (s_0 + n_0 V_0/B_0) / (s_R + n_R V_R/B_R);"
          " Adam, noise-limited: square root of the efficiency-times-budget ratio")
    print("run   seed  budget  predicted SGD  predicted Adam   observed AUC10-50 minus L128")
    for (name, seed), r in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        ref = runs.get(("L128", seed))
        if not ref or not r["train"]:
            continue
        a = window_stats([x for x in r["train"] if x["step"] < 50])
        b = window_stats([x for x in ref["train"] if x["step"] < 50])
        if not a or not b:
            continue
        budget = sum(x["answers"] for x in r["train"]) / len(r["train"])
        b0 = sum(x["answers"] for x in ref["train"]) / len(ref["train"])
        sgd = (a["q"] / b["q"]) ** 2 * (b["s"] + b["n"] * (b["m2"] - b["s"]) / b0) / (a["s"] + a["n"] * (a["m2"] - a["s"]) / budget)
        adam = math.sqrt((a["eff"] * budget) / (b["eff"] * b0)) if a["eff"] > 0 and b["eff"] > 0 else math.nan
        obs = auc(r["eval"], 10, 50) - auc(ref["eval"], 10, 50)
        print(f"{name:5s} {seed:4d}  {budget:6.0f}  {sgd:12.2f}x  {adam:12.2f}x   {obs:+.1f} points")

    print("\nPaired final accuracy on the 3,382-problem set (points, standard error)")
    for seed in sorted({s for _, s in runs}):
        base = runs.get(("L128", seed))
        if not base or not base["final"]:
            continue
        for name in ("O128", "O64", "S128", "L64", "L256", "L32"):
            other = runs.get((name, seed))
            if other and other["final"]:
                d, se = paired(other["final"]["correct"], base["final"]["correct"])
                print(f"  seed {seed}: {name} - L128 = {d:+.1f} ({se:.1f})")


if __name__ == "__main__":
    main()
