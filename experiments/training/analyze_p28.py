"""Registered analysis of Pre-registration 28 (published checkpoints on the curve) and its amendment.

Per checkpoint (one evaluation, 32 samples): gain = pass@16 minus the reference at the checkpoint's pass@1.
Reference A: Dr. GRPO's same-budget seeds (B1k 8-10 for 1.5B at 1,024 tokens; B7 8-9 for 7B at 2,048), each seed's
  envelope = larger defined value of its trajectory (base and saved steps, T = 1) and its step-150 temperature trajectory
  (T = 1, 1.15, 1.3); the reference value is the mean over seeds whose envelope covers the pass@1; a point outside every
  seed's support is "insufficient overlap".
Reference B: the frozen law (tau = -0.20) from the base's 32-sample counts at the same budget (strict reachability).
Bootstrap: 2,000 problem draws shared by all evaluations of a size; references recomputed in every draw (curves and the
  law's prior and initial state); draws outside support are dropped and their share reported (> 10%: not interpreted).
Verdicts: departure if |gain| >= 2 and Holm-adjusted two-sided p <= 0.05 (families: size x budget x reference);
  equivalence (pointwise) if the 90% interval lies within +/-2; otherwise inconclusive.
Usage: python analyze_p28.py [--draws 2000]
"""

import argparse
import glob
import json
import math
import re
from math import comb

import numpy as np

import law

LABEL = {"LUFFY-Qwen-Math-15B-Zero": "external", "LUFFY-Qwen-Math-7B-Zero": "external", "ReLIFT": "external",
         "ExGRPO-LUFFY": "external lineage (start is not the base)", "random": "random reward",
         "LUFFY-Qwen-Math-7B-Zero-On-Policy": "correctness-only (LUFFY's on-policy baseline)"}


def label(name):
    for k in ("ExGRPO-LUFFY", "LUFFY-Qwen-Math-7B-Zero-On-Policy", "LUFFY-Qwen-Math-15B-Zero", "LUFFY-Qwen-Math-7B-Zero",
              "ReLIFT", "random"):
        if k in name:
            return LABEL[k]
    return "correctness-only"


def counts(f):
    d = json.load(open(f))
    return np.array(d["counts"]), d["n"], d


def pk16(c, n):
    return float(np.mean([1 - comb(n - int(x), 16) / comb(n, 16) for x in c]))


def curve_at(pts, u):
    pts = sorted(pts)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return float(np.interp(u, xs, ys)) if xs[0] <= u <= xs[-1] else None


def ref_A(seeds, u, idx=None):
    vals = []
    for traj, temps in seeds:
        tv = curve_at([(c[idx].mean() / n if idx is not None else c.mean() / n, pk16(c[idx] if idx is not None else c, n)) for c, n in traj], u)
        mv = curve_at([(c[idx].mean() / n if idx is not None else c.mean() / n, pk16(c[idx] if idx is not None else c, n)) for c, n in temps], u) if len(temps) > 1 else None
        cand = [v for v in (tv, mv) if v is not None]
        if cand:
            vals.append(max(cand))
    return float(np.mean(vals)) if vals else None


def ref_B(c0, n0, u, idx=None):
    c = c0[idx] if idx is not None else c0
    tr = law.trajectory(law.initial_theta(c, n0, law.fit_prior(c, n0)), -0.20)
    try:
        return law.pass_k_true(law.at_pass1(tr, u, strict=True), 16)
    except ValueError:
        return None


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def seeds_15():
    out = []
    for s in (8, 9, 10):
        traj = [counts(glob.glob("results/worth-v15v-all-*/BASE15/eval/step0_T1.00_eval.json")[0])[:2]]
        traj += [counts(f)[:2] for f in sorted(glob.glob(f"results/worth-v15v-all-*/B1k{s}/eval/step*_T1.00_eval.json"))]
        temps = [counts(f)[:2] for f in sorted(glob.glob(f"results/worth-v15v-all-*/B1k{s}/eval/step150_T*_eval.json"))]
        out.append((traj, temps))
    return out


def seeds_7():
    """B7's vLLM evaluations as in analyze_v20.py: the test job (worth-v20v-testw6: B7s8 steps 25-150, B7s9 step 150,
    BASE7v) and the re-evaluation jobs (worth-v20v-<ARM>); one file per (seed, step, temperature)."""
    out = []
    base = glob.glob("results/worth-v20v-testw6-*/BASE7v/eval/step0_T1.00_eval.json")[0]
    pats = ("results/worth-v20v-testw6-*/B7s{s}/eval/{f}", "results/worth-v20v-[A-Z]*-s0-*/B7s{s}/eval/{f}")
    for s in (8, 9):
        found = {}
        for pat in pats:
            for f in glob.glob(pat.format(s=s, f="step*_T*_eval.json")):
                found.setdefault(f.split("/")[-1], f)
        traj = [counts(base)[:2]] + [counts(f)[:2] for k, f in sorted(found.items()) if k.endswith("_T1.00_eval.json")]
        temps = [counts(f)[:2] for k, f in sorted(found.items()) if k.startswith("step150_")]
        assert len(traj) == 5 and len(temps) == 3, (s, sorted(found))
        out.append((traj, temps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    groups = {
        ("1.5B", "1k"): ("PUB_*_1k", seeds_15(), glob.glob("results/worth-v15v-all-*/BASE15/eval/step0_T1.00_eval.json")[0]),
        ("1.5B", "3k"): ("PUB_*_3k", None, glob.glob("results/worth-p28-public-*/BASE15_3k/eval/step0_T1.00_eval.json")[0]),
        ("7B", "2k"): ("PUB_*_2k", seeds_7(), glob.glob("results/worth-v20-BASE7s-*/BASE7/eval/step0_T1.00_eval.json")[0]),
    }
    for (size, budget), (pat, seedsA, basef) in groups.items():
        c0, n0, _ = counts(basef)
        files = sorted(glob.glob(f"results/worth-p28*-public-*/{pat}/eval/step0_T1.00_eval.json"))
        P = len(c0)
        rng = np.random.default_rng(28 if size == "1.5B" else 29)
        idx = [rng.integers(0, P, P) for _ in range(a.draws)]
        lawB = [law.trajectory(law.initial_theta(c0[i], n0, law.fit_prior(c0[i], n0)), -0.20) for i in idx]
        print(f"\n== {size} at {budget} tokens: {len(files)} checkpoints (base pass@1 {100 * c0.mean() / n0:.1f})")
        for refname in ("A", "B"):
            if refname == "A" and seedsA is None:
                continue
            rows = []
            for f in files:
                name = f.split("/")[-3][4:]
                c, n, d = counts(f)
                if d.get("truncated_rate", 0) > 0.9:
                    rows.append((name, None, "protocol failure (truncation %.0f%%)" % (100 * d["truncated_rate"])))
                    continue
                u, obs = c.mean() / n, pk16(c, n)
                r0 = ref_A(seedsA, u) if refname == "A" else ref_B(c0, n0, u)
                if r0 is None:
                    rows.append((name, None, "insufficient overlap" if refname == "A" else "outside the law's range"))
                    continue
                boots, drop = [], 0
                for i, tr in zip(idx, lawB):
                    ui = c[i].mean() / n
                    if refname == "A":
                        ri = ref_A(seedsA, ui, i)
                    else:
                        try:
                            ri = law.pass_k_true(law.at_pass1(tr, ui, strict=True), 16)
                        except ValueError:
                            ri = None
                    if ri is None:
                        drop += 1
                        continue
                    boots.append(100 * (pk16(c[i], n) - ri))
                boots = np.array(boots)
                est = 100 * (obs - r0)
                p2 = float(min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))) if len(boots) else 1.0
                rows.append((name, {"u": 100 * u, "obs": 100 * obs, "ref": 100 * r0, "est": est, "ci90": np.percentile(boots, [5, 95]),
                                    "ci95": np.percentile(boots, [2.5, 97.5]), "p": p2, "drop": drop / a.draws}, label(name)))
            scored = [r for r in rows if r[1] is not None and r[1]["drop"] <= 0.10]
            adj = holm(np.array([r[1]["p"] for r in scored])) if scored else []
            amap = {r[0]: q for r, q in zip(scored, adj)}
            print(f"   reference {refname}:")
            tally = {}
            for name, r, lab in rows:
                if r is None:
                    print(f"      {name:58s} {lab}")
                    tally[lab.split(" (")[0]] = tally.get(lab.split(" (")[0], 0) + 1
                    continue
                if r["drop"] > 0.10:
                    v = "not interpreted (support)"
                else:
                    q = amap[name]
                    v = ("departure up" if r["est"] > 0 else "departure down") if abs(r["est"]) >= 2 and q <= 0.05 else \
                        ("equivalent" if r["ci90"][0] >= -2 and r["ci90"][1] <= 2 else "inconclusive")
                tally[v] = tally.get(v, 0) + 1
                print(f"      {name:58s} pass@1 {r['u']:5.1f} pass@16 {r['obs']:5.1f} ref {r['ref']:5.1f} gain {r['est']:+5.2f} "
                      f"90% [{r['ci90'][0]:+.2f}, {r['ci90'][1]:+.2f}] dropped {100 * r['drop']:.0f}% -> {v} [{lab}]")
            print(f"      tally: {tally}")


if __name__ == "__main__":
    main()
