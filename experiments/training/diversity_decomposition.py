"""Two kinds of diversity at matched pass@1 (descriptive analysis across experiments; 1.5B, 1,024 tokens, 511 test
problems; first 32 samples per problem for every point).

For a checkpoint with pass@1 u, each metric is compared with the same-seed Dr. GRPO trajectory (B1k_s: base and steps
25-150, T = 1) interpolated at u:
- coverage: pass@16 gap (pp);
- across-problem spread: pass@2 gap (pp); at matched pass@1, pass@2 = 2 E[p] - E[p^2] (unbiased per-problem
  estimates), so its gap equals minus the change in the across-problem second moment of success, i.e. a positive gap
  means success is spread more evenly across problems;
- within-problem answer diversity: distinct valid incorrect answer classes per problem (P32's outcome classes; "no
  parsed final box" excluded), and the collision probability among valid incorrect samples (problems with at least two;
  lower = more varied wrong answers).
Gaps are averaged over eligible checkpoints within seed (pass@1 inside B1k_s's range on the full data; at least two)
and equally over seeds; 90% intervals from 1,000 problem-bootstrap draws shared by all rows (seed 7). Control row:
each B1k seed against the next seed's trajectory (8->9, 9->10, 10->8), which shows the interpolation bias.
Usage: python diversity_decomposition.py [--draws 1000]
"""

import argparse
import glob
import gzip
import json
from math import comb
from pathlib import Path

import numpy as np

from analyze_p32 import outcome

P = 511
SEEDS = (8, 9, 10)
T32 = {k: np.array([1 - comb(32 - x, k) / comb(32, k) for x in range(33)]) for k in (1, 2, 16)}
TEXTS = Path("/tmp/div")
P32T = Path("/tmp/p32/texts")
RES = Path(__file__).resolve().parent / "results"
_CACHE = {}


def load(path):
    if path in _CACHE:
        return _CACHE[path]
    d = json.load(gzip.open(path, "rt"))
    per = {}
    for s in d["samples"]:
        per.setdefault(s["problem"], []).append(s)
    c, D, coll, ok = np.zeros(P, int), np.zeros(P), np.zeros(P), np.zeros(P, bool)
    for i in range(P):
        S = per[i][:32]
        assert len(S) == 32, path
        c[i] = sum(bool(s["reward"]) for s in S)
        wrong = [outcome(s["text"], s["reward"]) for s in S if not s["reward"]]
        valid = [w for w in wrong if w != ("none",)]
        D[i] = len(set(valid))
        if len(valid) >= 2:
            t = {}
            for w in valid:
                t[w] = t.get(w, 0) + 1
            coll[i] = sum(v * (v - 1) for v in t.values()) / (len(valid) * (len(valid) - 1))
            ok[i] = True
    _CACHE[path] = (c, D, coll, ok)
    return _CACHE[path]


def metrics(x, idx):
    c, D, coll, ok = x
    ci, oi = c[idx], ok[idx]
    return {"p1": T32[1][ci].mean(), "p16": 100 * T32[16][ci].mean(), "p2": 100 * T32[2][ci].mean(),
            "D": D[idx].mean(), "coll": coll[idx][oi].mean() if oi.any() else np.nan}


def ref_paths(seed):
    return [str(P32T / "BASE15" / "step0_T1.00_texts.json.gz")] + [str(P32T / f"B1k{seed}" / f"step{st}_T1.00_texts.json.gz") for st in (25, 50, 100, 150)]


def gaps(arm, refs, idx):
    """arm: {seed: [texts paths]}; refs: {seed: [texts paths]} -> per-metric pooled gap, or None per seed if < 2 eligible."""
    out = {k: [] for k in ("p16", "p2", "D", "coll")}
    for s, paths in arm.items():
        R = [metrics(load(p), idx) for p in refs[s]]
        xs = np.array([r["p1"] for r in R])
        o = np.argsort(xs)
        per = {k: [] for k in out}
        for p in paths:
            m = metrics(load(p), idx)
            if not xs.min() <= m["p1"] <= xs.max():
                continue
            for k in out:
                ys = np.array([r[k] for r in R])
                per[k].append(m[k] - float(np.interp(m["p1"], xs[o], ys[o])))
        for k in out:
            out[k].append(np.mean(per[k]) if len(per[k]) >= 2 else np.nan)
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1000)
    a = ap.parse_args()
    refs = {s: ref_paths(s) for s in SEEDS}
    st4, st3 = (25, 50, 100, 150), (50, 100, 150)
    arms = {}
    for name, steps in (("PSR1k", st4), ("NSR1k", st3), ("AC1k", st3), ("HG1k", st3), ("HGm1k", st3), ("PG1k", st4)):
        arms[name] = {s: [str(TEXTS / f"{name}{s}" / f"step{st}_T1.00_texts.json.gz") for st in steps] for s in SEEDS}
    for name in ("FL10", "FL5"):
        arms[f"Flat-RLVR {name}"] = {s: [str(TEXTS / f"{name}s{s}" / f"step{st}_T1.00_texts.json.gz") for st in st4] for s in SEEDS}
    for name, label in (("OU", "P32 UCB-Con"), ("OB", "P32 Batch")):
        arms[label] = {s: [glob.glob(str(RES / f"worth-p32-oe-*/{name}s{s}/eval/step{st}_T1.00_texts.json.gz"))[0] for st in st4] for s in SEEDS}
    early = (1, 2, 5, 10, 25, 50)
    for name, label in (("S2R25", "SFT start R25 + RL"), ("S2H25", "SFT start H25 + RL")):
        arms[label] = {s: [glob.glob(str(RES / f"worth-p30-stage2-*/{name}s{s}/eval/step{st}_T1.00_texts.json.gz"))[0] for st in early] for s in SEEDS}
    opd = {s: sorted(glob.glob(str(RES / f"worth-p3*-s*/P34OPD25s{s}/eval/step*_T1.00_texts.json.gz"))) for s in SEEDS}
    if all(len(opd[s]) >= 6 for s in SEEDS):
        arms["OPD start + RL (P34)"] = {s: [p for p in opd[s] if any(f"step{st}_" in p for st in early)] for s in SEEDS}
    nxt = {8: 9, 9: 10, 10: 8}
    control = {s: [str(P32T / f"B1k{s}" / f"step{st}_T1.00_texts.json.gz") for st in st4] for s in SEEDS}
    rng = np.random.default_rng(7)
    draws = [rng.integers(0, P, P) for _ in range(a.draws)]
    full = np.arange(P)
    for k in list(arms):  # keep the seeds whose texts exist (HGm1k has seeds 8 and 9 only)
        arms[k] = {s: v for s, v in arms[k].items() if all(Path(x).exists() for x in v)}
        if len(arms[k]) < 2:
            print(f"{k}: fewer than two seeds with texts; skipped")
            del arms[k]
    rows = [("control: Dr. GRPO seed vs next seed", control, {s: refs[nxt[s]] for s in SEEDS})] + \
           [(f"{k} (seeds {','.join(map(str, v))})" if len(v) < 3 else k, v, refs) for k, v in arms.items()]
    print(f"{'arm':40s} {'pass@16 gap':>22s} {'pass@2 gap (spread)':>22s} {'distinct wrong classes':>24s} {'wrong-answer collision':>24s}")
    for label, arm, rf in rows:
        g0 = gaps(arm, rf, full)
        bs = [gaps(arm, rf, i) for i in draws]
        cells = []
        for k, fmt in (("p16", "{:+.2f}"), ("p2", "{:+.2f}"), ("D", "{:+.2f}"), ("coll", "{:+.3f}")):
            v = np.array([b[k] for b in bs]); v = v[np.isfinite(v)]
            lo, hi = (np.percentile(v, [5, 95]) if len(v) else (np.nan, np.nan))
            cells.append(f"{fmt.format(g0[k])} [{fmt.format(lo)}, {fmt.format(hi)}]")
        print(f"{label:40s} " + " ".join(f"{c:>23s}" for c in cells), flush=True)
    # starts (step 0), single points against each seed's trajectory, averaged over seeds; no seeds to average over
    print("\nStarts (step 0; first 32 of 64 samples), mean over B1k seeds:")
    for label, path in (("H25 start", glob.glob(str(RES / "worth-p30-stage1b-*/H025_test/eval/step0_T1.00_texts.json.gz"))),
                        ("R25 start", glob.glob(str(RES / "worth-p30-stage1b-*/R025_test/eval/step0_T1.00_texts.json.gz"))),
                        ("OPD25 start", glob.glob(str(RES / "worth-p30-stage2-*/OPD25_test/eval/step0_T1.00_texts.json.gz")))):
        if not path:
            continue
        arm = {s: [path[0], path[0]] for s in SEEDS}  # the same point twice satisfies the two-checkpoint rule
        g0 = gaps(arm, refs, full)
        print(f"   {label}: pass@16 {g0['p16']:+.2f}, pass@2 {g0['p2']:+.2f}, distinct wrong {g0['D']:+.2f}, collision {g0['coll']:+.3f}")


if __name__ == "__main__":
    main()
