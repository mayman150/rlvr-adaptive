"""Registered analysis of Pre-registration 32 (outcome-based exploration: OU = UCB-Con, OB = Batch; seeds 8-10).

Primary: pass@16 at matched pass@1 against B1k's trajectory-and-temperature envelope (Pre-registration 21's estimator
and guards, matched.py), Holm over the two arms, verdicts "moves" / "lowers" / "average equivalence" / inconclusive.
Secondary (not multiplicity-adjusted): pass@32 at matched pass@1 against the same reference (+1.7 as a fixed reference
magnitude); the manipulation check on test texts (distinct valid incorrect outcome classes per problem, and the
conditional collision probability among incorrect samples), each against B1k's same-seed trajectory interpolated at the
checkpoint's pass@1, with the primary analysis' bootstrap draws (default_rng(0), problem resampling).
Texts: --texts DIR with DIR/<NAME>/step<S>_T1.00_texts.json.gz for BASE15, B1k8-10 and OU/OB s8-10 (from S3).
Usage: python analyze_p32.py [--draws 2000] [--texts /tmp/p32/texts]
"""

import argparse
import glob
import gzip
import json
from pathlib import Path

import numpy as np

import analyze_v12 as A
import analyze_v21 as V
import matched as M
from verifier import as_number, last_boxed, normalize

SEEDS = (8, 9, 10)
S = (25, 50, 100, 150)
ARMS = ("OU", "OB")


def outcome(text, rewarded):
    """train_grpo.oe_outcome, repeated here so the analysis needs neither torch nor transformers."""
    if rewarded:
        return ("correct",)
    b = last_boxed(text)
    if b is None:
        return ("none",)
    v = as_number(normalize(b))
    return ("number", v) if v is not None else ("text", normalize(b))


def text_metrics(path, n_problems):
    """Per problem: pass@1, distinct valid incorrect classes, collision probability among valid incorrect samples (nan
    if fewer than two), and the no-parsed-box share."""
    d = json.load(gzip.open(path, "rt"))
    assert len(d["samples"]) % n_problems == 0 and max(s["problem"] for s in d["samples"]) == n_problems - 1, path
    cls = [[] for _ in range(n_problems)]
    corr = np.zeros(n_problems)
    for s in d["samples"]:
        cls[s["problem"]].append(outcome(s["text"], s["reward"]))
        corr[s["problem"]] += bool(s["reward"])
    k = len(d["samples"]) // n_problems
    distinct, coll, none = np.zeros(n_problems), np.full(n_problems, np.nan), np.zeros(n_problems)
    for i, c in enumerate(cls):
        wrong = [x for x in c if x != ("correct",)]
        valid = [x for x in wrong if x != ("none",)]
        distinct[i] = len(set(valid))
        none[i] = sum(x == ("none",) for x in c) / len(c)
        if len(valid) >= 2:
            tally = {}
            for x in valid:
                tally[x] = tally.get(x, 0) + 1
            m = len(valid)
            coll[i] = sum(v * (v - 1) for v in tally.values()) / (m * (m - 1))
    return {"p1": corr / k, "distinct": distinct, "collision": coll, "none": none}


def wmean(x, w):
    m = ~np.isnan(x)
    return float((w[m] * x[m]).sum() / w[m].sum())


def diversity(texts, draws):
    """Manipulation check: arm minus B1k's same-seed trajectory (base and steps 25-150, T = 1) at matched pass@1."""
    n = 511
    get = {}

    def tm(name, step):
        key = (name, step)
        if key not in get:
            f = Path(texts) / name / f"step{step}_T1.00_texts.json.gz"
            get[key] = text_metrics(f, n) if f.exists() else None
        return get[key]
    base = tm("BASE15", 0)
    need = [("BASE15", 0)] + [(f"B1k{s}", st) for s in SEEDS for st in S] + [(f"{a}s{s}", st) for a in ARMS for s in SEEDS for st in S]
    miss = [k for k in need if tm(*k) is None]
    if miss:
        print(f"   manipulation check: missing texts {miss}")
        return
    for metric in ("distinct", "collision"):
        for a in ARMS:
            ones = np.ones(n)

            def gaps(w, clamp=None):
                out = {}
                for s in SEEDS:
                    ref = [base] + [tm(f"B1k{s}", st) for st in S]
                    pts = sorted((wmean(r["p1"], w), wmean(r[metric], w)) for r in ref)
                    px, py = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
                    g = []
                    for st in elig[s]:
                        r = tm(f"{a}s{s}", st)
                        x = wmean(r["p1"], w)
                        if not px.min() <= x <= px.max():
                            x = min(max(x, px.min()), px.max())
                            if clamp is not None:
                                clamp[s] = clamp.get(s, 0) + 1
                        g.append(wmean(r[metric], w) - float(np.interp(x, px, py)))
                    out[s] = float(np.mean(g))
                return out
            elig = {}
            for s in SEEDS:
                ref = [base] + [tm(f"B1k{s}", st) for st in S]
                lo, hi = min(wmean(r["p1"], ones) for r in ref), max(wmean(r["p1"], ones) for r in ref)
                elig[s] = [st for st in S if lo <= wmean(tm(f"{a}s{s}", st)["p1"], ones) <= hi]
            if any(len(v) < 2 for v in elig.values()):
                print(f"   {a} {metric}: insufficient overlap {elig}")
                continue
            full = gaps(ones)
            rng, clamp, boot = np.random.default_rng(0), {}, []
            for _ in range(draws):
                w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
                boot.append(np.mean(list(gaps(w, clamp).values())))
            lo, hi = np.percentile(boot, [5, 95])
            est = np.mean(list(full.values()))
            cl = max(clamp.get(s, 0) / (draws * len(elig[s])) for s in SEEDS)
            read = ""
            if metric == "distinct":
                read = " -> diversity increase shown" if lo > 0 else " -> an increase in the registered diversity metric has not been demonstrated"
            print(f"   {a} {metric} minus B1k at matched pass@1: {est:+.3f}, 90% [{lo:+.3f}, {hi:+.3f}]; per seed "
                  + ", ".join(f"{s}: {v:+.3f}" for s, v in full.items()) + f"; max clamped {100 * cl:.1f}%{read}")
    for a in ARMS:
        nn = np.mean([tm(f"{a}s{s}", 150)["none"].mean() for s in SEEDS])
        nb = np.mean([tm(f"B1k{s}", 150)["none"].mean() for s in SEEDS])
        print(f"   {a} no-parsed-box share at step 150: {100 * nn:.1f}% (B1k {100 * nb:.1f}%)")


def verdict(t, h, key):
    if "estimate" not in t:
        return t["verdict"]
    if not M.clamped_ok(t):
        return "not interpreted (more than 10% clamped)"
    if M.moves(t, h[key][1]):
        return "moves the curve (checkpoint-averaged gain detected)"
    if "estimate" in t and t["estimate"] <= -2 and t["ci95"][1] < 0 and all(v < 0 for v in t["per_seed"].values()) and h[key][1]:
        return "lowers the curve (checkpoint-averaged loss detected)"
    return "average equivalence" if M.equivalent(t) else "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--texts", default="/tmp/p32/texts")
    a = ap.parse_args()
    res = V.load(("results/worth-v15v-*/*/eval/step*_T*_eval.json", "results/worth-p32-oe-*/O[UB]s*/eval/step*_T*_eval.json"))
    print("Pre-registration 32 (outcome-based exploration on the matched-pass@1 curve; 1.5B, 1,024 tokens; vLLM evaluation)")
    need = V.B_NEED + [(f"{arm}s{sd}", st, 1.0) for arm in ARMS for sd in SEEDS for st in S]
    miss = V.missing(res, need)
    if miss:
        print(f"   incomplete evaluation; missing {miss}")
        return
    n = len(V.counts(res, "BASE15", 0))
    Benv = M.reference(V.traj(res, "B1k", S, ""), V.temps(res, "B1k", ""))
    B = M.reference(V.traj(res, "B1k", S, ""))
    tests = {f"{arm} vs B1k envelope": M.compare(V.arm_points(res, f"{arm}s", S), Benv, SEEDS, n, a.draws) for arm in ARMS}
    h = M.holm(tests)
    w = np.ones(n)
    for k, t in tests.items():
        arm = k.split()[0]
        print(M.show(k, t, verdict(t, h, k) + f" (Holm-adjusted p {h[k][0]:.3f})"))
        print(M.show(f"{arm} vs B1k trajectory (reported)", M.compare(V.arm_points(res, f"{arm}s", S), B, SEEDS, n, a.draws)))
        d1 = np.mean([A.met(V.counts(res, f"{arm}s{sd}", 150), w)[0] - A.met(V.counts(res, f"B1k{sd}", 150), w)[0] for sd in SEEDS])
        A.K = 32
        d32 = np.mean([A.met(V.counts(res, f"{arm}s{sd}", 150), w)[1] - A.met(V.counts(res, f"B1k{sd}", 150), w)[1] for sd in SEEDS])
        t32 = M.compare(V.arm_points(res, f"{arm}s", S), Benv, SEEDS, n, a.draws)
        A.K = 16
        read = (" -> excludes the +1.7 reference magnitude" if t32.get("ci90", [0, 9])[1] < 1.7 else " -> does not exclude +1.7") \
            if "estimate" in t32 else ""
        print(M.show(f"{arm} pass@32 vs B1k envelope (secondary)", t32, read.strip(" ->")))
        print(f"   {arm} step 150 minus B1k (raw): pass@1 {d1:+.2f}, pass@32 {d32:+.2f}")
        for sd in SEEDS:
            print(f"   {arm}s{sd} pass@16 minus B1k{sd} at step 150 by base count: {V.strata(res, f'{arm}s{sd}', 'BASE15', f'B1k{sd}')}")
        logs = []
        for sd in SEEDS:
            f = glob.glob(f"results/worth-p32-oe-*/{arm}s{sd}/baseline_train.json")
            if f:
                logs += [r for r in json.loads(open(f[0]).read()) if "oe" in r]
        if logs:
            oe = [r["oe"] for r in logs]
            dw = [x["distinct_answers_all_wrong_groups"] for x in oe if x["distinct_answers_all_wrong_groups"] is not None]
            pe = [r["probe"]["entropy"] for r in logs if isinstance(r.get("probe"), dict) and "entropy" in r["probe"]]
            print(f"   {arm} training ({len(oe)} steps over 3 seeds): nonzero bonuses {np.mean([x['bonus_nonzero'] for x in oe]):.1f}/128, "
                  f"positive {np.mean([x['positive_bonus'] for x in oe]):.1f}, mean bonus {np.mean([x['mean_bonus'] for x in oe]):+.4f}, "
                  f"all-correct groups {np.mean([x['groups_all_correct'] for x in oe]):.2f}/16 (skipped {np.mean([x['groups_all_correct_skipped'] for x in oe]):.2f}), "
                  f"distinct outcomes per group {np.mean([x['distinct_answers'] for x in oe]):.2f} (all-wrong groups {np.mean(dw) if dw else float('nan'):.2f}), "
                  f"no parsed box {np.mean([x['no_answer'] for x in oe]):.1f}/128"
                  + (f", probe entropy {np.mean(pe):.4f}" if pe else ""))
    print("Manipulation check (test texts; not multiplicity-adjusted)")
    diversity(a.texts, a.draws)


if __name__ == "__main__":
    main()
