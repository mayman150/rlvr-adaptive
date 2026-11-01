"""Registered analysis of Pre-registration 39 (Stage 0). Written before any Stage 0 result was read.

Reads results/worth-p39-stage0-*/ (one job). Reports, as fixed in PREREGISTRATION_39.md:
- Harvest: problems (of the 109 c = 0 training problems) with >= 1 and >= 2 verified finished solutions by depth
  (first 64, 256, 1,024 samples per problem in generation order); solutions per problem; token lengths; finish rate.
  Writes the audit set (every problem with a verified finished solution: up to two, shortest first) to
  results_p26/p39_audit_harvest.json, and the same for the PEER pool (SmolLM3's verified finished solutions on the
  109 problems) to results_p26/p39_audit_peer.json. Gate H is applied after the audit (audit_gate.py reads the scores).
- Gate V: the base's pass@128 on the 200 hard_val rows (RE_BASE counts[208:]), also by source (first 100
  OlympiadBench, last 100 AIME).
- Large k: pass@1, 16, 64, 128 for each RE_* model on the 208 MATH-hard and the 200 hard_val problems, with 95%
  problem-bootstrap intervals (2,000 draws, seed 39), and paired differences from RE_B1k8; flagged if the 95%
  interval of the pass@128 difference excludes 0 (descriptive).
- Retention: for the 143 probed problems, SELF8 (steps 50, 150) against the SELF start and B1k8 (steps 50, 150)
  against the base; a problem declines if the Newcombe hybrid-score 95% interval for the difference lies below 0.
  Constraint trigger: >= 20% of problems declining by step 150 in either run.
- Throughput: generated tokens per second per evaluation.
Usage: python analyze_p39.py
"""
import glob
import gzip
import json
from math import comb, sqrt

import numpy as np

RES = sorted(glob.glob("results/worth-p39-stage0-*"))
assert len(RES) == 1, RES
R = RES[0]
Z = 1.959964


def evals(name):
    f = sorted(glob.glob(f"{R}/{name}/eval/step*_T1.00_eval.json"))
    return {int(x.split("step")[-1].split("_")[0]): json.load(open(x)) for x in f}


def texts(name, step):
    return json.load(gzip.open(f"{R}/{name}/eval/step{step}_T1.00_texts.json.gz", "rt"))


def pk(c, n, k):
    c = np.asarray(c)
    return np.array([1 - comb(n - x, k) / comb(n, k) if n - x >= k else 1.0 for x in range(n + 1)])[c]


def wilson(x, n):
    p = x / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, c - h, c + h


def newcombe(x1, n1, x2, n2):
    """95% interval for p2 - p1 (Newcombe's hybrid score method)."""
    p1, l1, u1 = wilson(x1, n1)
    p2, l2, u2 = wilson(x2, n2)
    d = p2 - p1
    return d, d - sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)


def harvest():
    c0 = json.load(open("data/p39_train_c0.json"))
    per = {i: [] for i in range(len(c0))}
    toks, fin, secs, gen = [], [], 0.0, 0
    for k in range(8):
        e = evals(f"HARVEST_c0_{k}")[0]
        t = texts(f"HARVEST_c0_{k}", 0)
        off = 14 * k
        secs += e.get("seconds", 0)
        for s in t["samples"]:
            per[off + s["problem"]].append(s)
            toks.append(s["tokens"])
            fin.append(s["finished"])
        gen += sum(x["tokens"] for x in t["samples"])
    assert all(len(v) == 1024 for v in per.values()), {i: len(v) for i, v in per.items() if len(v) != 1024}
    print(f"Harvest: {len(c0)} problems x 1,024 samples at 2,048 tokens; mean tokens {np.mean(toks):.0f}, finished {100 * np.mean(fin):.1f}%; "
          f"{gen / max(secs, 1):.0f} generated tokens/s per GPU (summed over chunks)")
    good = {i: [s for s in v if s["reward"] and s["finished"]] for i, v in per.items()}
    for d in (64, 256, 1024):
        g = [sum(1 for s in per[i][:d] if s["reward"] and s["finished"]) for i in per]
        print(f"   depth {d:4d}: problems with >= 1 verified finished solution {sum(x >= 1 for x in g)}, >= 2: {sum(x >= 2 for x in g)}")
    n_sol = [len(v) for v in good.values() if v]
    if n_sol:
        print(f"   solutions per solved problem: median {np.median(n_sol):.0f}, mean {np.mean(n_sol):.1f}, max {max(n_sol)}; "
              f"verified finished lengths: median {np.median([s['tokens'] for v in good.values() for s in v]):.0f} tokens")
    rates = [len(good[i]) / 1024 for i in per]
    print(f"   success rate on solved problems: median {np.median([r for r in rates if r > 0]) if any(rates) else 0:.4f}")
    audit = [{"source": "HARVEST", "problem": c0[i]["question"], "answer": c0[i]["answer"], "index": i,
              "solutions": [s["text"] for s in sorted(v, key=lambda s: s["tokens"])[:2]], "verified_count": len(v)}
             for i, v in good.items() if v]
    json.dump(audit, open("results_p26/p39_audit_harvest.json", "w"), indent=0)
    print(f"   audit set: {len(audit)} problems -> results_p26/p39_audit_harvest.json (Gate H needs >= 44 audited-valid)")
    return c0


def peer(c0):
    train = json.load(open("data/train.json"))
    idx = {q["question"]: i for i, q in enumerate(train)}
    want = {idx[q["question"]]: j for j, q in enumerate(c0)}
    found = {j: [] for j in range(len(c0))}
    toks, secs, gen = [], 0.0, 0
    for k in range(8):
        e = evals(f"SMOLtrain64_2k_{k}")[0]
        t = texts(f"SMOLtrain64_2k_{k}", 0)
        secs += e.get("seconds", 0)
        gen += sum(s["tokens"] for s in t["samples"])
        for s in t["samples"]:
            i = 490 * k + s["problem"]
            toks.append(s["tokens"])
            if i in want and s["reward"] and s["finished"]:
                found[want[i]].append(s)
    solved = [j for j, v in found.items() if v]
    print(f"\nPEER (SmolLM3-3B-Base census, 64 samples at 2,048 tokens on all training problems; {gen / max(secs, 1):.0f} tokens/s): "
          f"{len(solved)} of the {len(c0)} c = 0 problems have a verified finished SmolLM3 solution; mean tokens {np.mean(toks):.0f}")
    audit = [{"source": "PEER", "problem": c0[j]["question"], "answer": c0[j]["answer"], "index": j,
              "solutions": [s["text"] for s in sorted(found[j], key=lambda s: s["tokens"])[:2]], "verified_count": len(found[j])} for j in solved]
    json.dump(audit, open("results_p26/p39_audit_peer.json", "w"), indent=0)


def large_k():
    rng = np.random.default_rng(39)
    models = ["RE_BASE", "RE_B1k8", "RE_B1k9", "RE_LH1", "RE_SELF8", "RE_TEACHF8", "RE_CWS8", "RE_OUs8", "RE_PSR1k8", "RE_S32"]
    C = {}
    for m in models:
        e = evals(m)
        if not e:
            print(f"   {m}: missing")
            continue
        (st, d), = e.items()
        C[m] = (np.array(d["counts"]), d["n"], d)
    base = C["RE_BASE"]
    v = pk(base[0][208:], base[1], 128)
    print(f"\nGate V: base pass@128 on hard_val {100 * v.mean():.1f}% (OlympiadBench {100 * v[:100].mean():.1f}%, AIME {100 * v[100:].mean():.1f}%) "
          f"-> {'pass' if 0.10 <= v.mean() <= 0.80 else 'FAIL'} (10-80%)")
    sets = {"MATH-hard (208)": slice(0, 208), "hard_val (200)": slice(208, 408), "hard_val OlympiadBench (100)": slice(208, 308),
            "hard_val AIME (100)": slice(308, 408)}
    for lab, sl in sets.items():
        P = len(range(408)[sl])
        draws = [rng.integers(0, P, P) for _ in range(2000)]
        print(f"\n{lab}: pass@1 / pass@16 / pass@64 / pass@128 (95% CI of pass@128); difference from B1k8 at pass@128 [95%]")
        ref = pk(C["RE_B1k8"][0][sl], 256, 128)
        for m, (c, n, d) in C.items():
            vals = {k: pk(c[sl], n, k) for k in (1, 16, 64, 128)}
            bs = np.array([vals[128][i].mean() for i in draws])
            diff = vals[128] - ref
            bd = np.array([diff[i].mean() for i in draws])
            lo, hi = np.percentile(bd, [2.5, 97.5])
            flag = " *" if (lo > 0 or hi < 0) and m != "RE_B1k8" else ""
            print(f"   {m:11s} {100 * vals[1].mean():5.1f} / {100 * vals[16].mean():5.1f} / {100 * vals[64].mean():5.1f} / {100 * vals[128].mean():5.1f} "
                  f"[{100 * np.percentile(bs, 2.5):.1f}, {100 * np.percentile(bs, 97.5):.1f}]; vs B1k8 {100 * diff.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]{flag}"
                  f"; truncated {100 * d.get('truncated_rate', np.nan):.0f}%")


def retention():
    def counts(name, step):
        e = evals(name)[step]
        return np.array(e["counts"]), e["n"]
    b, nb = counts("PR_BASE", 0)
    s0, ns = counts("PR_SELFstart", 0)
    base64 = np.array(json.load(open("data/p39_stage0_counts_v2.json")))
    train = json.load(open("data/train.json"))
    idx = {q["question"]: i for i, q in enumerate(train)}
    c12 = json.load(open("data/p39_train_c12.json"))
    bc = np.array([base64[idx[q["question"]]] for q in c12])
    print(f"\nRetention probes ({len(c12)} training problems with 1-2 of 64 base successes; 256 samples at 1,024 tokens)")
    print(f"   base success rate {100 * b.mean() / nb:.2f}%; SELF start {100 * s0.mean() / ns:.2f}%")
    trig = False
    for run, start, sn in (("PR_SELF8", s0, ns), ("PR_B1k8", b, nb)):
        for st in (50, 150):
            c, n = counts(run, st)
            res = [newcombe(int(x1), sn, int(x2), n) for x1, x2 in zip(start, c)]
            dec = np.array([u < 0 for _, _, u in res])
            inc = np.array([l > 0 for _, l, _ in res])
            print(f"   {run} step {st}: mean rate {100 * c.mean() / n:.2f}%; change {100 * np.mean([d for d, _, _ in res]):+.2f} points; "
                  f"declining {100 * dec.mean():.1f}% (c=1: {100 * dec[bc == 1].mean():.1f}%, c=2: {100 * dec[bc == 2].mean():.1f}%), rising {100 * inc.mean():.1f}%")
            if st == 150 and dec.mean() >= 0.20:
                trig = True
    print(f"   constraint trigger (>= 20% declining by step 150 in either run): {'YES' if trig else 'no'}")


def throughput():
    print("\nThroughput (generated tokens per second, per evaluation)")
    for f in sorted(glob.glob(f"{R}/*/eval/step*_T1.00_eval.json")):
        d = json.load(open(f))
        tok = d.get("mean_length_tokens", 0) * d["n"] * d["problems"]
        print(f"   {f.split('/')[-3]:22s} {d['n']:5d} x {d['problems']:4d}: {d.get('seconds', 0) / 3600:.2f} h, {tok / max(d.get('seconds', 1), 1):.0f} tok/s")


if __name__ == "__main__":
    print(f"Pre-registration 39, Stage 0 ({R})")
    c0 = harvest()
    peer(c0)
    large_k()
    retention()
    throughput()
