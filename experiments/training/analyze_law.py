"""Pre-registration 17 analysis. Usage: python analyze_law.py [--deep DIR] [--fit-only]
--deep: directory with deep-sampling results (discovery_shard*.json with counts of 256 on deep_base_309.json).
"""

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_v12 as A
import analyze_v14 as V
import analyze_v9 as O
import law

HERE = Path(__file__).resolve().parent


def strata(base):
    return {"never": base == 0, "rare": (base >= 1) & (base <= 4), "medium": (base >= 5) & (base <= 16)}


def deep_extra(path, n_q):
    rows = []
    for f in sorted(Path(path).rglob("discovery_shard*.json")):
        rows += json.loads(f.read_text())
    sub = json.loads((HERE / "data" / "deep_base_309.json").read_text())
    by_q = {r["question"]: r for r in rows}
    c, t = np.zeros(n_q), np.zeros(n_q)
    for q in sub:
        r = by_q.get(q["question"])
        if r is not None:
            c[q["eval_index"]], t[q["eval_index"]] = r["c"], r["n"]
    print(f"deep samples: {int((t > 0).sum())} of {len(sub)} questions, {int(c.sum())} successes in {int(t.sum())} draws")
    return c, t


def load_prefix(prefix, key_of):
    """Counts from evaluation jobs results/<prefix>*/<ARM>/eval/step<N>_T<t>_eval.json, keyed by key_of(job, arm, step, t)."""
    import re
    out = {}
    for job in sorted(A.RES.glob(f"{prefix}*"), key=lambda j: j.name[-15:]):
        for arm_dir in job.iterdir():
            for f in (arm_dir / "eval").glob("step*_T*_eval.json") if (arm_dir / "eval").exists() else []:
                step, t = re.match(r"step(\d+)_T([\d.]+)_eval", f.name).groups()
                d = json.loads(f.read_text())
                if "counts" in d and d.get("n") == A.N:
                    out[key_of(job.name, arm_dir.name, int(step), round(float(t), 2))] = np.array(d["counts"])
    return out


def seed_of(job):
    import re
    return int(re.search(r"-s(\d+)-", job).group(1))


def report(name, rows):
    if not rows:
        print(f"{name}: no runs available")
        return
    r16 = law.rmse(rows)
    verdict = "supported" if r16 <= 2 else "partial" if r16 <= 3 else "falsified"
    dll = sum(r["ll_law"] - r["ll_aff"] for r in rows)
    wins = sum(r["ll_law"] > r["ll_aff"] for r in rows)
    print(f"{name}: {len(rows)} checkpoints; pass@16 RMSE law {r16:.2f} (uniform {law.rmse(rows, 'obs@16', 'uni@16'):.2f}, "
          f"affine {law.rmse(rows, 'obs@16', 'aff@16'):.2f}) -> {verdict}; log-lik law - affine {dll:+.1f} (wins {wins}/{len(rows)})")
    for k in (4, 32):
        print(f"   pass@{k} RMSE law {law.rmse(rows, f'obs@{k}', f'law@{k}'):.2f}")
    for s in ("never", "rare", "medium"):
        if f"{s}:obs@16" in rows[0]:
            print(f"   {s}: mean obs - law pass@16 {np.mean([r[f'{s}:obs@16'] - r[f'{s}:law@16'] for r in rows]):+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", default=None)
    ap.add_argument("--fit-only", action="store_true")
    args = ap.parse_args()
    r = V.load()
    base = r[("BASE", 0, 0, 1.0)]
    n = A.N
    prior = law.fit_prior(base, n)
    th32 = law.initial_theta(base, n, prior)
    fit_runs = [r[("B", s, st, 1.0)] for s in A.SEEDS for st in A.STEPS]
    tau32 = law.fit_tau(th32, fit_runs, n)
    beta = law.fit_beta(th32, fit_runs, n)
    print(f"prior Beta({prior[0]:.3f}, {prior[1]:.3f}); tau_32 = {tau32:+.2f}; affine beta = {beta:.2f}")
    if args.deep:
        extra = deep_extra(args.deep, len(base))
        thd = law.initial_theta(base, n, prior, extra)
        taud = law.fit_tau(thd, fit_runs, n)
        rng, draws = np.random.default_rng(0), []
        for _ in range(200):
            idx = rng.integers(0, len(base), len(base))
            draws.append(law.fit_tau(thd[idx], [c[idx] for c in fit_runs], n))
        print(f"tau_deep = {taud:+.2f}, 90% [{np.percentile(draws, 5):+.2f}, {np.percentile(draws, 95):+.2f}]")
    report("fit (B, in-sample)", law.evaluate(th32, tau32, beta, [(f"B s{s} {st}", r[("B", s, st, 1.0)]) for s in A.SEEDS for st in A.STEPS], n, strata(base)))
    if args.fit_only:
        return
    t1 = [(f"{a} s{s} {st}", r[(a, s, st, 1.0)]) for a in ("CM", "M", "U", "D32", "U32", "C05", "G32", "T32", "BBG", "MXR", "WRF")
          for s in A.SEEDS for st in A.STEPS if (a, s, st, 1.0) in r]
    report("T1 other methods (Qwen)", law.evaluate(th32, tau32, beta, t1, n, strata(base)))
    t2 = []
    for s in A.SEEDS:
        if ("S32", s, 0, 1.0) in r:
            c0 = r[("S32", s, 0, 1.0)]
            th0 = law.initial_theta(c0, n, law.fit_prior(c0, n))
            t2 += law.evaluate(th0, tau32, beta, [(f"S32 s{s} {st}", r[("S32", s, st, 1.0)]) for st in A.STEPS if ("S32", s, st, 1.0) in r], n, strata(base))
    report("T2 warm start", t2)
    # T3: evaluations of the 450-step runs, results/worth-v17Le-*/<B13|B14|B15>/eval (checkpoints beyond step 150)
    long = load_prefix("worth-v17Le-", lambda job, arm, st, t: (arm, st, t))
    t3 = [(f"{a} {st}", c) for (a, st, t), c in sorted(long.items()) if t == 1.0 and st > 150]
    report("T3 longer training", law.evaluate(th32, tau32, beta, t3, n, strata(base)) if t3 else [])
    # T5: 1,024-token evaluations (Check 16), from the 1,024-token base counts
    t5r = load_prefix("worth-v16t-", lambda job, arm, st, t: (arm, seed_of(job), st, t))
    b1024 = next((c for (a, s, st, t), c in t5r.items() if a == "BASE"), None)
    if b1024 is not None:
        th5 = law.initial_theta(b1024, n, law.fit_prior(b1024, n))
        t5 = [(f"{a}@1024 s{s} {st}", c) for (a, s, st, t), c in sorted(t5r.items()) if a == "B" and t == 1.0]
        report("T5 1,024 tokens", law.evaluate(th5, tau32, beta, t5, n, strata(b1024)))
    t4 = []
    for arm in ("DR", "PSR", "NSR", "PK4", "O128"):
        for s in (1, 2):
            pts = O.eval_points(arm, s)
            if 0 not in pts:
                continue
            ob = pts[0]
            th_o = law.initial_theta(ob, n, law.fit_prior(ob, n))
            t4 += law.evaluate(th_o, tau32, beta, [(f"OLMo {arm} s{s} {st}", c) for st, c in pts.items() if st > 0], n, strata(ob))
    report("T4 OLMo (tau from Qwen)", t4)


if __name__ == "__main__":
    main()
