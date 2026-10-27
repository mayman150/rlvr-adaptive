"""Registered analysis of Pre-registration 20 (Qwen2.5-Math-7B, 2,048 tokens; vLLM evaluation), on the 511 test problems
and, separately, on the frozen hard cohort (data/hard7_cohort.json).
Results: results/worth-v20v-*/<NAME>/eval (NAME = <ARM>s<seed>[hard], BASE7hard) and the vLLM test job
results/worth-v20v-testw6-*/ (B7s8 steps 25-150, B7s9 step 150, BASE7v).
Usage: python analyze_v20.py [--draws 2000]
"""

import argparse
import glob
import json
import re

import numpy as np

import analyze_v12 as A
import matched as M

SEEDS = (8, 9)
ARMS = {"B7": (25, 50, 100, 150), "PSR7": (25, 50, 100, 150), "NSR7": (50, 100, 150), "AC7": (50, 100, 150)}


def load():
    out = {}
    for pat in ("results/worth-v20v-testw6-*/*/eval/step*_T*_eval.json", "results/worth-v20v-[A-Z]*-s0-*/*/eval/step*_T*_eval.json"):
        for f in glob.glob(pat):
            name = f.split("/")[-3].replace("BASE7v", "BASE7")
            step, t = re.search(r"step(\d+)_T([\d.]+)_eval", f).groups()
            out.setdefault((name, int(step), round(float(t), 2)), json.loads(open(f).read()))
    return out


def c(res, name, step, t=1.0):
    d = res.get((name, step, t))
    return None if d is None else np.array(d["counts"])


def run(res, suffix, draws, label):
    print(f"{label}")
    base = "BASE7" + suffix
    need = [(base, 0, 1.0)] + [(f"{a}s{s}{suffix}", st, 1.0) for a, steps in ARMS.items() for s in SEEDS for st in steps] + \
           [(f"B7s{s}{suffix}", 150, t) for s in SEEDS for t in (1.15, 1.3)]
    miss = [k for k in need if k not in res]
    if miss:
        print(f"   incomplete evaluation; missing {miss}")
        return
    n = len(c(res, base, 0))

    def traj(arm, steps):
        return lambda s: [(c(res, base, 0), 0)] + [(c(res, f"{arm}s{s}{suffix}", st), st) for st in steps]
    temps = lambda s: [(c(res, f"B7s{s}{suffix}", 150, t), t) for t in (1.0, 1.15, 1.3)]
    pts = lambda arm, steps: (lambda s: [(st, c(res, f"{arm}s{s}{suffix}", st)) for st in steps])
    B, Benv, PSR = M.reference(traj("B7", ARMS["B7"])), M.reference(traj("B7", ARMS["B7"]), temps), M.reference(traj("PSR7", ARMS["PSR7"]))
    S3 = (50, 100, 150)
    tests = {"NSR7 vs PSR7 trajectory": M.compare(pts("NSR7", S3), PSR, SEEDS, n, draws),
             "PSR7 vs B7 trajectory": M.compare(pts("PSR7", S3), B, SEEDS, n, draws),
             "NSR7 vs B7 trajectory": M.compare(pts("NSR7", S3), B, SEEDS, n, draws),
             "AC7 vs B7 envelope": M.compare(pts("AC7", S3), Benv, SEEDS, n, draws)}
    h = M.holm(tests)
    for k, t in tests.items():
        if "estimate" not in t:
            v = t["verdict"]
        elif not M.clamped_ok(t):
            v = "not interpreted (more than 10% clamped)"
        elif k.startswith("AC7"):
            v = "moves the curve" if M.moves(t, h[k][1]) else "replicates the null (average equivalence)" if M.equivalent(t) else "inconclusive"
        else:
            v = "average equivalence" if M.equivalent(t) else "departure" if M.departure(t, h[k][1]) else "inconclusive"
        print(M.show(k, t, v + f" (Holm-adjusted p {h[k][0]:.3f})"))
    print(M.show("AC7 vs B7 trajectory (reported)", M.compare(pts("AC7", S3), B, SEEDS, n, draws)))
    w = np.ones(n)
    for a, steps in ARMS.items():
        print(f"   {a}: " + " | ".join(f"s{s} " + ", ".join(f"{st}: {A.met(c(res, f'{a}s{s}{suffix}', st), w)[0]:.1f}/{A.met(c(res, f'{a}s{s}{suffix}', st), w)[1]:.1f}"
                                                   for st in steps) for s in SEEDS)
              + f"; trunc@150 " + ", ".join(f"{100 * res[(f'{a}s{s}{suffix}', 150, 1.0)]['truncated_rate']:.1f}%" for s in SEEDS))
    return n


def rule3(res):
    """Frozen frontier-model forecast (FORECAST_20.json) against B7's observed pass@16 at its observed pass@1, steps 25-150.
    Points outside the forecast grid's pass@1 range are unscorable. Strata are the frozen base counts' strata."""
    fc = json.loads(open("FORECAST_20.json").read())
    grid = np.array([r["pass@1"] for r in fc["grid"]])
    base = np.array(json.loads(open(fc["base"]).read())["counts"])
    strata = {"c=0": base == 0, "c=1-4": (base >= 1) & (base <= 4), "c=5-16": (base >= 5) & (base <= 16), "c>16": base > 16}
    err = {m: [] for m in ("law", "uniform", "affine")}
    rows, unscorable = [], []
    for s in SEEDS:
        for st in ARMS["B7"]:
            cc = c(res, f"B7s{s}", st)
            if cc is None:
                print(f"   rule 3: missing B7s{s} step {st}")
                return
            p1, p16 = A.met(cc, np.ones(len(cc)))
            if not grid.min() <= p1 / 100 <= grid.max():
                unscorable.append((s, st, p1))
                continue
            for m in err:
                err[m].append(p16 - 100 * float(np.interp(p1 / 100, grid, [r[m]["pass@16"] for r in fc["grid"]])))
            strat = {k: A.met(cc, mask.astype(float))[1] - 100 * float(np.interp(p1 / 100, grid, [r["law"][k] for r in fc["grid"]]))
                     for k, mask in strata.items() if mask.any()}
            rows.append((s, st, p1, err["law"][-1], strat))
    if not rows:
        print("   rule 3: no scorable point")
        return
    rmse = {m: float(np.sqrt(np.mean(np.square(e)))) for m, e in err.items()}
    cat = "supported" if rmse["law"] <= 2 else "partial" if rmse["law"] <= 3 else "falsified"
    print(f"Rule 3 (frontier model): RMSE over {len(rows)} scorable B7 seed-steps: law {rmse['law']:.2f} -> {cat}; "
          f"comparators uniform shift {rmse['uniform']:.2f}, affine {rmse['affine']:.2f}; unscorable {unscorable}")
    for s, st, p1, e, strat in rows:
        print(f"   s{s} step {st} (pass@1 {p1:.1f}): law error {e:+.2f}; by stratum " + ", ".join(f"{k} {v:+.1f}" for k, v in strat.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    res = load()
    run(res, "", args.draws, "Pre-registration 20, the 511 test problems (2,048 tokens, vLLM)")
    rule3(res)
    run(res, "hard", args.draws, "Pre-registration 20, the hard cohort (253 problems; reported separately)")
    g = {a: [res.get((f"{a}s{s}", 150, 1.0), {}).get("greedy", {}).get("accuracy") for s in SEEDS] for a in ARMS}
    print("Greedy test accuracy at step 150: " + "; ".join(f"{a} " + ", ".join("-" if x is None else f"{100 * x:.1f}" for x in v) for a, v in g.items()))


if __name__ == "__main__":
    main()
