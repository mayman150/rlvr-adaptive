"""Pre-registration 12 analysis (primary runs, seeds 8-10). See PREREGISTRATION_12.md.

Usage: python analyze_v12.py --branch M|S --candidate CM|EF --audits M,U|A,E [--draws 2000]
Results: results/worth-v12e-E{1,2}-s<seed>-*/<ARM>/eval/step<N>_T<t>_eval.json and .../BASE/eval/step0_T1.00_eval.json.
"""

import argparse
import json
import math
import re
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SEEDS = (8, 9, 10)
N, K = 32, 16
STEPS = (50, 100, 150)


def load():
    out = {}
    for job in sorted(RES.glob("worth-v12e-E*-s*-*"), key=lambda j: j.name[-15:]):
        seed = int(re.match(r"worth-v12e-E\w*-s(\d+)-", job.name).group(1))
        for arm_dir in job.iterdir():
            for f in (arm_dir / "eval").glob("step*_T*_eval.json") if (arm_dir / "eval").exists() else []:
                step, t = re.match(r"step(\d+)_T([\d.]+)_eval", f.name).groups()
                d = json.loads(f.read_text())
                if "counts" in d and d.get("n") == N:
                    key = ("BASE", 0, 0, 1.0) if arm_dir.name == "BASE" else (arm_dir.name, seed, int(step), round(float(t), 2))
                    out[key] = np.array(d["counts"])
    return out


def met(c, w):
    p1 = c / N
    pk = np.array([1 - comb(N - int(x), K) / comb(N, K) for x in c])
    return 100 * float(w @ p1) / w.sum(), 100 * float(w @ pk) / w.sum()


def curve(points, w):
    """points: list of (counts, tiebreak); returns sorted pass@1 and pass@16 arrays."""
    m = sorted((*met(c, w), tb) for c, tb in points)
    return np.array([x[0] for x in m]), np.array([x[1] for x in m])


def reference(res, seed, w, envelope):
    traj = curve([(res[("BASE", 0, 0, 1.0)], 0)] + [(res[("B", seed, s, 1.0)], s) for s in STEPS], w)
    temp = curve([(res[("B", seed, 150, t)], t) for t in (1.0, 1.15, 1.3)], w) if envelope else None

    def value(x):
        vals = []
        for px, py in [traj] + ([temp] if temp else []):
            if px.min() <= x <= px.max():
                vals.append(float(np.interp(x, px, py)))
        return max(vals) if vals else None
    lo = min(traj[0].min(), temp[0].min()) if temp else traj[0].min()
    hi = max(traj[0].max(), temp[0].max()) if temp else traj[0].max()
    return value, lo, hi


def arm_points(res, arm, seed, w):
    return [(s, *met(res[(arm, seed, s, 1.0)], w)) for s in STEPS]


def differences(res, arm, ref_fn, w, qualifying, seeds, stats=None):
    out = {}
    for seed in seeds:
        value, _, _ = ref_fn(res, seed, w)
        pts = {s: (x, y) for s, x, y in arm_points(res, arm, seed, w)}
        out[seed] = []
        for s in qualifying[seed]:
            v = value(pts[s][0])
            # a resampled point outside the reference ranges takes the nearest defined value
            if v is None:
                if stats is not None:
                    stats["clamped"] = stats.get("clamped", 0) + 1
                v = value(min(max(pts[s][0], ref_fn(res, seed, w)[1]), ref_fn(res, seed, w)[2]))
            out[seed].append(pts[s][1] - v)
            if stats is not None:
                stats["points"] = stats.get("points", 0) + 1
    return out


def compare(res, arm, ref_fn, seeds, draws, rng, mask=None, stats=None):
    n = len(res[("BASE", 0, 0, 1.0)])
    ones = np.ones(n) if mask is None else mask.astype(float)
    qualifying = {}
    for seed in seeds:
        _, lo, hi = ref_fn(res, seed, ones)
        qualifying[seed] = [s for s, x, _ in arm_points(res, arm, seed, ones) if lo <= x <= hi]
    if any(len(q) < 2 for q in qualifying.values()):
        return {"verdict": "insufficient overlap", "qualifying": qualifying}
    full = differences(res, arm, ref_fn, ones, qualifying, seeds)
    per_seed = {s: float(np.mean(v)) for s, v in full.items()}
    est = float(np.mean(list(per_seed.values())))
    boot, bp = [], []
    for _ in range(draws):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float) * ones
        d = differences(res, arm, ref_fn, w, qualifying, seeds, stats)
        boot.append(np.mean([np.mean(v) for v in d.values()]))
        bp.append(np.concatenate([d[s] for s in seeds]))
    boot = np.array(boot)
    flat, bp = np.concatenate([full[s] for s in seeds]), np.array(bp)
    se = bp.std(axis=0, ddof=1)
    q = np.percentile(np.max(np.abs(bp - flat) / np.where(se > 0, se, np.inf), axis=1), 90)
    return {"estimate": est, "per_seed": per_seed, "ci90": np.percentile(boot, [5, 95]), "ci95": np.percentile(boot, [2.5, 97.5]),
            "p_two_sided": float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))),
            "points": [((s, st), x, (x - q * e, x + q * e)) for (s, st), x, e in
                       zip([(s, st) for s in seeds for st in qualifying[s]], flat, se)]}


def three_way(r):
    if "estimate" not in r:
        return r["verdict"]
    lo90, hi90 = r["ci90"]
    lo95, hi95 = r["ci95"]
    if r["estimate"] >= 2 and lo95 > 0 and all(v > 0 for v in r["per_seed"].values()):
        return "moves the curve"
    if lo90 >= -2 and hi90 <= 2:
        return "average agreement"
    if r["estimate"] <= -2 and hi95 < 0:
        return "worse"
    return "inconclusive"


def show(name, r, verdict):
    if "estimate" not in r:
        print(f"   {name}: {verdict} {r.get('qualifying', '')}")
        return
    print(f"   {name}: {r['estimate']:+.2f}, 90% [{r['ci90'][0]:+.2f}, {r['ci90'][1]:+.2f}], 95% [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}], "
          f"per seed " + ", ".join(f"{s}: {v:+.2f}" for s, v in r["per_seed"].items()) + f" -> {verdict}")
    print("      band: " + "; ".join(f"s{s} step {st}: {x:+.1f} [{lo:+.1f}, {hi:+.1f}]" for (s, st), x, (lo, hi) in r["points"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True, choices=("M", "S"))
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--audits", required=True)
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    res = load()
    arms = ["B", args.candidate] + args.audits.split(",")
    need = [("BASE", 0, 0, 1.0)] + [(a, s, st, 1.0) for a in arms for s in SEEDS for st in STEPS] + \
           [("B", s, 150, t) for s in SEEDS for t in (1.15, 1.3)]
    missing = [k for k in need if k not in res]
    if missing:
        print(f"incomplete evaluation; missing {missing}")
        return
    n = len(res[("BASE", 0, 0, 1.0)])
    envelope = lambda r, s, w: reference(r, s, w, True)
    trajectory = lambda r, s, w: reference(r, s, w, False)

    print(f"Primary: {args.candidate} against B's envelope ({n} problems, seeds {SEEDS})")
    prim = compare(res, args.candidate, envelope, SEEDS, args.draws, rng)
    v = three_way(prim)
    show(args.candidate, prim, v)
    probs = json.loads((HERE / "data" / "eval_frontier_512.json").read_text())
    old = {q["question"] for q in json.loads((HERE / "data" / "eval_frontier_256.json").read_text())}
    untouched = np.array([q["question"] not in old for q in probs])
    conf = compare(res, args.candidate, envelope, SEEDS, args.draws, rng, mask=untouched)
    confirmed = "estimate" in conf and np.sign(conf["estimate"]) == np.sign(prim.get("estimate", 0)) and (conf["ci90"][0] > 0 or conf["ci90"][1] < 0)
    show(f"{args.candidate} (untouched {int(untouched.sum())} problems)", conf, "confirmed" if confirmed else "not confirmed on untouched problems")
    print(f"   PRIMARY VERDICT: {v}" + ("" if v != "moves the curve" else (" (confirmed)" if confirmed else " (not confirmed on untouched problems)")))

    print("Equal cost: step-150 pass@1 of the candidate minus B (paired over problems, pooled over seeds)")
    d = np.mean([res[(args.candidate, s, 150, 1.0)] / N - res[("B", s, 150, 1.0)] / N for s in SEEDS], axis=0) * 100
    boot = [d[rng.integers(0, n, n)].mean() for _ in range(args.draws)]
    lo, hi = np.percentile(boot, [5, 95])
    print(f"   {d.mean():+.2f}, 90% [{lo:+.2f}, {hi:+.2f}] -> " + ("same accuracy at equal answers" if lo >= -1 and hi <= 1 else "not shown"))

    print("Audit arms against B's trajectory (Holm over the directional tests)")
    rs = {a: compare(res, a, trajectory, SEEDS, args.draws, rng) for a in args.audits.split(",")}
    ps = sorted((r.get("p_two_sided", 1.0), a) for a, r in rs.items())
    m = len(ps)
    holm_ok, stop = {}, False
    for i, (p, a) in enumerate(ps):
        ok = not stop and p <= 0.05 / (m - i)
        stop = stop or not ok
        holm_ok[a] = ok
    for a, r in rs.items():
        verdict = three_way(r)
        if verdict in ("moves the curve", "worse") and not holm_ok[a]:
            verdict += " (not significant after Holm)"
        show(a, r, verdict)
    if args.branch == "M" and "M" in rs:
        print("Coverage weighting: CM against M's trajectory")
        m_traj = lambda r, s, w: (lambda pts: ((lambda px, py: (lambda x: float(np.interp(x, px, py)) if px.min() <= x <= px.max() else None))(*pts),
                                               pts[0].min(), pts[0].max()))(curve([(r[("BASE", 0, 0, 1.0)], 0)] + [(r[("M", s, st, 1.0)], st) for st in STEPS], w))
        r = compare(res, args.candidate, m_traj, SEEDS, args.draws, rng)
        ok = "estimate" in r and r["estimate"] >= 1 and r["ci95"][0] > 0
        show("CM vs M", r, "coverage weighting matters" if ok else "not shown")


if __name__ == "__main__":
    main()
