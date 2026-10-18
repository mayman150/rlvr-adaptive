"""Pre-registration 9 analysis: asymmetry, noise and speed, one curve (bootstrap), speed at
equal cost. See PREREGISTRATION_9.md for the decision rules.

Usage: python analyze_v9.py [--draws 2000]
Training results: results/worth-v9-<ARM>-s<seed>-*/<ARM>/ ; evaluation results:
results/worth-v9e-<ARM>-s<seed>-*/eval/step<N>_eval.json (the base model's step0 comes
from the Dr. GRPO evaluation job of the same seed).
"""

import argparse
import glob
import json
import math
import re
import sys
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
sys.path.insert(0, str(HERE))
from analyze_v2 import window_stats  # noqa: E402

ARM_FILE = {"DR": "baseline", "PSR": "psr", "NSR": "nsr", "PK4": "passk", "O128": "optimal"}
N_SAMPLES, K_DIV = 32, 16


def latest(pattern):
    found = sorted(glob.glob(str(RES / pattern)))
    return Path(found[-1]) if found else None


def train_log(arm, seed, version="v9"):
    job = latest(f"worth-{version}-{arm}-s{seed}-*")
    if not job:
        return None, None
    f = job / arm / f"{ARM_FILE[arm]}_train.json"
    c = job / arm / "calibration.json"
    return (json.loads(f.read_text()) if f.exists() else None), (json.loads(c.read_text()) if c.exists() else None)


def eval_points(arm, seed, version="v9e"):
    """{step: per-problem correct counts out of 32} for a run, plus the base model at step 0."""
    pts = {}
    for name in (arm, "DR"):
        job = latest(f"worth-{version}-{name}-s{seed}-*")
        if not job:
            continue
        for f in (job / "eval").glob("step*_eval.json"):
            d = json.loads(f.read_text())
            step = int(re.findall(r"\d+", f.name)[0])
            if "counts" in d and (name == arm or step == 0):
                pts.setdefault(step, np.array(d["counts"]))
    return dict(sorted(pts.items()))


def per_problem_metrics(counts):
    c = np.asarray(counts)
    p1 = c / N_SAMPLES
    pk = np.array([1.0 - comb(N_SAMPLES - int(x), K_DIV) / comb(N_SAMPLES, K_DIV) for x in c])
    return p1, pk


def curve(points, weights):
    xs, ys, steps = [], [], []
    for step, counts in points.items():
        p1, pk = per_problem_metrics(counts)
        xs.append(100 * float(weights @ p1) / weights.sum())
        ys.append(100 * float(weights @ pk) / weights.sum())
        steps.append(step)
    return np.array(steps), np.array(xs), np.array(ys)


def matched_difference(r_pts, p_pts, weights):
    """Mean over R's checkpoints (after step 0) of R's pass@16 minus P's interpolated at R's pass@1."""
    _, px, py = curve(p_pts, weights)
    order = np.argsort(px)
    px, py = px[order], py[order]
    rs, rx, ry = curve(r_pts, weights)
    diffs = [y - np.interp(x, px, py) for s, x, y in zip(rs, rx, ry) if s != 0 and px[0] <= x <= px[-1]]
    return (float(np.mean(diffs)), float(np.sqrt(np.mean(np.square(diffs)))), len(diffs)) if len(diffs) >= 2 else (math.nan, math.nan, len(diffs))


def one_curve(r_arm, p_arm, seeds, draws, rng, version="v9e"):
    per_seed, boot = [], []
    data = [(eval_points(r_arm, s, version), eval_points(p_arm, s, version)) for s in seeds]
    data = [(r, p) for r, p in data if r and p]
    if not data:
        return None
    n_prob = len(next(iter(data[0][1].values())))
    ones = np.ones(n_prob)
    for r, p in data:
        per_seed.append(matched_difference(r, p, ones))
    for _ in range(draws):
        w = np.bincount(rng.integers(0, n_prob, n_prob), minlength=n_prob).astype(float)
        vals = [matched_difference(r, p, w)[0] for r, p in data]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            boot.append(np.mean(vals))
    est = np.nanmean([m for m, _, _ in per_seed])
    lo90, hi90 = np.percentile(boot, [5, 95])
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    if lo90 >= -2 and hi90 <= 2:
        verdict = "one curve"
    elif (lo95 > 0 or hi95 < 0) and abs(est) >= 2:
        verdict = "different"
    else:
        verdict = "inconclusive"
    return {"estimate": est, "ci90": (lo90, hi90), "ci95": (lo95, hi95), "per_seed": per_seed, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--train-version", default="v9")
    ap.add_argument("--eval-version", default="v9e")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    seeds = (1, 2)

    print("1. Asymmetry F0 / F1 (95% bootstrap over answers)")
    for s in seeds:
        _, cal = train_log("DR", s, args.train_version)
        if not cal or "norms_wrong" not in cal:
            print(f"   seed {s}: no calibration")
            continue
        n1, n0 = np.array(cal["norms_correct"]), np.array(cal["norms_wrong"])
        bs = [rng.choice(n0, len(n0)).mean() / rng.choice(n1, len(n1)).mean() for _ in range(args.draws)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"   seed {s}: {n0.mean() / n1.mean():.2f} [{lo:.2f}, {hi:.2f}] from {len(n1)} correct, {len(n0)} wrong -> "
              f"{'pass' if lo > 1.2 else 'fail'}")

    print("2. Noise and speed")
    for s in seeds:
        tp, _ = train_log("PSR", s, args.train_version)
        tn, _ = train_log("NSR", s, args.train_version)
        if not tp or not tn:
            continue
        sp = [x for x in tp if x["step"] < 50 and x.get("split")]
        sn = [x for x in tn if x["step"] < 50 and x.get("split")]
        ratio = window_stats(sn)["m2"] / window_stats(sp)["m2"]
        bs = [window_stats([sn[i] for i in rng.integers(0, len(sn), len(sn))])["m2"] /
              window_stats([sp[i] for i in rng.integers(0, len(sp), len(sp))])["m2"] for _ in range(500)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        start = np.mean([x["first_block_reward"] for x in tp[:10]])
        target = start + 0.5 * (np.mean([x["first_block_reward"] for x in tp[250:300]]) - start)

        def reach(t):
            fb = [x["first_block_reward"] for x in t]
            sm = [np.mean(fb[max(0, i - 9):i + 1]) for i in range(len(fb))]
            return next((i for i, v in enumerate(sm) if v >= target), 300)
        print(f"   seed {s}: noise NSR/PSR {ratio:.2f} [{lo:.2f}, {hi:.2f}] -> {'pass' if lo > 1 else 'fail'}; "
              f"steps to midpoint PSR {reach(tp)} NSR {reach(tn)} -> {'pass' if reach(tn) > reach(tp) else 'fail'}")

    print("3. One curve (pass@16 at matched pass@1; pooled over seeds; bootstrap over problems)")
    for r_arm, p_arm in (("NSR", "PSR"), ("PK4", "DR")):
        res = one_curve(r_arm, p_arm, seeds, args.draws, rng, args.eval_version)
        if not res:
            print(f"   {r_arm} vs {p_arm}: no data")
            continue
        ps = ", ".join(f"seed: {m:+.2f} (RMS {r:.2f}, {n} points)" for m, r, n in res["per_seed"])
        print(f"   {r_arm} vs {p_arm}: {res['estimate']:+.2f}, 90% [{res['ci90'][0]:+.2f}, {res['ci90'][1]:+.2f}], "
              f"95% [{res['ci95'][0]:+.2f}, {res['ci95'][1]:+.2f}] -> {res['verdict']}   ({ps})")

    print("4. Optimal rule minus Dr. GRPO, greedy on 3,382 problems at step 300 (paired, pooled)")
    diffs = []
    for s in seeds:
        g = {}
        for arm in ("O128", "DR"):
            job = latest(f"worth-{args.eval_version}-{arm}-s{s}-*")
            f = job / "eval" / "step300_eval.json" if job else None
            if f and f.exists() and "greedy" in json.loads(f.read_text()):
                g[arm] = np.array(json.loads(f.read_text())["greedy"]["correct"])
        if len(g) == 2:
            d = 100 * (g["O128"] - g["DR"])
            print(f"   seed {s}: {d.mean():+.2f} ({d.std(ddof=1) / math.sqrt(len(d)):.2f})")
            diffs.append(d)
        t, _ = train_log("O128", s, args.train_version)
        if t:
            print(f"   seed {s}: optimal rule generated {sum(x['answers'] for x in t)} answers (target 38,400), "
                  f"{sum(x['tokens'] for x in t):,} tokens")
    if diffs:
        d = np.concatenate(diffs)
        est, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
        lo, hi = est - 1.96 * se, est + 1.96 * se
        verdict = "faster" if est >= 1 and lo > 0 else "slower" if hi < 0 else "inconclusive"
        print(f"   pooled: {est:+.2f} [{lo:+.2f}, {hi:+.2f}] -> {verdict}")

    print("Formatting (mean over steps 250 to 299): boxed-answer rate / truncation rate")
    for arm in ARM_FILE:
        for s in seeds:
            t, _ = train_log(arm, s, args.train_version)
            if t and "boxed_rate" in t[0]:
                late = t[250:300]
                print(f"   {arm} s{s}: {100 * np.mean([x['boxed_rate'] for x in late]):.0f}% / "
                      f"{100 * np.mean([x['truncated_rate'] for x in late]):.0f}%")


if __name__ == "__main__":
    main()
