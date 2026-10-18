"""Pre-registration 10 analysis: noise-only intervention (PSR+N), learning-rate arms, average
matched-accuracy equivalence, F0 / F1 over training, Adam statistics. See
PREREGISTRATION_10.md for the decision rules.

Usage: python analyze_v10.py [--draws 2000]
Runs: results/worth-v10-*-s<seed>-*/<ARM>/ (four-lane jobs hold PSR, NSR, PSRN, NSR2; PSR2 runs
alone), and the pinned Pre-registration 5 runs for PSR and NSR in seeds 3 and 4.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_v3 import predicted_noise  # noqa: E402
from analyze_v9 import RES, curve  # noqa: E402

ARM_FILE = {"PSR": "psr", "NSR": "nsr", "PSRN": "psrnoise", "NSR2": "nsr", "PSR2": "psr"}
PINNED = {("PSR", 3): "worth-v5-PSR-s3-20261017-164938", ("NSR", 3): "worth-v5-NSR-s3-20261017-164942",
          ("PSR", 4): "worth-v5-PSR-s4-20261018-034043", ("NSR", 4): "worth-v5-NSR-s4-20261017-164950"}
REF_STEPS = {0, 25, 50, 75, 100, 150}
RUN_STEPS = {25, 50, 75, 100, 150}
FRESH = (5, 6, 7)
WINDOW = range(25, 50)
MARGIN = 1 / 3


def load(folder, name):
    get = lambda f: json.loads((folder / f).read_text()) if (folder / f).exists() else None
    passk = get(f"{name}_passk.json") or []
    return {"train": get(f"{name}_train.json") or [], "calib": get("calibration.json"),
            "points": {p["step"]: np.array(p["counts"]) for p in passk if p.get("n") == 32},
            "remeasure": get(f"{name}_remeasure.json") or [], "name": name, "job": folder.parent.name}


def run(arm, seed, res=None):
    """The most complete run of an arm and seed (relaunches can leave several attempts)."""
    res = res or RES
    name = ARM_FILE[arm]
    if seed in (3, 4):
        job = PINNED.get((arm, seed))
        return load(res / job / arm, name) if job and (res / job / arm).exists() else None
    best = None
    for folder in sorted(res.glob(f"worth-v10-*-s{seed}-*/{arm}")):
        r = load(folder, name)
        if best is None or (len(r["points"]), len(r["train"])) > (len(best["points"]), len(best["train"])):
            best = r
    return best


def by_step(r, key=lambda x: x["first_block_reward"], steps=WINDOW):
    d = {x["step"]: x for x in r["train"]}
    return np.array([key(d[t]) for t in steps])


def block_indices(rng, n=25, block=5):
    starts = rng.integers(0, n - block + 1, math.ceil(n / block))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def pooled_phi(A, arm, seeds, idx=None):
    pick = lambda s, a: A[s][a] if idx is None else A[s][a][idx[s]]
    num = sum(pick(s, arm).mean() - pick(s, "NSR").mean() for s in seeds)
    den = sum(pick(s, "PSR").mean() - pick(s, "NSR").mean() for s in seeds)
    return num / den


def phi_with_interval(A, arm, seeds, draws, rng):
    est = pooled_phi(A, arm, seeds)
    bs = [pooled_phi(A, arm, seeds, {s: block_indices(rng) for s in seeds}) for _ in range(draws)]
    return est, np.percentile(bs, [5, 95]), np.percentile(bs, [2.5, 97.5])


def split_m2(steps, pre=False, shadow=False):
    """Per-question second moment and signal from split-half statistics; optionally after Adam
    scaling, or for the shadow estimator."""
    tot = dict.fromkeys(("ab", "aa", "pp", "pa2", "np"), 0.0)
    suffix = "_pre" if pre else ""
    for r in steps:
        sp = r.get("split")
        if sp and shadow:
            sp = dict(sp, **sp["shadow"]) if "shadow" in sp else None
        if not sp or f"aa{suffix}" not in sp or sp["prompts_a"] == 0 or sp["prompts_b"] == 0:
            continue
        pa, pb, z2 = sp["prompts_a"], sp["prompts_b"], sp["norm"] ** 2
        tot["ab"] += sp[f"ab{suffix}"] * z2
        tot["aa"] += (sp[f"aa{suffix}"] + sp[f"bb{suffix}"]) * z2
        tot["pp"] += pa * pb
        tot["pa2"] += pa * (pa - 1) + pb * (pb - 1)
        tot["np"] += pa + pb
    if tot["pp"] == 0:
        return math.nan, math.nan
    s = tot["ab"] / tot["pp"]
    return (tot["aa"] - s * tot["pa2"]) / tot["np"], s


def predicted_interp(r, steps, lo_m, hi_m, lo, hi):
    """As predicted(), with F0 and F1 interpolated linearly at each step between the measurements
    at steps lo and hi, before weighting by that step's counts."""
    num = cnt = 0.0
    for x in steps:
        f = (x["step"] - lo) / (hi - lo)
        F = {"F0": lo_m["F0"] + f * (hi_m["F0"] - lo_m["F0"]), "F1": lo_m["F1"] + f * (hi_m["F1"] - lo_m["F1"])}
        k = len(x["group_counts"])
        num += predicted(r, [x], F) * k
        cnt += k
    return num / cnt if cnt else math.nan


def predicted(r, steps, F=None):
    """Theorem's per-question second moment from F0, F1 (calibration unless given) and the observed counts."""
    calib = dict(r["calib"], **(F or {}))
    if r["name"] != "psrnoise":
        return predicted_noise(r["name"], steps, calib)
    tot, cnt = 0.0, 0
    for x in steps:
        c2 = x.get("noise_c", 0.0) ** 2
        for k, n in x["group_counts"]:
            tot += k * (1 + c2) * calib["F1"] + (n - k) * c2 * calib["F0"]
            cnt += 1
    return tot / cnt if cnt else math.nan


def f_ratio(m, rng, draws=1000):
    """F0 / F1 with a 95% interval from resampling questions."""
    if "norms_correct" not in m:
        return (m["F0"] / m["F1"] if m.get("F1") else math.inf), (math.nan, math.nan), m.get("n_correct", 0), m.get("n_wrong", 0)
    n1, n0 = np.array(m["norms_correct"]), np.array(m["norms_wrong"])
    q1, q0 = np.array(m["questions_correct"]), np.array(m["questions_wrong"])
    if not len(n1) or not len(n0):
        return math.inf, (math.nan, math.nan), len(n1), len(n0)
    qs = np.unique(np.concatenate([q1, q0]))
    bs = []
    for _ in range(draws):
        u, c = np.unique(rng.choice(qs, len(qs)), return_counts=True)
        cnt = dict(zip(u, c))
        w1, w0 = np.array([cnt.get(q, 0) for q in q1]), np.array([cnt.get(q, 0) for q in q0])
        if w1.sum() and w0.sum():
            bs.append((w0 @ n0 / w0.sum()) / (w1 @ n1 / w1.sum()))
    return n0.mean() / n1.mean(), tuple(np.percentile(bs, [2.5, 97.5])), len(n1), len(n0)


def pointwise(r_pts, p_pts, weights, step):
    """R's pass@16 minus P's pass@16 interpolated at R's pass@1, at one checkpoint (P's points
    ordered by pass@1, ties by step; outside P's range np.interp holds the end value)."""
    ps, px, py = curve(p_pts, weights)
    order = np.lexsort((ps, px))
    rs, rx, ry = curve(r_pts, weights)
    i = list(rs).index(step)
    return float(ry[i] - np.interp(rx[i], px[order], py[order]))


def equivalence(pairs, draws, rng):
    """pairs: (R points, P points) per seed. Average matched-accuracy equivalence, with a
    simultaneous 90% band over all qualifying checkpoints."""
    if not pairs or not all(r and p for r, p in pairs):
        return {"verdict": "missing data"}
    missing = [sorted(RUN_STEPS - set(r)) + sorted(REF_STEPS - set(p)) for r, p in pairs]
    if any(missing):
        return {"verdict": "incomplete evaluation", "missing": missing}
    ones = np.ones(len(next(iter(pairs[0][1].values()))))
    steps = []
    for r, p in pairs:
        _, px, _ = curve(p, ones)
        rs, rx, _ = curve(r, ones)
        steps.append([s for s, x in zip(rs, rx) if s != 0 and px.min() <= x <= px.max()])
    if any(len(st) < 2 for st in steps):
        return {"verdict": "insufficient overlap", "qualifying": steps}
    n_prob = len(ones)

    def all_points(w):
        return [[pointwise(r, p, w, s) for s in st] for (r, p), st in zip(pairs, steps)]

    full = all_points(ones)
    est = float(np.mean([np.mean(d) for d in full]))
    boot_mean, boot_pts = [], []
    for _ in range(draws):
        w = np.bincount(rng.integers(0, n_prob, n_prob), minlength=n_prob).astype(float)
        pts = all_points(w)
        boot_mean.append(np.mean([np.mean(d) for d in pts]))
        boot_pts.append(np.concatenate(pts))
    lo90, hi90 = np.percentile(boot_mean, [5, 95])
    lo95, hi95 = np.percentile(boot_mean, [2.5, 97.5])
    flat, bp = np.concatenate(full), np.array(boot_pts)
    se = bp.std(axis=0, ddof=1)
    q = np.percentile(np.max(np.abs(bp - flat) / np.where(se > 0, se, np.inf), axis=1), 90)
    band = [(d - q * e, d + q * e) for d, e in zip(flat, se)]
    verdict = ("average equivalence" if lo90 >= -2 and hi90 <= 2 else
               "different" if (lo95 > 0 or hi95 < 0) and abs(est) >= 2 else "inconclusive")
    labels = [(i, s) for i, st in enumerate(steps) for s in st]
    return {"estimate": est, "ci90": (lo90, hi90), "ci95": (lo95, hi95), "verdict": verdict,
            "per_seed": [float(np.mean(d)) for d in full], "points": list(zip(labels, flat, band)),
            "within3": all(lo >= -3 and hi <= 3 for lo, hi in band)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    runs = {(a, s): run(a, s) for a in ARM_FILE for s in (3, 4, 5, 6, 7)}
    have = lambda a, s, n=50: runs[(a, s)] is not None and len(runs[(a, s)]["train"]) >= n
    print("runs:", {f"{a}-s{s}": r["job"] for (a, s), r in runs.items() if r})

    print("\n1. Manipulation check: PSR+N / same-policy NSR shadow, per-question second moment, steps 0 to 49")
    manip_state = "incomplete"
    def complete_shadow(s):
        r = runs[("PSRN", s)]
        if r is None:
            return False
        d = {x["step"]: x for x in r["train"]}
        return all(t in d and d[t].get("split") and "shadow" in d[t]["split"] for t in range(50))
    seeds = [s for s in FRESH if complete_shadow(s)]
    manip_ok = len(seeds) == len(FRESH)
    if not manip_ok:
        print(f"   incomplete: steps 0 to 49 with paired raw and shadow statistics only for seeds {seeds}")
        manip_state = "incomplete"
    if seeds:
        w = {s: sorted((x for x in runs[("PSRN", s)]["train"] if x["step"] < 50), key=lambda x: x["step"]) for s in seeds}

        def ratio(idx=None):
            num = den = 0.0
            for s in seeds:
                st = w[s] if idx is None else [w[s][i] for i in idx[s]]
                num += split_m2(st)[0]
                den += split_m2(st, shadow=True)[0]
            return num / den
        est = ratio()
        bs = [ratio({s: block_indices(rng, len(w[s])) for s in seeds}) for _ in range(args.draws)]
        lo, hi = np.percentile(bs, [5, 95])
        if manip_ok:
            manip_ok = 1 / 1.3 <= lo and hi <= 1.3
            manip_state = "pass" if manip_ok else "fail"
        print(f"   pooled {est:.2f}, 90% [{lo:.2f}, {hi:.2f}] -> {manip_state}")
        for s in seeds:
            m, sig = split_m2(w[s])
            msh, sigsh = split_m2(w[s], shadow=True)
            pre = split_m2(w[s], pre=True)[0] / split_m2(w[s], pre=True, shadow=True)[0]
            sep = split_m2([x for x in runs[("NSR", s)]["train"] if x["step"] < 50])[0] if have("NSR", s) else math.nan
            cs = np.array([x.get("noise_c", math.nan) for x in runs[("PSRN", s)]["train"]])
            clip = np.mean([x["grad_norm"] > 1.0 for x in runs[("PSRN", s)]["train"]])
            print(f"   seed {s}: shadow ratio {m / msh:.2f}; centered {(m - sig) / (msh - sigsh):.2f}; after Adam scaling {pre:.2f}; "
                  f"against the separate NSR run {m / sep:.2f}; c_t mean {np.nanmean(cs):.2f}, zero on {np.mean(cs == 0):.0%} of steps; clipped {clip:.0%}")
    else:
        print("   missing data")

    print("\n2-3. Speed: A = training accuracy, steps 25 to 49; pooled phi = sum(A - A_NSR) / sum(A_PSR - A_NSR)")
    seeds = [s for s in FRESH if all(have(a, s) for a in ("PSR", "NSR", "PSRN", "NSR2"))]
    if len(seeds) < len(FRESH):
        print(f"   only seeds {seeds} complete")
    if seeds:
        A = {s: {a: by_step(runs[(a, s)]) for a in ("PSR", "NSR", "PSRN", "NSR2") + (("PSR2",) if have("PSR2", s) else ())}
             for s in seeds}
        gap = 100 * np.mean([A[s]["PSR"].mean() - A[s]["NSR"].mean() for s in seeds])
        for s in seeds:
            line = f"   seed {s}: " + ", ".join(f"{a} {100 * v.mean():.1f}" for a, v in A[s].items())
            for a in ("PSRN", "NSR2"):
                e, ci90, _ = phi_with_interval(A, a, [s], args.draws, rng)
                line += f"; phi({a}) {e:+.2f} [{ci90[0]:+.2f}, {ci90[1]:+.2f}]"
            print(line)
        e, ci90, ci95 = phi_with_interval(A, "PSRN", seeds, args.draws, rng)
        if gap < 5 or len(seeds) < len(FRESH):
            verdict = "inconclusive (pooled gap below 5 points or a seed missing)"
        elif ci90[0] >= -MARGIN and ci90[1] <= MARGIN:
            verdict = "trace-matched sign noise reproduces NSR's early accuracy lag"
        elif ci95[0] > MARGIN:
            verdict = "trace-matched sign noise does not reproduce NSR's lag (PSR+N clearly faster)"
        elif ci95[1] < -MARGIN:
            verdict = "PSR+N slower than NSR"
        else:
            verdict = "inconclusive"
        if manip_state == "incomplete":
            verdict = f"incomplete manipulation check (the rule alone gives: {verdict})"
        elif not manip_ok:
            verdict = f"manipulation failed (the rule alone gives: {verdict})"
        per = [pooled_phi(A, "PSRN", [s]) for s in seeds]
        cls = {(-1 if x < -MARGIN else 1 if x > MARGIN else 0) for x in per}
        hetero = len(cls) > 1
        print(f"   pooled gap {gap:.1f} points; phi(PSR+N) {e:+.2f}, 90% [{ci90[0]:+.2f}, {ci90[1]:+.2f}], "
              f"95% [{ci95[0]:+.2f}, {ci95[1]:+.2f}]; seed spread (SD) {np.std(per, ddof=1) if len(per) > 1 else math.nan:.2f}"
              + ("; HETEROGENEOUS" if hetero else ""))
        print(f"   primary verdict: {verdict}")
        e, ci90, ci95 = phi_with_interval(A, "NSR2", seeds, args.draws, rng)
        lr = ("inconclusive (pooled gap below 5 points or a seed missing)" if gap < 5 or len(seeds) < len(FRESH) else
              "doubling the learning rate recovers at least half of NSR's lag" if ci95[0] > 0.5 else
              "not recovered by doubling" if ci95[1] < 0.5 else "inconclusive")
        print(f"   phi(NSR x2) {e:+.2f}, 95% [{ci95[0]:+.2f}, {ci95[1]:+.2f}] -> {lr}")
        both = [s for s in seeds if "PSR2" in A[s]]
        if both and both != [5, 6]:
            print(f"   gap at 2e-6: incomplete (PSR x2 only for seeds {both})")
        elif both:
            pick = lambda s, a, idx: A[s][a] if idx is None else A[s][a][idx[s]]
            g2 = lambda idx=None: np.mean([pick(s, "PSR2", idx).mean() - pick(s, "NSR2", idx).mean() for s in both])
            g1 = np.mean([A[s]["PSR"].mean() - A[s]["NSR"].mean() for s in both])
            bs = [g2({s: block_indices(rng) for s in both}) for _ in range(args.draws)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"   gap at 2e-6 (seeds {both}): {100 * g2():.1f} points, 95% [{100 * lo:.1f}, {100 * hi:.1f}], "
                  f"{g2() / g1:.2f} of the 1e-6 gap -> "
                  + ("inconclusive (1e-6 gap below 5 points)" if 100 * g1 < 5 else "NSR still lags at 2e-6" if lo > 0 else "no clear lag at 2e-6"))

    print("\n4. Average matched-accuracy equivalence against PSR (pass@16 from 32 samples; 128 problems)")
    comps = (("PSRN", FRESH, "primary"), ("NSR", FRESH, "primary"), ("NSR2", FRESH, ""),
             ("NSR", (3, 4, 5, 6, 7), "secondary, historical plus fresh"), ("NSR", (3, 4), "by protocol"))
    for r_arm, seeds, tag in comps:
        pairs = [((runs[(r_arm, s)] or {}).get("points"), (runs[("PSR", s)] or {}).get("points")) for s in seeds]
        res = equivalence(pairs, args.draws, rng)
        head = f"   {r_arm} seeds {seeds}" + (f" ({tag})" if tag else "")
        if "estimate" not in res:
            print(f"{head}: {res['verdict']} {res.get('missing', res.get('qualifying', ''))}")
            continue
        print(f"{head}: {res['estimate']:+.2f}, 90% [{res['ci90'][0]:+.2f}, {res['ci90'][1]:+.2f}], 95% [{res['ci95'][0]:+.2f}, "
              f"{res['ci95'][1]:+.2f}] -> {res['verdict']}" + ("; no checkpoint outside ±3" if res["within3"] else "; some checkpoint band outside ±3"))
        print("      " + "; ".join(f"s{seeds[i]} step {st}: {d:+.1f} [{lo:+.1f}, {hi:+.1f}]" for (i, st), d, (lo, hi) in res["points"]))
    for s in FRESH:
        if have("PSR2", s, 150) and runs[("PSR2", s)]["points"] and runs[("NSR2", s)] and runs[("NSR2", s)]["points"] and runs[("PSR", s)]:
            p2 = dict(runs[("PSR2", s)]["points"])
            if 0 in runs[("PSR", s)]["points"]:
                p2.setdefault(0, runs[("PSR", s)]["points"][0])
            res = equivalence([(runs[("NSR2", s)]["points"], p2)], args.draws, rng)
            print(f"   NSR x2 vs PSR x2 seed {s} (matched rate, reported): {res.get('estimate', math.nan):+.2f} {res['verdict']}")

    print("\n5. F0 / F1 over training (same 64 calibration questions and sampling seed; 95% intervals over questions)")
    persist, states = True, []
    for s in FRESH:
        for a in ARM_FILE:
            r = runs[(a, s)]
            if not r or not r["calib"]:
                if a != "PSR2" or s in (5, 6):
                    persist = False
                continue
            out = []
            for m in [dict(r["calib"], step=0)] + r["remeasure"]:
                v, (lo, hi), n1, n0 = f_ratio(m, rng)
                states.append("incomplete" if min(n1, n0) == 0 or not math.isfinite(lo) else
                              "above" if lo > 1.2 else "below" if hi < 1.2 else "unclear")
                out.append(f"step {m['step']}: {v:.2f} [{lo:.2f}, {hi:.2f}]" + (f" (only {n1} correct, {n0} wrong)" if min(n1, n0) < 20 else ""))
            if sorted(m["step"] for m in r["remeasure"]) != [50, 100, 150]:
                states.append("incomplete")
            print(f"   {a} s{s}: " + ", ".join(out))
    verdict = ("incomplete" if "incomplete" in states or not persist else "asymmetry persists" if all(x == "above" for x in states)
               else "asymmetry fails somewhere" if "below" in states else "inconclusive")
    print(f"   verdict: {verdict} (persists: every 95% interval above 1.2; fails: some interval below 1.2)")

    print("\nReported without a rule")
    for s in FRESH:
        for a in ARM_FILE:
            r = runs[(a, s)]
            if not r or not r["train"] or not r["calib"]:
                continue
            F = {0: r["calib"], **{m["step"]: m for m in r["remeasure"]}}
            for lo, hi in ((0, 49), (50, 99), (100, 149)):
                steps = [x for x in r["train"] if lo <= x["step"] <= hi and x.get("split")]
                if not steps:
                    continue
                meas = split_m2(steps)[0]
                ends = [F.get(lo), F.get(hi + 1)]
                valid = all(e and e.get("n_correct", 0) > 0 and e.get("n_wrong", 0) > 0
                            and all(math.isfinite(e.get(k, math.nan)) and e.get(k, 0) > 0 for k in ("F0", "F1")) for e in ends)
                again = meas / predicted_interp(r, steps, ends[0], ends[1], lo, hi + 1) if valid else math.nan
                ent = [x["probe"]["entropy"] for x in r["train"]
                       if lo <= x["step"] <= hi and x.get("probe") and x["probe"].get("entropy") is not None]
                prog = [x["split"]["u_ref"] for x in steps if "u_ref" in x["split"]]
                uu = [x["split"]["uu"] for x in steps if "uu" in x["split"]]
                lr = 2e-6 if a.endswith("2") else 1e-6
                print(f"   {a} s{s} steps {lo}-{hi}: measured / theorem {meas / predicted(r, steps):.2f} (F interpolated "
                      f"between steps {lo} and {hi + 1}: {again:.2f}); lagged Adam direction alignment <u, ref> {np.mean(prog) if prog else math.nan:.3g} "
                      f"(x lr: {lr * np.mean(prog) if prog else math.nan:.3g}), |u|^2 {np.mean(uu) if uu else math.nan:.3g}; "
                      f"entropy {np.mean(ent) if ent else math.nan:.3f}; clipped {np.mean([x['grad_norm'] > 1.0 for x in steps]):.0%}")
        p, n = runs[("PSR", s)], runs[("NSR", s)]
        if p and n and p["train"] and n["train"]:
            w = lambda r: [x for x in r["train"] if 1 <= x["step"] < 50]
            prog = lambda r: np.mean([x["split"]["u_ref"] for x in w(r) if "u_ref" in x.get("split", {})] or [math.nan])  # same lr
            print(f"   seed {s}: NSR / PSR second moment raw {split_m2(w(n))[0] / split_m2(w(p))[0]:.2f}, "
                  f"after scaling by Adam's previous scale {split_m2(w(n), True)[0] / split_m2(w(p), True)[0]:.2f}; "
                  f"lagged Adam direction alignment NSR / PSR {prog(n) / prog(p):.2f}")
        for a in ("PSRN", "NSR2", "NSR"):
            ra = runs[(a, s)]
            if ra and p and len(ra["train"]) >= 150 and len(p["train"]) >= 150 and "boxed_rate" in ra["train"][0]:
                rate = lambda r, k: 100 * np.mean([x[k] for x in r["train"][100:150]])
                flag = (abs(rate(ra, "boxed_rate") - rate(p, "boxed_rate")) > 10
                        or abs(rate(ra, "truncated_rate") - rate(p, "truncated_rate")) > 10)
                print(f"   {a} vs PSR s{s}: boxed {rate(ra, 'boxed_rate'):.0f}% vs {rate(p, 'boxed_rate'):.0f}%, "
                      f"truncated {rate(ra, 'truncated_rate'):.0f}% vs {rate(p, 'truncated_rate'):.0f}%" + ("  FLAG" if flag else ""))


if __name__ == "__main__":
    main()
