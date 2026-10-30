"""Registered analysis of Pre-registration 30, Stage 2 (Amendment 4: R-25K and H-25K starts) and Pre-registration 34
(OPD-R-25K start). Written before any Stage 2 evaluation was inspected.

Notation: for a start X and RL seed s, the checkpoints of run X_s at steps 1, 2, 5, 10, 25, 50, 100, 150 (T = 1, 32
samples, 511 test problems); B1k_s is Dr. GRPO from the base with the same seed (base and steps 25-150, T = 1).
- Observed gap at a checkpoint with pass@1 u: its pass@16 minus B1k_s's pass@16 interpolated at u (Pre-registration
  21's interpolation; eligibility fixed on the full data: u inside the B1k_s trajectory's pass@1 range); averaged over
  eligible checkpoints within seed, seeds weighted equally; at least two eligible checkpoints per seed, otherwise
  "insufficient overlap". Envelope variant: the maximum with B1k_s's step-150 temperature curve (reported).
- Forecast gap: f_X(u) - f_B(u), the frozen law (tau = -0.20, strict) from the first 32 of the 64 samples per test
  problem of the start X and of the base (Stage 1's 64-sample base evaluation). Residual = observed - forecast gap.
- Bootstrap: 2,000 resamples of the 511 problems (indices, seed 30), shared by every quantity; the references, the
  laws' priors and initial states are recomputed in every draw. A checkpoint outside a reference's or a law's support
  in a draw is dropped for that draw; a seed with more than 10% of draws affected is "not interpreted".
- Verdicts (Holm over the advanced starts R25, H25; separately over P34's two contrasts):
  forecast agrees if the residual's 90% interval lies within +/-2; forecast fails if |residual| >= 2 and Holm p <= 0.05
  (two-sided test of residual = 0); else inconclusive. Curve raised: observed gap >= +2, 95% interval above 0, positive
  in all seeds, Holm p <= 0.05 (two-sided test of zero); lowered: the mirror rule; average equivalence: pooled and every
  seed's 90% interval within +/-2 (P34's definition, applied to both registrations here and reported as such).
  Targeting: H's observed gap minus R's >= 2 with a 95% interval above 0.
- P34 secondary: each OPD checkpoint's pass@16 minus the same-seed SFT-R trajectory (the R-25K start and its RL
  checkpoints) at that checkpoint's pass@1; same averaging and rules.
- Reported: step 0 (the start) gaps separately; per-checkpoint gaps; LH1 (seed 8, P27) as a descriptive extension
  for checkpoints beyond B1k's support.
Usage: python analyze_p30.py [--draws 2000]
"""

import argparse
import glob
import json
from math import comb

import numpy as np

import law

SEEDS = (8, 9, 10)
STEPS = (1, 2, 5, 10, 25, 50, 100, 150)
S2 = "results/worth-p30-stage2-s0-20261029-182830"
S1B = "results/worth-p30-stage1b-s0-20261029-164830"
S1 = "results/worth-p30-stage1-s0-20261029-084113"
_T = {}


def table(n, k):
    if (n, k) not in _T:
        _T[(n, k)] = np.array([1 - comb(n - x, k) / comb(n, k) for x in range(n + 1)])
    return _T[(n, k)]


def pk(c, n, k, idx):
    return float(table(n, k)[c[idx]].mean())


def counts(f):
    d = json.load(open(f))
    return np.array(d["counts"], int), d["n"]


def one(pat):
    f = sorted(glob.glob(pat))
    return counts(f[0]) if f else None


def first32(path):
    import gzip
    d = json.load(gzip.open(path, "rt"))
    n_prob = max(s["problem"] for s in d["samples"]) + 1
    seen, c = np.zeros(n_prob, int), np.zeros(n_prob, int)
    for s in d["samples"]:
        i = s["problem"]
        if seen[i] < 32:
            c[i] += bool(s["reward"])
        seen[i] += 1
    assert (seen == 64).all(), path
    return c


def run_points(name, seed, root):
    pts = {}
    for st in STEPS:
        r = one(f"{root}/{name}s{seed}/eval/step{st}_T1.00_eval.json")
        if r is not None:
            pts[st] = r
    return pts


def b1k(seed):
    base = one("results/worth-v15v-all-*/BASE15/eval/step0_T1.00_eval.json")
    traj = [(0, base)] + [(st, one(f"results/worth-v15v-all-*/B1k{seed}/eval/step{st}_T1.00_eval.json")) for st in (25, 50, 100, 150)]
    temps = [(t, one(f"results/worth-v15v-all-*/B1k{seed}/eval/step150_T{t:.2f}_eval.json")) for t in (1.0, 1.15, 1.3)]
    return traj, temps


def lh1():
    pts = [(0, one("results/worth-v15v-all-*/BASE15/eval/step0_T1.00_eval.json"))]
    for f in glob.glob("results/worth-p27-long-*/LH1/eval/step*_T1.00_eval.json"):
        pts.append((int(f.split("step")[1].split("_")[0]), counts(f)))
    return pts


def curve(points, idx):
    m = sorted(((pk(c, n, 1, idx), tb, pk(c, n, 16, idx)) for tb, (c, n) in points), key=lambda x: (x[0], x[1]))
    return np.array([x[0] for x in m]), np.array([x[2] for x in m])


class LawCache:
    def __init__(self, c32):
        self.c32, self.cache = c32, {}

    def at(self, u, idx, key):
        if key not in self.cache:
            c = self.c32[idx]
            self.cache = {key: law.trajectory(law.initial_theta(c, 32, law.fit_prior(c, 32)), -0.20)}
        try:
            return law.pass_k_true(law.at_pass1(self.cache[key], u, strict=True), 16)
        except ValueError:
            return None


def gaps_for(arm_pts, ref_fn, idx, elig, laws=None, key=None):
    """Per eligible checkpoint: observed gap, forecast gap (if laws), or None when out of support in this draw."""
    out = {}
    for st in elig:
        c, n = arm_pts[st]
        u, v = pk(c, n, 1, idx), pk(c, n, 16, idx)
        r = ref_fn(u, idx)
        if r is None:
            out[st] = None
            continue
        g = {"obs": 100 * (v - r)}
        if laws is not None:
            fx, fb = laws[0].at(u, idx, key), laws[1].at(u, idx, key)
            g["fc"] = None if fx is None or fb is None else 100 * (fx - fb)
        out[st] = g
    return out


def interp_ref(points, envelope=None):
    def f(u, idx):
        vals = []
        for P in [points] + ([envelope] if envelope else []):
            xs, ys = curve(P, idx)
            if xs[0] <= u <= xs[-1]:
                vals.append(float(np.interp(u, xs, ys)))
        return max(vals) if vals else None
    return f


def summarize(name, per_seed_draws, full, key="obs"):
    """per_seed_draws[s] = list over draws of per-checkpoint dicts; full[s] = full-data dict."""
    est, ps, drop = {}, {}, {}
    for s in full:
        vals = [g[key] for g in full[s].values() if g is not None and g.get(key) is not None]
        est[s] = float(np.mean(vals)) if vals else np.nan
        bd, nbad = [], 0
        for d in per_seed_draws[s]:
            v = [g[key] for g in d.values() if g is not None and g.get(key) is not None]
            if len(v) < len(full[s]):
                nbad += 1
            bd.append(np.mean(v) if v else np.nan)
        ps[s], drop[s] = np.array(bd), nbad / len(per_seed_draws[s])
    pooled = np.nanmean(np.vstack([ps[s] for s in full]), axis=0)
    ok = pooled[~np.isnan(pooled)]
    return {"name": name, "est": float(np.nanmean(list(est.values()))), "per_seed": est,
            "ci90": np.percentile(ok, [5, 95]), "ci95": np.percentile(ok, [2.5, 97.5]),
            "per_seed_ci90": {s: np.nanpercentile(ps[s], [5, 95]) for s in full},
            "p": float(min(1.0, 2 * min((ok <= 0).mean(), (ok >= 0).mean()))), "drop": drop, "draws": pooled}


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def curve_verdict(t, q):
    if any(v > 0.10 for v in t["drop"].values()):
        return "not interpreted (support)"
    pos = all(v > 0 for v in t["per_seed"].values())
    neg = all(v < 0 for v in t["per_seed"].values())
    if t["est"] >= 2 and t["ci95"][0] > 0 and pos and q <= 0.05:
        return "curve raised"
    if t["est"] <= -2 and t["ci95"][1] < 0 and neg and q <= 0.05:
        return "curve lowered"
    if t["ci90"][0] >= -2 and t["ci90"][1] <= 2 and all(lo >= -2 and hi <= 2 for lo, hi in t["per_seed_ci90"].values()):
        return "average equivalence"
    return "inconclusive"


def forecast_verdict(t, q):
    if any(v > 0.10 for v in t["drop"].values()):
        return "not interpreted (support)"
    if t["ci90"][0] >= -2 and t["ci90"][1] <= 2:
        return "forecast agrees"
    if abs(t["est"]) >= 2 and q <= 0.05:
        return "forecast fails (bias detected)"
    return "inconclusive"


def show(t, v, q):
    ps = ", ".join(f"{s}: {t['per_seed'][s]:+.2f} [{t['per_seed_ci90'][s][0]:+.2f}, {t['per_seed_ci90'][s][1]:+.2f}]" for s in t["per_seed"])
    dr = ", ".join(f"{s}: {100 * x:.1f}%" for s, x in t["drop"].items())
    return (f"   {t['name']}: {t['est']:+.2f}, 90% [{t['ci90'][0]:+.2f}, {t['ci90'][1]:+.2f}], 95% [{t['ci95'][0]:+.2f}, "
            f"{t['ci95'][1]:+.2f}]; per seed {ps}; draws affected {dr}; Holm p {q:.3f} -> {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    starts = {"R25": ("S2R25", S2, f"{S1B}/R025_test"), "H25": ("S2H25", S2, f"{S1B}/H025_test"),
              "OPD25": ("P34OPD25", S2, f"{S2}/OPD25_test")}
    extra_roots = glob.glob("results/worth-p34-s10-s0-*")  # P34 seed 10 (separate job)
    base_c32 = first32(f"{S1}/BASE_test/eval/step0_T1.00_texts.json.gz")
    lawB = LawCache(base_c32)
    refs = {s: b1k(s) for s in SEEDS}
    P = 511
    rng = np.random.default_rng(30)
    draws = [rng.integers(0, P, P) for _ in range(a.draws)]
    full_idx = np.arange(P)
    results, data = {}, {}
    for X, (run, root, start_dir) in starts.items():
        pts = {}
        for s in SEEDS:
            p = run_points(run, s, root)
            if len(p) < len(STEPS) and X == "OPD25":
                for r in extra_roots:
                    p = run_points(run, s, r) or p
            pts[s] = p
        miss = {s: [st for st in STEPS if st not in pts[s]] for s in SEEDS}
        if any(miss.values()):
            print(f"{X}: missing checkpoints {miss}; not analysed")
            continue
        sc = counts(f"{start_dir}/eval/step0_T1.00_eval.json")
        lawX = LawCache(first32(f"{start_dir}/eval/step0_T1.00_texts.json.gz"))
        data[X] = (pts, sc)
        out = {}
        for kind in ("traj", "env"):
            per_draw, full, per_ckpt = {}, {}, {}
            for s in SEEDS:
                traj, temps = refs[s]
                R = interp_ref(traj, temps if kind == "env" else None)
                xs, _ = curve(traj, full_idx)
                elig = [st for st in STEPS if xs[0] <= pk(*pts[s][st], 1, full_idx) <= xs[-1]]
                if len(elig) < 2:
                    full[s] = None
                    continue
                full[s] = gaps_for(pts[s], R, full_idx, elig, (lawX, lawB) if kind == "traj" else None, ("full", s))
                per_draw[s] = [gaps_for(pts[s], R, i, elig, (lawX, lawB) if kind == "traj" else None, (k, s))
                               for k, i in enumerate(draws)]
                per_ckpt[s] = full[s]
            if any(v is None for v in full.values()):
                out[kind] = {"verdict": "insufficient overlap", "elig": {s: (list(full[s]) if full[s] else []) for s in SEEDS}}
                continue
            out[kind] = {"obs": summarize(f"{X} observed gap vs B1k {'envelope' if kind == 'env' else 'trajectory'}", per_draw, full, "obs"),
                         "per_ckpt": per_ckpt}
            if kind == "traj":
                res_full = {s: {st: (None if g is None or g.get("fc") is None else {"res": g["obs"] - g["fc"], "fc": g["fc"]})
                                for st, g in full[s].items()} for s in SEEDS}
                res_draw = {s: [{st: (None if g is None or g.get("fc") is None else {"res": g["obs"] - g["fc"], "fc": g["fc"]})
                                 for st, g in d.items()} for d in per_draw[s]] for s in SEEDS}
                out[kind]["res"] = summarize(f"{X} forecast residual", res_draw, res_full, "res")
                out[kind]["fc"] = summarize(f"{X} forecast gap", res_draw, res_full, "fc")
        # step 0: the start itself against B1k_s (64-sample counts; pass@16 unbiased from 64)
        s0 = []
        for s in SEEDS:
            R = interp_ref(refs[s][0])
            r = R(pk(*sc, 1, full_idx), full_idx)
            s0.append(None if r is None else 100 * (pk(*sc, 16, full_idx) - r))
        out["step0"] = s0
        results[X] = out
    print("Pre-registration 30, Stage 2 (Amendment 4) and Pre-registration 34; 511 test problems, 1.5B, 1,024 tokens")
    fam = [X for X in ("R25", "H25") if X in results and "obs" in results[X]["traj"]]
    for family, members, kinds in (("P30 (advanced starts R25, H25)", fam, ("obs", "res")),):
        for kind in kinds:
            qs = holm(np.array([results[X]["traj"][kind]["p"] for X in members])) if members else []
            for X, q in zip(members, qs):
                t = results[X]["traj"][kind]
                print(show(t, curve_verdict(t, q) if kind == "obs" else forecast_verdict(t, q), q))
                if kind == "res":
                    f = results[X]["traj"]["fc"]
                    print(f"      forecast gap (mean over eligible checkpoints): {f['est']:+.2f} [{f['ci90'][0]:+.2f}, {f['ci90'][1]:+.2f}]")
    for X in ("R25", "H25", "OPD25"):
        if X not in results:
            continue
        r = results[X]
        if "obs" in r["env"]:
            t = r["env"]["obs"]
            print(f"   {t['name']} (reported): {t['est']:+.2f}, 90% [{t['ci90'][0]:+.2f}, {t['ci90'][1]:+.2f}]")
        print(f"   {X} step 0 (start) minus B1k_s at its pass@1, seeds 8/9/10: {', '.join('n/a' if v is None else f'{v:+.2f}' for v in r['step0'])}")
        if "per_ckpt" in r["traj"]:
            for s in SEEDS:
                row = r["traj"]["per_ckpt"][s]
                print(f"      {X} seed {s} per-checkpoint observed gaps: " + ", ".join(
                    f"{st}: {g['obs']:+.1f}" for st, g in row.items() if g is not None))
    if "R25" in results and "H25" in results and "obs" in results["R25"]["traj"] and "obs" in results["H25"]["traj"]:
        d = results["H25"]["traj"]["obs"]["draws"] - results["R25"]["traj"]["obs"]["draws"]
        e = results["H25"]["traj"]["obs"]["est"] - results["R25"]["traj"]["obs"]["est"]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        print(f"   targeting: H25 minus R25 observed gap {e:+.2f}, 95% [{lo:+.2f}, {hi:+.2f}] -> "
              f"{'targeting claim supported' if e >= 2 and lo > 0 else 'targeting claim not supported'}")
    # P34
    if "OPD25" in results and "obs" in results["OPD25"]["traj"] and "R25" in data:
        prim = results["OPD25"]["traj"]["obs"]
        ptsO, _ = data["OPD25"]
        ptsR, scR = data["R25"]
        per_draw, full = {}, {}
        for s in SEEDS:
            sft_traj = [(0, scR)] + [(st, ptsR[s][st]) for st in STEPS]
            R = interp_ref(sft_traj)
            xs, _ = curve(sft_traj, full_idx)
            elig = [st for st in STEPS if xs[0] <= pk(*ptsO[s][st], 1, full_idx) <= xs[-1]]
            full[s] = gaps_for(ptsO[s], R, full_idx, elig) if len(elig) >= 2 else None
            per_draw[s] = [gaps_for(ptsO[s], R, i, elig) for i in draws] if full[s] is not None else None
        if any(v is None for v in full.values()):
            sec = None
            print("   P34 secondary: insufficient overlap with the SFT-R trajectories")
        else:
            sec = summarize("P34 secondary: OPD25 minus SFT-R25 trajectory", per_draw, full, "obs")
        tests = [prim] + ([sec] if sec else [])
        qs = holm(np.array([t["p"] for t in tests]))
        print("Pre-registration 34 (Holm over its contrasts)")
        for t, q in zip(tests, qs):
            print(show(t, curve_verdict(t, q), q))
    elif "OPD25" not in results:
        print("Pre-registration 34: incomplete (see missing checkpoints above)")
    lh = lh1()
    for X in ("R25", "H25", "OPD25"):
        if X not in data:
            continue
        pts, _ = data[X]
        R = interp_ref(lh)
        vals = []
        for s in SEEDS:
            for st in STEPS:
                c, n = pts[s][st]
                u = pk(c, n, 1, full_idx)
                if u > max(pk(*refs[s][0][-1][1], 1, full_idx), 0):
                    r = R(u, full_idx)
                    if r is not None:
                        vals.append((s, st, 100 * (pk(c, n, 16, full_idx) - r)))
        if vals:
            print(f"   {X} beyond B1k's support, gap vs LH1 (seed 8, descriptive): " + ", ".join(f"s{s} step {st}: {g:+.1f}" for s, st, g in vals))


if __name__ == "__main__":
    main()
