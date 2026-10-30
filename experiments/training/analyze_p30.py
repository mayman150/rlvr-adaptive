"""Registered analysis of Pre-registration 30, Stage 2 (Amendment 4: R-25K and H-25K starts) and Pre-registration 34
(OPD-R-25K start). Written before any Stage 2 evaluation was inspected; revised after Codex's pre-run review
(codex_analyze_p30_review.md), still before inspection.

Notation: for a start X and RL seed s, run X_s has checkpoints at steps 1, 2, 5, 10, 25, 50, 100, 150 (T = 1, 32
samples, 511 test problems); B1k_s is Dr. GRPO from the base with the same seed (base and steps 25-150, T = 1).
- Observed gap at a checkpoint with pass@1 u: pass@16 minus B1k_s's pass@16 interpolated at u (sorted by pass@1, ties
  by step); eligibility fixed on the full data (u inside B1k_s's pass@1 range; for the envelope, inside the union of
  the trajectory's and the step-150 temperature curve's ranges); averaged over eligible checkpoints within seed, seeds
  weighted equally; at least two eligible checkpoints per seed, otherwise "insufficient overlap".
- Forecast gap: f_X(u) - f_B(u), frozen law (tau = -0.20, strict) from the first 32 of the 64 samples per test problem
  of start X and of the base (Stage 1's 64-sample base evaluation). Residual = observed - forecast gap. A checkpoint
  is dropped from a draw if it leaves the reference's or either law's support (joint rule); a full-data eligible
  checkpoint without a forecast makes the forecast "unscorable".
- Bootstrap: 2,000 index resamples of the 511 problems (seed 30), shared by every quantity; references and laws
  recomputed in every draw. A seed with more than 10% of draws affected is "not interpreted (support)".
- Families: P30 = {R25, H25} for the observed gap, separately for the residual, separately for the envelope (Holm,
  two-sided tests of zero; an unavailable member enters with p = 1). P34 = {primary, secondary}.
- Rules: raised: gap >= +2, 95% interval above 0, positive in all seeds, Holm p <= 0.05; lowered: the mirror rule;
  forecast agrees: residual's 90% interval within +/-2; forecast fails: |residual| >= 2 and Holm p <= 0.05; P34 only:
  average equivalence if the pooled and every seed's 90% interval lie within +/-2; otherwise inconclusive.
  Targeting (P30): H minus R observed gap >= 2 with a 95% interval above 0, interpreted only if both are supported.
- P34 secondary: each OPD checkpoint's pass@16 minus the same-seed SFT-R trajectory (the R-25K start and its RL
  checkpoints) at that checkpoint's pass@1.
- Reported: step 0 (the start) separately; per-checkpoint gaps with pointwise 90% bands and absolute values; LH1
  (seed 8 only, P27) as a descriptive extension beyond B1k8's support; pre-RL pass@k of the starts; evaluation
  diagnostics; base-count strata at step 150; OPD training log.
Usage: python analyze_p30.py   (confirmatory: 2,000 draws; --draws N for exploratory runs only)
"""

import argparse
import glob
import gzip
import json
from math import comb

import numpy as np

import law

SEEDS = (8, 9, 10)
STEPS = (1, 2, 5, 10, 25, 50, 100, 150)
S2 = "results/worth-p30-stage2-s0-20261029-182830"
S1B = "results/worth-p30-stage1b-s0-20261029-164830"
S1 = "results/worth-p30-stage1-s0-20261029-084113"
OPD10 = "results/worth-p34-s10-s0-20261030-050310"  # P34 seed 10 (separate one-GPU job)
P = 511
_T = {}


def table(n, k):
    if (n, k) not in _T:
        _T[(n, k)] = np.array([1 - comb(n - x, k) / comb(n, k) for x in range(n + 1)])
    return _T[(n, k)]


def pk(c, n, k, idx):
    return float(table(n, k)[c[idx]].mean())


def one(pat, need=True):
    f = sorted(glob.glob(pat))
    if len(f) > 1:
        raise SystemExit(f"multiple matches for {pat}: {f}")
    if not f:
        if need:
            raise SystemExit(f"missing {pat}")
        return None
    d = json.load(open(f[0]))
    c = np.array(d["counts"], int)
    assert len(c) == P, (f[0], len(c))
    return c, d["n"], d


def first32(path):
    d = json.load(gzip.open(path, "rt"))
    seen, c = np.zeros(P, int), np.zeros(P, int)
    for s in d["samples"]:
        i = s["problem"]
        if seen[i] < 32:
            c[i] += bool(s["reward"])
        seen[i] += 1
    assert (seen == 64).all(), path
    return c


def run_points(run, seed, root):
    out = {}
    for st in STEPS:
        r = one(f"{root}/{run}s{seed}/eval/step{st}_T1.00_eval.json", need=False)
        if r is not None:
            out[st] = r
    return out


def b1k(seed):
    base = one("results/worth-v15v-all-*/BASE15/eval/step0_T1.00_eval.json")
    traj = [(0, base)] + [(st, one(f"results/worth-v15v-all-*/B1k{seed}/eval/step{st}_T1.00_eval.json")) for st in (25, 50, 100, 150)]
    temps = [(t, one(f"results/worth-v15v-all-*/B1k{seed}/eval/step150_T{t:.2f}_eval.json")) for t in (1.0, 1.15, 1.3)]
    return traj, temps


def lh1():
    pts = [(0, one("results/worth-v15v-all-*/BASE15/eval/step0_T1.00_eval.json"))]
    for f in sorted(glob.glob("results/worth-p27-long-*/LH1/eval/step*_T1.00_eval.json")):
        d = json.load(open(f))
        pts.append((int(f.split("step")[1].split("_")[0]), (np.array(d["counts"], int), d["n"], d)))
    return pts


def curve(points, idx):
    m = sorted(((pk(c[0], c[1], 1, idx), tb, pk(c[0], c[1], 16, idx)) for tb, c in points), key=lambda x: (x[0], x[1]))
    return np.array([x[0] for x in m]), np.array([x[2] for x in m])


def interp_ref(points, envelope=None):
    def f(u, idx):
        vals = []
        for Pts in [points] + ([envelope] if envelope else []):
            xs, ys = curve(Pts, idx)
            if xs[0] <= u <= xs[-1]:
                vals.append(float(np.interp(u, xs, ys)))
        return max(vals) if vals else None
    return f


def support(points, envelope, idx):
    ranges = [curve(Pts, idx)[0][[0, -1]] for Pts in [points] + ([envelope] if envelope else [])]
    return min(r[0] for r in ranges), max(r[1] for r in ranges)


class LawCache:
    """The law's trajectory for the current draw only (the bootstrap loop is draw-major)."""
    def __init__(self, c32):
        self.c32, self.key, self.tr = c32, None, None

    def at(self, u, idx, key):
        if key != self.key:
            c = self.c32[idx]
            self.key, self.tr = key, law.trajectory(law.initial_theta(c, 32, law.fit_prior(c, 32)), -0.20)
        try:
            return law.pass_k_true(law.at_pass1(self.tr, u, strict=True), 16)
        except ValueError:
            return None


def gaps_for(pts, R, idx, elig, laws, key):
    out = {}
    for st in elig:
        c, n, _ = pts[st]
        u, v = pk(c, n, 1, idx), pk(c, n, 16, idx)
        r = R(u, idx)
        if r is None:
            out[st] = None
            continue
        g = {"u": 100 * u, "v": 100 * v, "r": 100 * r, "obs": 100 * (v - r)}
        if laws is not None:
            fx, fb = laws[0].at(u, idx, key), laws[1].at(u, idx, key)
            if fx is None or fb is None:
                if key != "full":
                    out[st] = None  # joint support rule: the checkpoint leaves a law's support in this draw
                    continue
                g["fc"] = None
            else:
                g.update({"fx": 100 * fx, "fb": 100 * fb, "fc": 100 * (fx - fb), "res": 100 * (v - r) - 100 * (fx - fb)})
        out[st] = g
    return out


def summarize(name, per_draw, full, key):
    est, ps, drop, ok_seed = {}, {}, {}, True
    for s in SEEDS:
        vals = [g[key] for g in full[s].values() if g is not None and g.get(key) is not None]
        if len(vals) < len(full[s]):
            return {"name": name, "status": "unscorable (a full-data eligible checkpoint has no value)"}
        est[s] = float(np.mean(vals))
        bd, nbad = [], 0
        for d in per_draw[s]:
            v = [g[key] for g in d.values() if g is not None and g.get(key) is not None]
            nbad += len(v) < len(full[s])
            bd.append(np.mean(v) if v else np.nan)
        ps[s], drop[s] = np.array(bd), nbad / len(per_draw[s])
    pooled = np.mean(np.vstack([ps[s] for s in SEEDS]), axis=0)  # equal seed weights; a draw missing a seed is dropped
    ok = pooled[np.isfinite(pooled)]
    if ok.size == 0:
        return {"name": name, "status": "not interpreted (support)"}
    t = {"name": name, "est": float(np.mean(list(est.values()))), "per_seed": est, "draws": pooled,
         "ci90": np.percentile(ok, [5, 95]), "ci95": np.percentile(ok, [2.5, 97.5]),
         "per_seed_ci90": {s: np.nanpercentile(ps[s], [5, 95]) for s in SEEDS},
         "p": float(min(1.0, 2 * min((ok <= 0).mean(), (ok >= 0).mean()))), "drop": drop}
    t["p_beyond_2"] = float(np.mean(ok <= 2)) if t["est"] >= 0 else float(np.mean(ok >= -2))
    t["status"] = "not interpreted (support)" if any(x > 0.10 for x in drop.values()) else "ok"
    return t


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def curve_verdict(t, q, equivalence=False):
    if t.get("status") != "ok":
        return t.get("status", "unavailable")
    if t["est"] >= 2 and t["ci95"][0] > 0 and all(v > 0 for v in t["per_seed"].values()) and q <= 0.05:
        return "curve raised"
    if t["est"] <= -2 and t["ci95"][1] < 0 and all(v < 0 for v in t["per_seed"].values()) and q <= 0.05:
        return "curve lowered"
    if equivalence and t["ci90"][0] >= -2 and t["ci90"][1] <= 2 and all(lo >= -2 and hi <= 2 for lo, hi in t["per_seed_ci90"].values()):
        return "average equivalence"
    return "inconclusive"


def forecast_verdict(t, q):
    if t.get("status") != "ok":
        return t.get("status", "unavailable")
    if t["ci90"][0] >= -2 and t["ci90"][1] <= 2:
        return "forecast agrees"
    if abs(t["est"]) >= 2 and q <= 0.05:
        return "forecast fails (bias detected)"
    return "inconclusive"


def show(t, v, q):
    if "est" not in t:
        return f"   {t['name']}: {t.get('status')}"
    ps = ", ".join(f"{s}: {t['per_seed'][s]:+.2f} [{t['per_seed_ci90'][s][0]:+.2f}, {t['per_seed_ci90'][s][1]:+.2f}]" for s in t["per_seed"])
    dr = ", ".join(f"{s}: {100 * x:.1f}%" for s, x in t["drop"].items())
    return (f"   {t['name']}: {t['est']:+.2f}, 90% [{t['ci90'][0]:+.2f}, {t['ci90'][1]:+.2f}], 95% [{t['ci95'][0]:+.2f}, "
            f"{t['ci95'][1]:+.2f}]; per seed {ps}; draws affected {dr}; Holm p {q:.3f}; "
            f"one-sided p(|x| <= 2 side) {t['p_beyond_2']:.3f} -> {v}")


def contrast(pts, ref_fn, elig_fn, draws, laws=None):
    """Full-data and per-draw gaps for one arm against per-seed references; None if any seed lacks two checkpoints."""
    full, per, el = {}, {s: [] for s in SEEDS}, {}
    for s in SEEDS:
        lo, hi = elig_fn(s)
        el[s] = [st for st in STEPS if lo <= pk(pts[s][st][0], pts[s][st][1], 1, np.arange(P)) <= hi]
        if len(el[s]) < 2:
            return None, el
        full[s] = gaps_for(pts[s], ref_fn(s), np.arange(P), el[s], laws, "full")
    for k, i in enumerate(draws):
        for s in SEEDS:
            per[s].append(gaps_for(pts[s], ref_fn(s), i, el[s], laws, k))
    return (full, per), el


def bands(full, per, key="obs"):
    out = {}
    for s in SEEDS:
        out[s] = {}
        for st, g in full[s].items():
            vals = [np.nan if d[st] is None or d[st].get(key) is None else d[st][key] for d in per[s]]
            out[s][st] = (g, np.nanpercentile(vals, [5, 95]) if np.isfinite(vals).any() else (np.nan, np.nan))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    tag = "" if a.draws == 2000 else f" (EXPLORATORY: {a.draws} draws)"
    full_idx = np.arange(P)
    rng = np.random.default_rng(30)
    draws = [rng.integers(0, P, P) for _ in range(a.draws)]
    refs = {s: b1k(s) for s in SEEDS}
    base_test = one(f"{S1}/BASE_test/eval/step0_T1.00_eval.json")
    lawB = LawCache(first32(f"{S1}/BASE_test/eval/step0_T1.00_texts.json.gz"))
    starts = {"R25": ("S2R25", {s: S2 for s in SEEDS}, f"{S1B}/R025_test"),
              "H25": ("S2H25", {s: S2 for s in SEEDS}, f"{S1B}/H025_test"),
              "OPD25": ("P34OPD25", {8: S2, 9: S2, 10: OPD10}, f"{S2}/OPD25_test")}
    print(f"Pre-registration 30, Stage 2 (Amendment 4: D = 25K, a registered fallback deviation) and Pre-registration 34; "
          f"511 test problems, 1.5B, 1,024 tokens{tag}")
    data, res = {}, {}
    for X, (run, roots, sdir) in starts.items():
        sc = one(f"{sdir}/eval/step0_T1.00_eval.json")
        print(f"\n== {X} start (step 0, 64 samples): " + ", ".join(f"pass@{k} {100 * pk(sc[0], sc[1], k, full_idx):.1f}" for k in (1, 16, 32, 64))
              + f"; truncated {100 * sc[2].get('truncated_rate', float('nan')):.1f}%")
        s0 = []
        for s in SEEDS:
            r = interp_ref(refs[s][0])(pk(sc[0], sc[1], 1, full_idx), full_idx)
            s0.append("n/a" if r is None else f"{100 * (pk(sc[0], sc[1], 16, full_idx) - r):+.2f}")
        print(f"   step 0 minus B1k_s at its pass@1 (seeds 8/9/10; not part of the post-RL estimate): {', '.join(s0)}")
        pts = {s: run_points(run, s, roots[s]) for s in SEEDS}
        miss = {s: [st for st in STEPS if st not in pts[s]] for s in SEEDS}
        if any(miss.values()):
            print(f"   missing checkpoints {miss}: not analysed")
            continue
        data[X] = (pts, sc)
        lawX = LawCache(first32(f"{sdir}/eval/step0_T1.00_texts.json.gz"))
        out = {}
        tr, el = contrast(pts, lambda s: interp_ref(refs[s][0]), lambda s: support(refs[s][0], None, full_idx), draws, (lawX, lawB))
        out["elig"] = el
        if tr is None:
            for k in ("obs", "res", "fc"):
                out[k] = {"name": f"{X} {k}", "status": f"insufficient overlap (eligible steps {el})"}
        else:
            full, per = tr
            out["obs"] = summarize(f"{X} observed gap vs B1k trajectory", per, full, "obs")
            out["res"] = summarize(f"{X} forecast residual", per, full, "res")
            out["fc"] = summarize(f"{X} forecast gap", per, full, "fc")
            out["bands"] = bands(full, per)
        env, el2 = contrast(pts, lambda s: interp_ref(refs[s][0], refs[s][1]), lambda s: support(refs[s][0], refs[s][1], full_idx), draws)
        out["env"] = summarize(f"{X} observed gap vs B1k envelope", env[1], env[0], "obs") if env else \
            {"name": f"{X} envelope", "status": f"insufficient overlap (eligible steps {el2})"}
        res[X] = out
    # P30 families
    print("\nPre-registration 30 (Holm over R25, H25 within each family)")
    for kind, verdict in (("obs", curve_verdict), ("res", forecast_verdict), ("env", curve_verdict)):
        ts = [res.get(X, {}).get(kind, {"name": f"{X} {kind}", "status": "unavailable"}) for X in ("R25", "H25")]
        qs = holm([t["p"] if t.get("status") == "ok" else 1.0 for t in ts])
        for t, q in zip(ts, qs):
            print(show(t, verdict(t, q), q))
            if kind == "res" and "fc" in res.get(t["name"].split()[0], {}):
                f = res[t["name"].split()[0]]["fc"]
                if "est" in f:
                    print(f"      forecast gap itself: {f['est']:+.2f} [{f['ci90'][0]:+.2f}, {f['ci90'][1]:+.2f}]")
    tR, tH = res.get("R25", {}).get("obs", {}), res.get("H25", {}).get("obs", {})
    if tR.get("status") == "ok" and tH.get("status") == "ok":
        d = tH["draws"] - tR["draws"]
        d = d[np.isfinite(d)]
        e = tH["est"] - tR["est"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"   targeting: H25 minus R25 observed gap {e:+.2f}, 95% [{lo:+.2f}, {hi:+.2f}] -> "
              f"{'targeting claim supported' if e >= 2 and lo > 0 else 'targeting claim not supported'}")
    else:
        print("   targeting: not interpreted (a contrast lacks support)")
    # P34
    print("\nPre-registration 34 (Holm over primary and secondary)")
    prim = res.get("OPD25", {}).get("obs", {"name": "P34 primary (OPD25 vs B1k)", "status": "unavailable"})
    if "OPD25" in data and "R25" in data:
        ptsO, _ = data["OPD25"]
        ptsR, scR = data["R25"]
        sft = {s: [(0, scR)] + [(st, ptsR[s][st]) for st in STEPS] for s in SEEDS}
        sec_c, el3 = contrast(ptsO, lambda s: interp_ref(sft[s]), lambda s: support(sft[s], None, full_idx), draws)
        sec = summarize("P34 secondary: OPD25 minus the same-seed SFT-R25 trajectory", sec_c[1], sec_c[0], "obs") if sec_c else \
            {"name": "P34 secondary", "status": f"insufficient overlap (eligible steps {el3})"}
    else:
        sec = {"name": "P34 secondary", "status": "unavailable"}
    qs = holm([t["p"] if t.get("status") == "ok" else 1.0 for t in (prim, sec)])
    for t, q in zip((prim, sec), qs):
        print(show(t, curve_verdict(t, q, equivalence=True), q))
    # per-checkpoint reports
    print("\nPer-checkpoint observed gaps vs B1k_s (pointwise 90% bands; absolute pass@1 / pass@16 / reference)")
    for X, out in res.items():
        if "bands" not in out:
            print(f"   {X}: {out['obs'].get('status')}")
            continue
        for s in SEEDS:
            row = []
            for st, (g, b) in out["bands"][s].items():
                if g is not None:
                    fcs = f", forecast {g['fc']:+.1f}" if g.get("fc") is not None else ""
                    row.append(f"{st}: {g['obs']:+.1f} [{b[0]:+.1f}, {b[1]:+.1f}] ({g['u']:.1f}/{g['v']:.1f}/{g['r']:.1f}{fcs})")
            unsup = [st for st in STEPS if st not in out["elig"][s]]
            print(f"   {X} seed {s}: " + "; ".join(row) + (f" | beyond B1k's support: steps {unsup}" if unsup else ""))
    # LH1 extension (seed 8 only)
    R = interp_ref(lh1())
    for X, (pts, _) in data.items():
        hi = curve(refs[8][0], full_idx)[0][-1]
        vals = []
        for st in STEPS:
            c, n, _ = pts[8][st]
            u = pk(c, n, 1, full_idx)
            if u > hi:
                r = R(u, full_idx)
                vals.append(f"step {st}: {'n/a' if r is None else f'{100 * (pk(c, n, 16, full_idx) - r):+.1f}'}")
        if vals:
            print(f"   {X} seed 8 beyond B1k8's support, gap vs LH1 (seed 8, descriptive): {', '.join(vals)}")
    # diagnostics and strata at step 150
    c0 = base_test[0]
    strata = {"base c=0 (of 64)": c0 == 0, "c=1-8": (c0 >= 1) & (c0 <= 8), "c>=9": c0 >= 9}
    print("\nStep 150 (T = 1): pass@1, pass@16, truncation, boxed rate; pass@16 by base-count stratum (Stage 1 base, 64 samples)")
    for X, (pts, _) in data.items():
        for s in SEEDS:
            c, n, d = pts[s][150]
            st = ", ".join(f"{k} ({int(m.sum())}): {100 * pk(c, n, 16, np.flatnonzero(m)):.1f}" for k, m in strata.items())
            print(f"   {X} seed {s}: pass@1 {100 * pk(c, n, 1, full_idx):.1f}, pass@16 {100 * pk(c, n, 16, full_idx):.1f}, "
                  f"truncated {100 * d.get('truncated_rate', np.nan):.1f}%, boxed {100 * d.get('boxed_rate', np.nan) if d.get('boxed_rate') is not None else float('nan'):.1f}% | {st}")
    for s in SEEDS:
        c, n, _ = refs[s][0][-1][1]
        st = ", ".join(f"{k}: {100 * pk(c, n, 16, np.flatnonzero(m)):.1f}" for k, m in strata.items())
        print(f"   B1k seed {s} step 150: pass@1 {100 * pk(c, n, 1, full_idx):.1f}, pass@16 {100 * pk(c, n, 16, full_idx):.1f} | {st}")
    log = sorted(glob.glob(f"{S2}/OPD_R/opd_log.json"))
    if log:
        L = json.load(open(log[0]))
        meta = json.load(open(f"{S2}/OPD_R/start_25000/start_meta.json"))
        print(f"\nOPD training: {len(L)} steps, presented {meta['presented_tokens']} tokens; reverse KL per token "
              + ", ".join(f"{r['reverse_kl_per_token']:.3f}" for r in L) + "; batch reward " + ", ".join(f"{r['batch_reward']:.2f}" for r in L))


if __name__ == "__main__":
    main()
