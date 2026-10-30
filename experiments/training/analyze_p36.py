"""Registered analysis of Pre-registration 36 (coverage-weighted self-SFT warm start) and Pre-registration 35's phase 2.
Written before any P36 or phase-2 run; the rules are those of PREREGISTRATION_36.md ("Estimator", "Readings",
"Decision").

Notation: a run X_s (RL seed s) has checkpoints 1, 2, 5, 10, 25, 50, 100, 150 (T = 1, 32 samples, 511 test problems,
1,024 tokens); a reference trajectory R_s is a list of evaluated models with the same RL seed (B1k_s: the base and
steps 25-150; B1kL_s: the base and steps 25-700; a baseline arm: its 64-sample start and its checkpoints 1-150; TIME:
the base and its checkpoints).
- Gap at a checkpoint with pass@1 u: pass@16 minus R_s's pass@16 linearly interpolated at u (R_s sorted by pass@1,
  ties by step). Eligible checkpoints fixed on the full data (u inside R_s's pass@1 range); at least two per seed,
  otherwise "insufficient overlap". Seed estimate: mean over eligible checkpoints; pooled: seeds weighted equally.
- Bootstrap: 2,000 index resamples of the 511 problems (seed 36), shared by every quantity; references recomputed in
  every draw. Fixed checkpoint set: a seed's draw is unavailable if any eligible checkpoint leaves R_s's range in that
  draw (and so is the pooled draw). More than 10% unavailable draws in a seed: "not interpreted (support)".
- Rule A ("curve"): raised if pooled >= +2, 95% interval above 0, every seed positive, Holm p <= 0.05; lowered: mirror.
- Rule B ("method"): beats if pooled >= +0.5, 95% interval above 0, every seed positive, Holm p <= 0.05 (test of
  zero); is beaten: mirror; otherwise inconclusive, with "practical average equivalence" noted (descriptive,
  unadjusted) if the pooled 90% interval lies within +/-1. Every registered seed pair is required.
- Families (Holm; an unavailable member enters with p = 1): A = {CWS, UNI, HARD, TAIL, TCWS, TIME} vs B1k;
  A' = the same vs B1kL (secondary); B = CWS vs {UNI, HARD, TAIL, HSELF}; C = TCWS vs CWS; D = HSELF vs TIME;
  E = {SELF, TEACHF, CORR, H25b} vs B1k; E' = TEACHF vs SELF. F (CWS vs OPD25) and every other contrast: descriptive.
Usage: python analyze_p36.py   (confirmatory: 2,000 draws; --draws N for exploratory runs only)
"""

import argparse
import glob
import json
from math import comb

import numpy as np

P = 511
STEPS = (1, 2, 5, 10, 25, 50, 100, 150)
SEEDS = (8, 9, 10)
SFT_SEED = {8: 3636, 9: 3737, 10: 3838, 11: 3939}
S1 = "results/worth-p30-stage1-s0-20261029-084113"
S1B = "results/worth-p30-stage1b-s0-20261029-164830"
S2 = "results/worth-p30-stage2-s0-20261029-182830"
OPD10 = "results/worth-p34-s10-s0-20261030-050310"
P35C = "results/worth-p35-controls-s0-20261030-172723"
CHECK2K = "results/worth-p30-check2k-s0-20261030-061010"
REF2K = "results/worth-p30-ref2k-s0-20261030-201539"
P36 = "results/worth-p36-job*"
_T = {}


def table(n, k):
    if (n, k) not in _T:
        _T[(n, k)] = np.array([1 - comb(n - x, k) / comb(n, k) if n - x >= k else 1.0 for x in range(n + 1)])
    return _T[(n, k)]


class Model:
    """One evaluation: per-problem pass@1 and pass@16 values, full-data means and per-draw means (cached)."""
    W = None  # draw-count matrix (draws x problems), set in main

    def __init__(self, path):
        d = json.load(open(path))
        self.c, self.n, self.meta, self.path = np.array(d["counts"], int), d["n"], d, path
        assert len(self.c) == P, (path, len(self.c))
        self.v = {1: self.c / self.n, 16: table(self.n, 16)[self.c]}
        self.full = {k: float(v.mean()) for k, v in self.v.items()}
        self._draws = {}

    def draws(self, k):
        if k not in self._draws:
            self._draws[k] = np.einsum("dp,p->d", Model.W, self.v[k]) / P  # einsum: no BLAS (spurious Accelerate warnings)
        return self._draws[k]


def one(pat, need=False):
    f = sorted(glob.glob(pat))
    if len(f) > 1:
        raise SystemExit(f"multiple matches for {pat}: {f}")
    if not f:
        if need:
            raise SystemExit(f"missing {pat}")
        return None
    return Model(f[0])


def ckpts(root, run, steps=STEPS, tag="_T1.00"):
    """{step: Model} for a run under root (a glob), or None if any checkpoint is missing."""
    out = {st: one(f"{root}/{run}/eval/step{st}{tag}_eval.json") for st in steps}
    return out if all(m is not None for m in out.values()) else None


def traj(start, points):
    """Reference trajectory: [(step, Model)] with the start at step 0."""
    return [(0, start)] + sorted(points.items())


def contrast(name, arm, ref, seeds):
    """arm: {s: {step: Model}}; ref: {s: [(step, Model)]}. Returns the summary dict of the registered estimator."""
    missing = [s for s in seeds if arm.get(s) is None or ref.get(s) is None]
    if missing:  # every registered seed pair is required; a partial contrast is not estimated
        return {"name": name, "status": f"unavailable (missing seeds {missing})"}
    seeds = list(seeds)
    D = Model.W.shape[0]
    est, per_draw, elig, ckpt, unav = {}, {}, {}, {}, {}
    for s in seeds:
        steps = sorted(arm[s])
        rx = np.array([m.full[1] for _, m in ref[s]])
        el = [st for st in steps if rx.min() <= arm[s][st].full[1] <= rx.max()]
        elig[s] = el
        if len(el) < 2:
            return {"name": name, "status": f"insufficient overlap (eligible steps {elig})", "elig": elig}
        order = sorted(range(len(ref[s])), key=lambda j: (rx[j], ref[s][j][0]))
        xs, ys = rx[order], np.array([ref[s][j][1].full[16] for j in order])
        ckpt[s] = {st: 100 * (arm[s][st].full[16] - np.interp(arm[s][st].full[1], xs, ys)) for st in el}
        est[s] = float(np.mean(list(ckpt[s].values())))
        RX = np.stack([m.draws(1) for _, m in ref[s]], 1)
        RY = np.stack([m.draws(16) for _, m in ref[s]], 1)
        U = np.stack([arm[s][st].draws(1) for st in el], 1)
        V = np.stack([arm[s][st].draws(16) for st in el], 1)
        tags = np.array([t for t, _ in ref[s]])
        out = np.full(D, np.nan)
        G = np.full((D, len(el)), np.nan)
        for d in range(D):
            lo, hi = RX[d].min(), RX[d].max()
            if U[d].min() < lo or U[d].max() > hi:
                continue
            o = np.lexsort((tags, RX[d]))
            g = 100 * (V[d] - np.interp(U[d], RX[d][o], RY[d][o]))
            G[d] = g
            out[d] = g.mean()
        per_draw[s], unav[s] = out, float(np.isnan(out).mean())
        ckpt[s] = {st: (ckpt[s][st], np.nanpercentile(G[:, j], [5, 95]) if np.isfinite(G[:, j]).any() else (np.nan, np.nan))
                   for j, st in enumerate(el)}
    pooled = np.mean(np.vstack([per_draw[s] for s in seeds]), 0)
    ok = pooled[np.isfinite(pooled)]
    t = {"name": name, "seeds": seeds, "elig": elig, "ckpt": ckpt, "unav": unav, "per_seed": est, "draws": pooled}
    if ok.size == 0:
        t["status"] = "not interpreted (support)"
        return t
    t.update({"est": float(np.mean(list(est.values()))), "ci90": np.percentile(ok, [5, 95]), "ci95": np.percentile(ok, [2.5, 97.5]),
              "per_seed_ci90": {s: np.nanpercentile(per_draw[s], [5, 95]) if np.isfinite(per_draw[s]).any() else (np.nan, np.nan) for s in seeds},
              "p": float(min(1.0, 2 * min((ok <= 0).mean(), (ok >= 0).mean())))})
    t["status"] = "not interpreted (support)" if any(x > 0.10 for x in unav.values()) else "ok"
    return t


def holm(ps):
    ps = np.asarray(ps, float)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(np.argsort(ps)):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def rule_a(t, q):
    if t.get("status") != "ok":
        return t.get("status", "unavailable")
    if t["est"] >= 2 and t["ci95"][0] > 0 and all(v > 0 for v in t["per_seed"].values()) and q <= 0.05:
        return "curve raised"
    if t["est"] <= -2 and t["ci95"][1] < 0 and all(v < 0 for v in t["per_seed"].values()) and q <= 0.05:
        return "curve lowered"
    return "inconclusive"


def rule_b(t, q, win="beats", lose="is beaten"):
    """Priority: the Holm-adjusted superiority verdicts first; equivalence is descriptive and unadjusted, reported only
    when neither superiority verdict holds."""
    if t.get("status") != "ok":
        return t.get("status", "unavailable")
    if t["est"] >= 0.5 and t["ci95"][0] > 0 and all(v > 0 for v in t["per_seed"].values()) and q <= 0.05:
        return win
    if t["est"] <= -0.5 and t["ci95"][1] < 0 and all(v < 0 for v in t["per_seed"].values()) and q <= 0.05:
        return lose
    if t["ci90"][0] >= -1 and t["ci90"][1] <= 1:
        return "inconclusive; practical average equivalence (descriptive, unadjusted)"
    return "inconclusive"


def show(t, v=None, q=None):
    if "est" not in t:
        return f"   {t['name']}: {t.get('status')}"
    ps = ", ".join(f"{s}: {t['per_seed'][s]:+.2f} [{t['per_seed_ci90'][s][0]:+.2f}, {t['per_seed_ci90'][s][1]:+.2f}]" for s in t["seeds"])
    un = ", ".join(f"{s}: {100 * x:.1f}%" for s, x in t["unav"].items())
    tail = f"; Holm p {q:.3f} -> {v}" if q is not None else f"; p {t['p']:.3f} (descriptive; status: {t['status']})"
    return (f"   {t['name']}: {t['est']:+.2f}, 90% [{t['ci90'][0]:+.2f}, {t['ci90'][1]:+.2f}], 95% [{t['ci95'][0]:+.2f}, "
            f"{t['ci95'][1]:+.2f}]; per seed (90%) {ps}; unavailable draws {un}{tail}")


def family(title, ts, rule):
    print(f"\n{title}")
    qs = holm([t["p"] if t.get("status") == "ok" else 1.0 for t in ts])
    verdicts = []
    for t, q in zip(ts, qs):
        v = rule(t, q)
        verdicts.append(v)
        print(show(t, v, q))
    return dict(zip([t["name"] for t in ts], verdicts))


def per_checkpoint(t):
    if "ckpt" not in t:
        return
    for s in t["seeds"]:
        row = "; ".join(f"{st}: {g:+.1f} [{b[0]:+.1f}, {b[1]:+.1f}]" for st, (g, b) in t["ckpt"][s].items())
        print(f"      seed {s}: {row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(36)
    W = np.zeros((a.draws, P))
    for d in range(a.draws):
        W[d] = np.bincount(rng.integers(0, P, P), minlength=P)
    Model.W = W
    print("Pre-registration 36 and Pre-registration 35 phase 2; 511 test problems, 1.5B, 1,024 tokens"
          + ("" if a.draws == 2000 else f" (EXPLORATORY: {a.draws} draws)"))
    base = one("results/worth-v15v-all-*/BASE15/eval/step0_T1.00_eval.json", need=True)
    base64 = one(f"{S1}/BASE_test/eval/step0_T1.00_eval.json", need=True)
    b1k = {s: traj(base, ckpts("results/worth-v15v-all-*", f"B1k{s}", (25, 50, 100, 150))) for s in SEEDS}
    long = {}
    for s in SEEDS:
        pts = ckpts(f"results/worth-p30-longref{s}-*", f"B1kL{s}", (25, 50, 100, 150, 200, 300, 400, 500, 600, 700))
        long[s] = traj(base, pts) if pts else None
    print(f"   B1kL available for seeds {[s for s in SEEDS if long[s]]}")

    # arms: {name: ({s: {step: Model}}, {s: start Model or None}, {s: run name})}
    arms = {}

    def add(name, runs, start_of):
        pts, st = {}, {}
        for s, (root, run) in runs.items():
            pts[s] = ckpts(root, run)
            st[s] = start_of(s)
        arms[name] = (pts, st, runs)

    for X in ("CWS", "UNI", "HARD", "TAIL", "TCWS"):
        seeds = (8, 9, 10, 11) if X in ("CWS", "UNI") else SEEDS
        add(X, {s: (P36, f"P36{X}{SFT_SEED[s]}s{s}") for s in seeds},
            lambda s, X=X: one(f"{P36}/P36{X}{SFT_SEED[s]}_start_1k/eval/step0_T1.00_eval.json"))
    add("TIME", {s: (P36, f"P36TIMEs{s}") for s in SEEDS}, lambda s: base)
    for X in ("SELF", "TEACHF", "CORR", "H25b"):
        add(X, {s: (P36, f"P35b{X}s{s}") for s in SEEDS}, lambda s, X=X: one(f"{P35C}/P35{X}_start_1k/eval/step0_T1.00_eval.json"))
    add("H25", {s: (S2, f"S2H25s{s}") for s in SEEDS}, lambda s: one(f"{S1B}/H025_test/eval/step0_T1.00_eval.json"))
    add("OPD25", {8: (S2, "P34OPD25s8"), 9: (S2, "P34OPD25s9"), 10: (OPD10, "P34OPD25s10")}, lambda s: one(f"{S2}/OPD25_test/eval/step0_T1.00_eval.json"))
    arms["HSELF"] = arms["SELF"]
    for X, (pts, st, runs) in arms.items():
        miss = [s for s in pts if pts[s] is None or st[s] is None]
        if miss:
            print(f"   {X}: missing runs or starts for seeds {miss} (treated as unavailable)")
            for s in miss:
                pts[s] = None

    def P_(X):
        return {s: m for s, m in arms[X][0].items() if m is not None}

    def R_(X):  # an arm's own trajectory as a reference
        return {s: traj(arms[X][1][s], arms[X][0][s]) for s in arms[X][0] if arms[X][0][s] is not None}

    # starts
    print("\nStarts (64 samples; TIME starts from the base): pass@1, pass@16, truncation; start minus B1k_s at its pass@1")
    for X in ("CWS", "UNI", "HARD", "TAIL", "TCWS", "SELF", "TEACHF", "CORR", "H25b", "H25", "OPD25"):
        for s, m in arms[X][1].items():
            if m is None:
                continue
            r = []
            for s2 in SEEDS:
                rx = sorted((mm.full[1], t, mm.full[16]) for t, mm in b1k[s2])
                xs, ys = np.array([x[0] for x in rx]), np.array([x[2] for x in rx])
                r.append(f"{100 * (m.full[16] - np.interp(m.full[1], xs, ys)):+.2f}" if xs[0] <= m.full[1] <= xs[-1] else "n/a")
            print(f"   {X} (run seed {s}): pass@1 {100 * m.full[1]:.1f}, pass@16 {100 * m.full[16]:.1f}, "
                  f"truncated {100 * m.meta.get('truncated_rate', np.nan):.1f}%; vs B1k8/9/10 {', '.join(r)}")
            if X in ("SELF", "TEACHF", "CORR", "H25b", "H25", "OPD25"):
                break  # one start shared by the RL seeds

    # A and A'
    A_ARMS = ("CWS", "UNI", "HARD", "TAIL", "TCWS", "TIME")
    tA = [contrast(f"{X} vs B1k", P_(X), b1k, SEEDS) for X in A_ARMS]
    vA = family("A. Each arm against Dr. GRPO (B1k_s; Holm over six; rule A)", tA, rule_a)
    if any(long.values()):
        tA2 = [contrast(f"{X} vs B1kL", P_(X), {s: long[s] for s in SEEDS if long[s]}, SEEDS) for X in A_ARMS]
        family("A'. Each arm against the long replicates B1kL_s (secondary; Holm over six; rule A)", tA2, rule_a)
    # B
    tB = [contrast("CWS vs UNI trajectory", P_("CWS"), R_("UNI"), (8, 9, 10, 11))] + \
         [contrast(f"CWS vs {X} trajectory", P_("CWS"), R_(X), SEEDS) for X in ("HARD", "TAIL", "HSELF")]
    vB = family("B. Method against baselines (primary; Holm over four; rule B)", tB, rule_b)
    # C, D
    tC = contrast("TCWS vs CWS trajectory", P_("TCWS"), R_("CWS"), SEEDS)
    family("C. Teacher versus self (family of one; rule B)", [tC], rule_b)
    TIME_R = {s: traj(base, arms["TIME"][0][s]) for s in SEEDS if arms["TIME"][0].get(s) is not None}
    tD = contrast("HSELF vs TIME trajectory", P_("HSELF"), TIME_R, SEEDS)
    family("D. Timing (family of one; rule B: 'beats' = before RL is better)", [tD],
           lambda t, q: rule_b(t, q, "before RL is better", "during RL is better"))
    post = {}
    for s in SEEDS:
        f = sorted(glob.glob(f"{P36}/P36TIMEs{s}/baseline_train.json"))
        if f and TIME_R.get(s):
            log = json.load(open(f[0]))
            done = [r["step"] for r in log if r.get("isft", {}).get("presented", 0) >= 25000]
            if done:
                post[s] = [(st, m) for st, m in TIME_R[s] if st > min(done)]
                print(f"   TIME seed {s}: interleaved dose complete at logged step {min(done)}; post-dose reference checkpoints {[st for st, _ in post[s]]}")
    if post:
        print(show(contrast("HSELF vs TIME's post-dose checkpoints (descriptive)", P_("HSELF"), post, SEEDS)))
    # E and E'
    tE = [contrast(f"{X} vs B1k", P_(X), b1k, SEEDS) for X in ("SELF", "TEACHF", "CORR", "H25b")]
    family("E. Pre-registration 35 phase 2 (Holm over four; rule A)", tE, rule_a)
    tE2 = contrast("TEACHF vs SELF trajectory", P_("TEACHF"), R_("SELF"), SEEDS)
    family("E'. Teacher content (family of one; rule B)", [tE2], rule_b)
    tH = contrast("H25 vs B1k (Stage 2 runs, this estimator; descriptive)", P_("H25"), b1k, SEEDS)
    print(show(tH))
    for t in tE[2:]:
        if t.get("status") == "ok" and tH.get("status") == "ok":
            d = t["draws"] - tH["draws"]
            d = d[np.isfinite(d)]
            print(f"   {t['name'].split()[0]} minus H25: {t['est'] - tH['est']:+.2f}, 90% [{np.percentile(d, 5):+.2f}, {np.percentile(d, 95):+.2f}] "
                  f"(descriptive; registered reading: within 0.5 points)")
    # F and other descriptive contrasts
    print("\nDescriptive contrasts (no verdicts)")
    for t in (contrast("F. CWS vs OPD25 trajectory", P_("CWS"), R_("OPD25"), SEEDS),
              contrast("HARD vs UNI trajectory", P_("HARD"), R_("UNI"), SEEDS),
              contrast("CWS vs H25 trajectory", P_("CWS"), R_("H25"), SEEDS),
              contrast("TCWS vs H25 trajectory", P_("TCWS"), R_("H25"), SEEDS)):
        print(show(t))

    print("\nPer-checkpoint gaps (pointwise 90% bands) for the primary contrasts")
    for t in tA + tB + [tD]:
        print(f"   {t['name']}: {t.get('status')}")
        per_checkpoint(t)

    # 2,048-token diagnostics
    base2k, lab = one(f"{P36}/BASE_test_2k64/eval/step0_T1.00_eval.json"), "registered"
    if base2k is None:  # not the registered reference: the 32-sample base, exploratory only
        base2k, lab = one(f"{CHECK2K}/BASE15_2k/eval/step0_T1.00_eval.json"), "EXPLORATORY fallback"
    r2k = {8: ckpts(CHECK2K, "B1k8_2k", (25, 50, 100, 150)), 9: ckpts(REF2K, "B1k9_2k", (25, 50, 100, 150)),
           10: ckpts(REF2K, "B1k10_2k", (25, 50, 100, 150))}
    r2k = {s: traj(base2k, p) for s, p in r2k.items() if p and base2k}
    print(f"\n2,048 tokens (descriptive; {lab} base {base2k.path if base2k else 'missing'}, n = {base2k.n if base2k else 0}): checkpoints 50-150 vs B1k_s at 2,048")
    for X in A_ARMS + ("SELF", "TEACHF", "CORR", "H25b"):
        runs = arms[X][2]
        pts = {s: ckpts(root, run + "_2k", (50, 100, 150)) for s, (root, run) in runs.items() if s in SEEDS}
        print(show(contrast(f"{X} at 2,048", {s: p for s, p in pts.items() if p}, r2k, SEEDS)))

    # step-150 strata and SFT accounting
    c0 = base64.c
    strata = {"c=0": c0 == 0, "c=1-8": (c0 >= 1) & (c0 <= 8), "c>=9": c0 >= 9}
    print("\nStep 150: pass@1, pass@16, truncated, boxed | pass@16 by base-count stratum (Stage 1 base, 64 samples)")
    for X in A_ARMS + ("SELF", "TEACHF", "CORR", "H25b"):
        for s, p in arms[X][0].items():
            if p is None:
                continue
            m = p[150]
            st = ", ".join(f"{k} ({int(mk.sum())}): {100 * m.v[16][mk].mean():.1f}" for k, mk in strata.items())
            print(f"   {X} seed {s}: {100 * m.full[1]:.1f}, {100 * m.full[16]:.1f}, {100 * m.meta.get('truncated_rate', np.nan):.1f}%, "
                  f"{100 * m.meta.get('boxed_rate', np.nan):.1f}% | {st}")
    for s in SEEDS:
        m = b1k[s][-1][1]
        print(f"   B1k seed {s}: {100 * m.full[1]:.1f}, {100 * m.full[16]:.1f} | " +
              ", ".join(f"{k}: {100 * m.v[16][mk].mean():.1f}" for k, mk in strata.items()))
    print("\nSFT starts: presented / retained tokens, updates, problems, solutions, unique solutions seen, visits")
    for f in sorted(glob.glob(f"{P36}/SFT36_*/start_25000/start_meta.json")):
        m = json.load(open(f))
        print(f"   {f.split('/')[-3]}: {m['presented_tokens']} / {m['retained_tokens']}, {m['steps']}, {m['problems']}, "
              f"{m['solutions']}, {m['unique_solutions_seen']}, {m['problem_visits']}")

    # decision (week 4)
    print("\nDecision (PREREGISTRATION_36.md)")
    win = {X: vB.get(f"CWS vs {X} trajectory") for X in ("UNI", "HARD", "TAIL", "HSELF")}
    beaten = any(v == "is beaten" for v in win.values())
    aU, aC = tA[1], tA[0]
    both = all(t.get("status") == "ok" and t["est"] >= 1 and t["ci95"][0] > 0 for t in (aC, aU))
    if not all(t.get("status") == "ok" for t in tB + [aC, aU]):
        print("   decision unavailable: a decision-relevant contrast (B, or A for CWS/UNI) is not interpretable")
    elif win["UNI"] == "beats" and win["HARD"] == "beats" and not beaten:
        print("   method paper: coverage-weighted self-SFT before RL")
    elif win["UNI"] == "beats" and win["HARD"] != "beats" and not beaten:
        print("   hard-problem self-SFT before RL (weighting not credited)")
    elif win["UNI"] != "beats" and both:
        print("   self-SFT before RL, without a demonstrated targeting benefit")
    else:
        print("   measurement paper on coverage at matched accuracy")
    print(f"   (B verdicts: {win}; A verdicts: {vA})")


if __name__ == "__main__":
    main()
