"""Registered analyses of Pre-registrations 21 (1.5B, 1,024 tokens: B1k, PSR1k, NSR1k, AC1k), 22 (HG1k, HGm1k, PG1k)
and 23 (pilot: MH, AVH on validation problems), from the vLLM re-evaluation (results/worth-v15v-*/<NAME>/eval),
and 26 (pilots OD, PW, TD, RTD; the par/perp/other checkpoints), from results/worth-p26-pack-*/<NAME>/eval.
Usage: python analyze_v21.py [--draws 2000] [--which 21,22,23,26,26a]
"""

import argparse
import glob
import json
import re

import numpy as np

import analyze_v12 as A
import matched as M

SEEDS = (8, 9, 10)


PACK = {"OD": "OD8val", "PW": "PW8val", "TD": "TD8val", "RTD": "RTD8val"}  # Pre-registration 26 pilots (validation)


def load(patterns=("results/worth-v15v-*/*/eval/step*_T*_eval.json", "results/worth-p26-pack-*/*/eval/step*_T*_eval.json",
                   "results/worth-p31-flat-*/FL*/eval/step*_T*_eval.json")):
    out = {}
    for f in (f for pat in patterns for f in sorted(glob.glob(pat))):
        name = f.split("/")[-3]
        if "worth-p26-pack-" in f:
            name = PACK.get(name, name)
        step, t = re.search(r"step(\d+)_T([\d.]+)_eval", f).groups()
        d = json.loads(open(f).read())
        out[(name, int(step), round(float(t), 2))] = d
    return out


def counts(res, name, step, t=1.0):
    d = res.get((name, step, t))
    return None if d is None else np.array(d["counts"])


def traj(res, name, steps, base):
    def f(seed):
        pts = [(counts(res, "BASE15" + base, 0), 0)]
        pts += [(counts(res, f"{name}{seed}{base and 'val'}", st), st) for st in steps if counts(res, f"{name}{seed}{base and 'val'}", st) is not None]
        return pts
    return f


def temps(res, name, base):
    def f(seed):
        nm = f"{name}{seed}{base and 'val'}"
        return [(counts(res, nm, 150, t), t) for t in (1.0, 1.15, 1.3) if counts(res, nm, 150, t) is not None]
    return f


def arm_points(res, name, steps, base=""):
    return lambda seed: [(st, counts(res, f"{name}{seed}{base and 'val'}", st)) for st in steps if counts(res, f"{name}{seed}{base and 'val'}", st) is not None]


def missing(res, need):
    return [k for k in need if k not in res]


def trunc(res, name, seed, step=150, base=""):
    d = res.get((f"{name}{seed}{base and 'val'}", step, 1.0))
    return None if d is None else d.get("truncated_rate")


def qualifier(res, arms, seeds, limit=0.20):
    """Pre-registration 21's budget qualifier for one comparison: any of its arms above 20% truncation at step 150."""
    over = sorted({a for a in arms for s in seeds if (trunc(res, a, s) or 0) > limit})
    return f"; qualified: truncation still substantial at 1,024 ({', '.join(over)})" if over else ""


def verdict_eq(t, h, key):
    if "estimate" not in t:
        return t["verdict"]
    if not M.clamped_ok(t):
        return "not interpreted (more than 10% clamped)"
    return "average equivalence" if M.equivalent(t) else "departure" if M.departure(t, h[key][1]) else "inconclusive"


def verdict_sup(t, h, key, null="average equivalence"):
    if "estimate" not in t:
        return t["verdict"]
    if not M.clamped_ok(t):
        return "not interpreted (more than 10% clamped)"
    return "moves the curve (fresh confirmation required)" if M.moves(t, h[key][1]) else null if M.equivalent(t) else "inconclusive"


B_NEED = [("BASE15", 0, 1.0)] + [(f"B1k{s}", st, 1.0) for s in SEEDS for st in (25, 50, 100, 150)] + \
         [(f"B1k{s}", 150, t) for s in SEEDS for t in (1.15, 1.3)]


def p21(res, draws):
    print("Pre-registration 21 (1,024 tokens; vLLM evaluation)")
    need = B_NEED + [(f"{a}{s}", st, 1.0) for a, steps in (("PSR1k", (25, 50, 100, 150)), ("NSR1k", (50, 100, 150)), ("AC1k", (50, 100, 150)))
                     for s in SEEDS for st in steps]
    miss = missing(res, need)
    if miss:
        print(f"   incomplete evaluation; missing {miss}")
        return
    n = len(counts(res, "BASE15", 0))
    B = M.reference(traj(res, "B1k", (25, 50, 100, 150), ""))
    Benv = M.reference(traj(res, "B1k", (25, 50, 100, 150), ""), temps(res, "B1k", ""))
    PSR = M.reference(traj(res, "PSR1k", (25, 50, 100, 150), ""))
    S3 = (50, 100, 150)
    spec = {"NSR1k vs PSR1k trajectory": ("NSR1k", PSR, ("NSR1k", "PSR1k")), "PSR1k vs B1k trajectory": ("PSR1k", B, ("PSR1k", "B1k")),
            "NSR1k vs B1k trajectory": ("NSR1k", B, ("NSR1k", "B1k")), "AC1k vs B1k envelope": ("AC1k", Benv, ("AC1k", "B1k"))}
    tests = {k: M.compare(arm_points(res, a, S3), r, SEEDS, n, draws) for k, (a, r, _) in spec.items()}
    h = M.holm(tests)
    joint = True
    for k, t in tests.items():
        v = verdict_sup(t, h, k, "average equivalence (replicates Pre-registration 18's null)") if k.startswith("AC1k") else verdict_eq(t, h, k)
        joint = joint and v.startswith("average equivalence")
        print(M.show(k, t, v + f" (Holm-adjusted p {h[k][0]:.3f})" + qualifier(res, spec[k][2], SEEDS)))
    print(f"   joint statement (all four at average equivalence): {'the core result replicates at 1,024 tokens' if joint else 'not established'}")
    print(M.show("AC1k vs B1k trajectory (reported)", M.compare(arm_points(res, "AC1k", S3), B, SEEDS, n, draws)))
    print("   step-150 truncation: " + "; ".join(f"{a} " + ", ".join(f"{100 * trunc(res, a, s):.1f}%" for s in SEEDS) for a in ("B1k", "PSR1k", "NSR1k", "AC1k")))
    B3 = M.reference(traj(res, "B1k", (50, 100, 150), ""))
    PSR3 = M.reference(traj(res, "PSR1k", (50, 100, 150), ""))
    for k, (a, r) in {"PSR1k vs B1k (no step-25 knot)": ("PSR1k", B3), "NSR1k vs B1k (no step-25 knot)": ("NSR1k", B3),
                      "NSR1k vs PSR1k (no step-25 knot)": ("NSR1k", PSR3)}.items():
        print(M.show(k, M.compare(arm_points(res, a, S3), r, SEEDS, n, draws), "sensitivity"))
    g = {a: [res[(f"{a}{s}", 150, 1.0)].get("greedy", {}).get("accuracy") for s in SEEDS] for a in ("B1k", "PSR1k", "NSR1k", "AC1k")}
    print("   greedy test accuracy at step 150: " + "; ".join(f"{a} " + ", ".join("-" if x is None else f"{100 * x:.1f}" for x in v) for a, v in g.items()))


def p22(res, draws):
    print("Pre-registration 22 (guidance; vLLM evaluation)")
    need = B_NEED + [(f"{a}{s}", st, 1.0) for a in ("HG1k", "HGm1k") for s in SEEDS for st in (50, 100, 150)] + \
           [(f"PG1k{s}", st, 1.0) for s in SEEDS for st in (25, 50, 100, 150)]
    miss = missing(res, need)
    if miss:
        print(f"   incomplete evaluation; missing {miss}")
        return
    n = len(counts(res, "BASE15", 0))
    Benv = M.reference(traj(res, "B1k", (25, 50, 100, 150), ""), temps(res, "B1k", ""))
    PG = M.reference(traj(res, "PG1k", (25, 50, 100, 150), ""))
    S3 = (50, 100, 150)
    tests = {"HG1k vs B1k envelope": M.compare(arm_points(res, "HG1k", S3), Benv, SEEDS, n, draws),
             "HGm1k vs PG1k trajectory": M.compare(arm_points(res, "HGm1k", S3), PG, SEEDS, n, draws)}
    h = M.holm(tests)
    k = "HG1k vs B1k envelope"
    v = verdict_sup(tests[k], h, k, "on the curve (average equivalence)")
    print(M.show("rule 1: " + k, tests[k], v + qualifier(res, ("HG1k", "B1k"), SEEDS)))
    k = "HGm1k vs PG1k trajectory"
    t = tests[k]
    if "estimate" not in t:
        v = t["verdict"]
    elif not M.clamped_ok(t):
        v = "not interpreted (more than 10% clamped)"
    else:
        ok = t["estimate"] >= 1 and t["ci95"][0] > 0 and len({np.sign(x) for x in t["per_seed"].values()}) == 1 and h[k][1]
        v = "source matters (provisional until fresh confirmation)" if ok else "not shown"
    print(M.show("rule 2: " + k, t, v + qualifier(res, ("HGm1k", "PG1k"), SEEDS)))
    for a, r in (("PG1k", Benv), ("HGm1k", Benv)):
        print(M.show(f"{a} vs B1k envelope (reported)", M.compare(arm_points(res, a, S3), r, SEEDS, n, draws)))
    w = np.ones(n)
    for a in ("HG1k", "HGm1k", "PG1k"):  # during guidance (step 50) and 50 steps after it ends (step 150), point gaps
        row = []
        for st in (50, 150):
            g = []
            for s in SEEDS:
                value, lo, hi = Benv(s, w)
                p1, p16 = A.met(counts(res, f"{a}{s}", st), w)
                g.append(p16 - value(p1) if lo <= p1 <= hi else float("nan"))
            row.append(f"step {st}: " + ", ".join(f"{x:+.2f}" for x in g))
        print(f"   {a} gap to B1k's envelope by seed (reported): " + "; ".join(row))


def p23(res, draws):
    print("Pre-registration 23 (pilot, seed 8, validation problems; vLLM evaluation)")
    need = [("BASE15val", 0, 1.0)] + [(f"{a}8val", st, 1.0) for a in ("MH", "AVH") for st in (50, 100, 150)] + \
           [("B1k8val", st, 1.0) for st in (25, 50, 100, 150)] + [("B1k8val", 150, t) for t in (1.15, 1.3)]
    miss = missing(res, need)
    if miss:
        print(f"   incomplete evaluation; missing {miss}")
        return
    n = len(counts(res, "BASE15val", 0))
    seeds = (8,)
    Benv = M.reference(traj(res, "B1k", (25, 50, 100, 150), "val"), temps(res, "B1k", "val"))
    w = np.ones(n)
    gates = {}
    for arm in ("MH", "AVH"):
        t = M.compare(arm_points(res, arm, (50, 100, 150), "val"), Benv, seeds, n, draws)
        p1_150 = A.met(counts(res, f"{arm}8val", 150), w)[0] - A.met(counts(res, "B1k8val", 150), w)[0]
        f = glob.glob(f"results/worth-v23-{arm}-s8-*/{arm}/baseline_train.json")
        log = json.loads(open(f[0]).read()) if f else None
        if log is None:
            print(f"   {arm}: training log missing; selection blocked")
            gates[arm] = None
            continue
        mh = [r.get("mh", {}) for r in log if r["step"] < 100]
        credited = sum(m.get("credited", 0) for m in mh)
        probed = sum(m.get("groups", 0) for m in mh)  # problem-steps whose prefixes were probed
        contrast = sum(m.get("contrast_groups", 0) for m in mh)
        dose_ok = credited >= 150
        share = contrast / max(probed, 1)
        interp = "estimate" in t and M.clamped_ok(t)
        q = interp and dose_ok and t["estimate"] >= 1.5 and p1_150 >= -2 and share >= 0.30
        gates[arm] = q if dose_ok else "inconclusive: insufficient credit"
        print(M.show(f"{arm} vs B1k envelope", t) + f"; step-150 pass@1 minus B1k {p1_150:+.2f}")
        print(f"   {arm} gates: credited prefixes (steps 0-99) {credited} (floor 150: {'met' if dose_ok else 'not met'}); contrast in "
              f"{contrast} of {probed} probed problem-steps ({100 * share:.0f}%; gate 30%; credit is given only where values differ, so "
              f"the share of probed problem-steps is the operative reading); interpretable: {interp} -> qualifies: {gates[arm]}")
    AVHtraj = M.reference(lambda s: [(counts(res, "BASE15val", 0), 0)] + [(counts(res, "AVH8val", st), st) for st in (50, 100, 150)])
    d = M.compare(arm_points(res, "MH", (50, 100, 150), "val"), AVHtraj, seeds, n, draws)
    print(M.show("MH vs AVH trajectory", d))
    if None in gates.values():
        print("   SELECTION: blocked (missing training log)")
    elif isinstance(gates.get("MH"), str) or isinstance(gates.get("AVH"), str):
        print(f"   SELECTION: {gates}")
    elif gates["MH"] and "estimate" in d and d["estimate"] >= 0.5:
        print("   SELECTION: MH goes to the registered three-seed test")
    elif gates["AVH"]:
        print("   SELECTION: AVH (the averaged-value control) is the candidate instead")
    else:
        print("   SELECTION: no candidate; the method is dropped and reported")


def p26_log(arm):
    f = glob.glob(f"results/worth-p26-pack-*/{arm}/baseline_train.json")
    return json.loads(open(f[0]).read()) if f else None


def p26_gates(arm, log, warmup=10):
    """Pre-registration 26's mechanism gates from the training log; steps without the method's record count as
    inactive (share 0, cosine 1, alpha ratio 1, removed share 0)."""
    steps = [r for r in log if "step" in r]
    if arm == "OD":
        late = [r for r in steps if r["step"] >= warmup]
        share = np.mean([r.get("od", {}).get("dominant_share_before", 0.0) for r in late])
        cos = np.mean([r.get("od", {}).get("realized_cosine", 1.0) if r.get("od", {}).get("active") else 1.0 for r in late])
        return share >= 0.05 and cos <= 0.98, f"mean dominant share {share:.3f} (gate 0.05), mean realized cosine {cos:.4f} (gate 0.98)"
    if arm == "PW":
        def ratio(x):
            if "alpha_max" not in x:
                return 1.0
            return np.inf if x["alpha_min"] <= 0 < x["alpha_max"] else x["alpha_max"] / x["alpha_min"]
        frac = np.mean([ratio(r.get("pw", {})) > 1.5 for r in steps])
        cos = np.mean([r.get("pw", {}).get("realized_cosine", 1.0) for r in steps])
        return frac >= 0.5 and cos <= 0.98, f"steps with alpha max/min > 1.5: {100 * frac:.0f}% (gate 50%), mean realized cosine {cos:.4f} (gate 0.98)"
    if arm == "RTD":
        late = [r for r in steps if r["step"] >= warmup]
        rem = np.mean([r.get("teacher", {}).get("removed_share", 0.0) for r in late])
        return rem >= 0.05, f"mean removed share of the teacher gradient {rem:.3f} (gate 0.05)"
    return True, "no mechanism gate"


def strata(res, arm, base, ref, step=150):
    """pass@16 minus the reference's at step 150, by the base model's success count on each problem (of 32)."""
    c0 = counts(res, base, 0)
    a, b = counts(res, arm, step), counts(res, ref, step)
    out = []
    for lab, m in (("c=0", c0 == 0), ("c=1-4", (c0 >= 1) & (c0 <= 4)), ("c>=5", c0 >= 5)):
        if m.sum():
            out.append(f"{lab} ({int(m.sum())}): {A.met(a[m], np.ones(m.sum()))[1] - A.met(b[m], np.ones(m.sum()))[1]:+.1f}")
    return ", ".join(out)


def p26(res, draws):
    print("Pre-registration 26 (pilots, seed 8, validation problems; vLLM evaluation)")
    arms = ("OD", "PW", "TD", "RTD")
    need = [("BASE15val", 0, 1.0)] + [(f"{a}8val", st, 1.0) for a in arms for st in (50, 100, 150)] + \
           [("B1k8val", st, 1.0) for st in (25, 50, 100, 150)] + [("B1k8val", 150, t) for t in (1.15, 1.3)]
    miss = missing(res, need)
    if miss:
        print(f"   incomplete evaluation; missing {miss}")
        return
    n = len(counts(res, "BASE15val", 0))
    seeds, w = (8,), np.ones(n)
    Benv = M.reference(traj(res, "B1k", (25, 50, 100, 150), "val"), temps(res, "B1k", "val"))
    qual, est = {}, {}
    for arm in arms:
        t = M.compare(arm_points(res, arm, (50, 100, 150), "val"), Benv, seeds, n, draws)
        p1 = A.met(counts(res, f"{arm}8val", 150), w)[0] - A.met(counts(res, "B1k8val", 150), w)[0]
        log = p26_log(arm)
        if log is None:
            print(f"   {arm}: training log missing; selection blocked")
            qual[arm] = None
            continue
        g_ok, g_txt = p26_gates(arm, log)
        interp = "estimate" in t and M.clamped_ok(t)
        est[arm] = t.get("estimate")
        qual[arm] = bool(interp and t["estimate"] >= 1.5 and p1 >= -2 and g_ok)
        print(M.show(f"{arm} vs B1k envelope", t) + f"; step-150 pass@1 minus B1k {p1:+.2f}")
        extra = ""
        if arm in ("TD", "RTD"):
            sol = [r.get("teacher", {}).get("solutions", 0) for r in log if "step" in r]
            extra = f"; teacher solutions per step {np.mean(sol):.2f}"
        print(f"   {arm} gates: {g_txt}{extra}; interpretable: {interp} -> qualifies: {qual[arm]}")
        print(f"   {arm} pass@16 minus B1k at step 150 by base count: {strata(res, f'{arm}8val', 'BASE15val', 'B1k8val')}")
    TDtraj = M.reference(lambda s: [(counts(res, "BASE15val", 0), 0)] + [(counts(res, "TD8val", st), st) for st in (50, 100, 150)])
    d = M.compare(arm_points(res, "RTD", (50, 100, 150), "val"), TDtraj, seeds, n, draws)
    print(M.show("RTD vs TD trajectory", d))
    if qual.get("RTD"):
        qual["RTD"] = "estimate" in d and M.clamped_ok(d) and d["estimate"] >= 1.0
        print(f"   RTD beats TD by at least +1: {qual['RTD']}")
    if None in qual.values():
        print("   SELECTION: blocked (missing training log)")
        return
    chosen = [a for a in arms if qual[a]]
    print(f"   SELECTION: {', '.join(chosen) + ' qualify for the three-seed test' if chosen else 'no arm qualifies; the methods are reported as failed pilots'}")


def p26_split(res, draws):
    """Step 1(a), reported: the par/perp/other checkpoints of B1k8's step-150 update on the test problems."""
    print("Pre-registration 26, step 1(a): par / perp / other (test problems, seed 8)")
    need = [("BASE15", 0, 1.0)] + [(x, 150, 1.0) for x in ("B1k8_par", "B1k8_perp", "B1k8_other")] + \
           [("B1k8", st, 1.0) for st in (25, 50, 100, 150)] + [("B1k8", 150, t) for t in (1.15, 1.3)]
    miss = missing(res, need)
    if miss:
        print(f"   incomplete evaluation; missing {miss}")
        return
    n = len(counts(res, "BASE15", 0))
    w = np.ones(n)
    for x in ("BASE15", "B1k8_other", "B1k8_par", "B1k8_perp", "B1k8"):
        st = 0 if x == "BASE15" else 150
        p1, p16 = A.met(counts(res, x, st), w)
        print(f"   {x}: pass@1 {p1:.2f}, pass@16 {p16:.2f}")
    c_full, c_par, c_oth = (counts(res, x, 150) for x in ("B1k8", "B1k8_par", "B1k8_other"))
    rng = np.random.default_rng(26)
    gains, recov = [], []
    for _ in range(draws):
        i = rng.integers(0, n, n)
        g = (c_full[i] - c_oth[i]).mean() / 32 * 100
        gains.append(g)
        recov.append((c_par[i] - c_oth[i]).mean() / 32 * 100 / g if g > 0 else np.nan)
    g0 = (c_full - c_oth).mean() / 32 * 100
    r0 = (c_par - c_oth).mean() / 32 * 100 / g0 if g0 > 0 else float("nan")
    lo, hi = np.percentile(gains, [5, 95])
    print(f"   B1k8 pass@1 gain over 'other' {g0:+.2f}, 90% [{lo:+.2f}, {hi:+.2f}]; par recovers {100 * r0:.0f}% "
          f"(90% [{100 * np.nanpercentile(recov, 5):.0f}%, {100 * np.nanpercentile(recov, 95):.0f}%]; registered reading needs >= 80% "
          f"and a positive gain interval; par is distorted by bf16 rounding, see RESULTS_26_interim)")
    Benv8 = M.reference(traj(res, "B1k", (25, 50, 100, 150), ""), temps(res, "B1k", ""))
    t = M.compare(lambda s: [(150, counts(res, "B1k8_perp", 150))], Benv8, (8,), n, draws)
    print(M.show("perp vs B1k8 envelope (reported)", t))


def p31(res, draws):
    """Pre-registration 31: Flat-RLVR's SAM step on Dr. GRPO (FL10: every 10 steps; FL5: every 5), seeds 8-10, against
    B1k's trajectory-and-temperature envelope (primary) and trajectory (reported); Holm over the two arms."""
    print("Pre-registration 31 (Flat-RLVR on the matched-pass@1 curve; 1.5B, 1,024 tokens; vLLM evaluation)")
    S = (25, 50, 100, 150)
    need = B_NEED + [(f"{a}s{sd}", st, 1.0) for a in ("FL10", "FL5") for sd in SEEDS for st in S]
    miss = missing(res, need)
    if miss:
        print(f"   incomplete evaluation; missing {miss}")
        return
    n = len(counts(res, "BASE15", 0))
    Benv = M.reference(traj(res, "B1k", (25, 50, 100, 150), ""), temps(res, "B1k", ""))
    B = M.reference(traj(res, "B1k", (25, 50, 100, 150), ""))
    tests = {f"{a} vs B1k envelope": M.compare(arm_points(res, f"{a}s", S), Benv, SEEDS, n, draws) for a in ("FL10", "FL5")}
    h = M.holm(tests)
    w = np.ones(n)
    for k, t in tests.items():
        a = k.split()[0]
        if "estimate" in t and M.clamped_ok(t):
            v = "moves the curve" if M.moves(t, h[k][1]) else "average equivalence" if M.equivalent(t) else "inconclusive"
        else:
            v = t.get("verdict", "not interpreted (more than 10% clamped)")
        p1 = np.mean([A.met(counts(res, f"{a}s{sd}", 150), w)[0] - A.met(counts(res, f"B1k{sd}", 150), w)[0] for sd in SEEDS])
        print(M.show(k, t, v + f" (Holm-adjusted p {h[k][0]:.3f}); step-150 pass@1 minus B1k {p1:+.2f}"))
        print(M.show(f"{a} vs B1k trajectory (reported)", M.compare(arm_points(res, f"{a}s", S), B, SEEDS, n, draws)))
        sam = []
        for sd in SEEDS:
            f = glob.glob(f"results/worth-p31-flat-*/{a}s{sd}/baseline_train.json")
            if f:
                sam += [r["sam"] for r in json.loads(open(f[0]).read()) if r.get("sam", {}).get("applied")]
        if sam:
            print(f"   {a} SAM steps applied: {len(sam)} over 3 seeds; mean radius {np.mean([x['rho_eff'] for x in sam]):.4f}; "
                  f"max |ratio - 1| {max(x['max_abs_ratio_minus_1'] for x in sam):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--which", default="21,22,23")
    args = ap.parse_args()
    res = load()
    for w in args.which.split(","):
        {"21": p21, "22": p22, "23": p23, "26": p26, "26a": p26_split, "31": p31}[w](res, args.draws)


if __name__ == "__main__":
    main()
