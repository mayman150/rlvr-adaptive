"""Registered analyses of Pre-registrations 21 (1.5B, 1,024 tokens: B1k, PSR1k, NSR1k, AC1k), 22 (HG1k, HGm1k, PG1k)
and 23 (pilot: MH, AVH on validation problems), from the vLLM re-evaluation (results/worth-v15v-*/<NAME>/eval).
Usage: python analyze_v21.py [--draws 2000] [--which 21,22,23]
"""

import argparse
import glob
import json
import re

import numpy as np

import analyze_v12 as A
import matched as M

SEEDS = (8, 9, 10)


def load(pattern="results/worth-v15v-*/*/eval/step*_T*_eval.json"):
    out = {}
    for f in glob.glob(pattern):
        name = f.split("/")[-3]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--which", default="21,22,23")
    args = ap.parse_args()
    res = load()
    for w in args.which.split(","):
        {"21": p21, "22": p22, "23": p23}[w](res, args.draws)


if __name__ == "__main__":
    main()
