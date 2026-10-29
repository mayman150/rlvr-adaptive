"""Registered analysis of Pre-registration 33 (sensitivity of Pre-registration 28's 7B readings).

References per seed s in {8, 9} (pass@16 interpolated in pass@1 along points sorted by pass@1, ties by step; maximum
with the step-150 temperature curve; mean over the seeds that cover the pass@1):
  sparse  = base, B7 25, 50, 100, 150                     (Pre-registration 28's reference A)
  dense   = base, B7e 3, 6, 12, 25, B7 50, 100, 150        (primary)
  splice  = base, B7e 3, 6, 12, B7 25, 50, 100, 150        (sensitivity)
  early   = base, B7e 3, 6, 12, 25 (no temperature curve)  (sensitivity, where it covers)
Families (Holm within each; Pre-registration 28's verdict rules and guards): A1 = 511 problems vs dense; A2 = hard
cohort vs dense (hard-cohort evaluations); B2 = hard cohort vs the frozen law (tau = -0.20) from the base's hard counts.
Bootstrap: 2,000 problem draws per cohort shared by every comparison on it; references and the law recomputed per draw.
Usage: python analyze_p33.py [--draws 2000]
"""

import argparse
import glob
import json
from math import comb

import numpy as np

import law
from analyze_p28 import holm, label

SEEDS = (8, 9)


def counts(f):
    d = json.load(open(f))
    return np.array(d["counts"]), d["n"], d


_PK = {}


def pk(c, n, k):
    """Unbiased pass@k from per-problem counts (a cached per-count table; the same values as the direct formula)."""
    if (n, k) not in _PK:
        _PK[(n, k)] = np.array([1 - comb(n - x, k) / comb(n, k) for x in range(n + 1)])
    return float(np.mean(_PK[(n, k)][np.asarray(c, int)]))


def one(pattern):
    f = sorted(glob.glob(pattern))
    assert len(f) == 1, (pattern, f)
    return counts(f[0])


def b7_files(seed, hard):
    """B7's evaluations for one seed: {(step, T): file} (Pre-registration 28's seeds_7 sources; hard cohort: B7s<seed>hard)."""
    out = {}
    pats = ([f"results/worth-v20v-B7-*/B7s{seed}hard/eval/step*_T*_eval.json"] if hard else
            [f"results/worth-v20v-testw6-*/B7s{seed}/eval/step*_T*_eval.json", f"results/worth-v20v-[A-Z]*-s0-*/B7s{seed}/eval/step*_T*_eval.json"])
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            name = f.split("/")[-1]
            st, t = name.split("_")[0][4:], name.split("_")[1][1:]
            out.setdefault((int(st), round(float(t), 2)), f)
    return out


def load(hard):
    tag = "hard" if hard else ""
    base = one("results/worth-v20v-B7-*/BASE7hard/eval/step0_T1.00_eval.json" if hard else
               "results/worth-v20v-testw6-*/BASE7v/eval/step0_T1.00_eval.json")
    seeds = {}
    for s in SEEDS:
        b7 = {k: counts(f)[:2] for k, f in b7_files(s, hard).items()}
        e = {}
        for st in (3, 6, 12, 25):
            f = glob.glob(f"results/worth-p33-sens7-*/B7e{s}{tag}/eval/step{st}_T1.00_eval.json")
            if f:
                e[st] = counts(f[0])[:2]
        seeds[s] = {"b7": b7, "b7e": e}
    return base[:2], seeds


def traj(base, sd, kind):
    b7, e = sd["b7"], sd["b7e"]
    pts = [(0, base)]
    if kind == "sparse":
        pts += [(st, b7[(st, 1.0)]) for st in (25, 50, 100, 150)]
    elif kind == "dense":
        pts += [(st, e[st]) for st in (3, 6, 12, 25)] + [(st, b7[(st, 1.0)]) for st in (50, 100, 150)]
    elif kind == "splice":
        pts += [(st, e[st]) for st in (3, 6, 12)] + [(st, b7[(st, 1.0)]) for st in (25, 50, 100, 150)]
    elif kind == "early":
        pts += [(st, e[st]) for st in (3, 6, 12, 25)]
    temps = [] if kind == "early" else [(t, b7[(150, t)]) for t in (1.0, 1.15, 1.3)]
    return pts, temps


def curve(points, idx, k):
    """points: list of (tiebreak, (counts, n)); returns sorted (pass@1, pass@k) arrays on the resampled problems."""
    m = sorted(((c[idx].mean() / n, tb, pk(c[idx], n, k)) for tb, (c, n) in points), key=lambda x: (x[0], x[1]))
    return np.array([x[0] for x in m]), np.array([x[2] for x in m])


def ref(base, seeds, kind, u, idx, k=16):
    vals = []
    for s in SEEDS:
        pts, temps = traj(base, seeds[s], kind)
        cand = []
        for P in ([pts] + ([temps] if temps else [])):
            xs, ys = curve(P, idx, k)
            if xs[0] <= u <= xs[-1]:
                cand.append(float(np.interp(u, xs, ys)))
        if cand:
            vals.append(max(cand))
    return float(np.mean(vals)) if vals else None


_LAW = {}


def law_ref(base, u, idx, k=16):
    c, n = base[0][idx], base[1]
    key = (id(base[0]), idx.tobytes())  # the law's trajectory depends only on the resampled base counts: cache it
    if key not in _LAW:
        _LAW[key] = law.trajectory(law.initial_theta(c, n, law.fit_prior(c, n)), -0.20)
    tr = _LAW[key]
    try:
        return law.pass_k_true(law.at_pass1(tr, u, strict=True), k)
    except ValueError:
        return None


def compare(c, n, R, draws_idx, k=16):
    """Gain (points) of one checkpoint against reference function R(u, idx, k), with bootstrap draws."""
    full = np.arange(len(c))
    r0 = R(c.mean() / n, full, k)
    if r0 is None:
        return None
    boots = []
    for i in draws_idx:
        r = R(c[i].mean() / n, i, k)
        boots.append(np.nan if r is None else 100 * (pk(c[i], n, k) - r))
    b = np.array(boots)
    ok = b[~np.isnan(b)]
    return {"est": 100 * (pk(c, n, k) - r0), "boots": b, "drop": float(np.isnan(b).mean()),
            "ci90": np.percentile(ok, [5, 95]) if len(ok) else (np.nan, np.nan),
            "p": float(min(1.0, 2 * min((ok <= 0).mean(), (ok >= 0).mean()))) if len(ok) else 1.0}


def verdicts(rows):
    scored = [r for r in rows if r[1] is not None and r[1]["drop"] <= 0.10]
    adj = dict(zip([r[0] for r in scored], holm(np.array([r[1]["p"] for r in scored])))) if scored else {}
    out = {}
    for name, g, why in rows:
        if g is None:
            out[name] = (why, None)
        elif g["drop"] > 0.10:
            out[name] = ("not interpreted (support)", None)
        else:
            q = adj[name]
            v = ("departure up" if g["est"] > 0 else "departure down") if abs(g["est"]) >= 2 and q <= 0.05 else \
                ("equivalent" if g["ci90"][0] >= -2 and g["ci90"][1] <= 2 else "inconclusive")
            out[name] = (v, q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    for hard in (False, True):
        cohort = "hard cohort (253)" if hard else "511 test problems"
        base, seeds = load(hard)
        miss = [(s, st) for s in SEEDS for st in (3, 6, 12, 25) if st not in seeds[s]["b7e"]]
        if miss:
            print(f"== {cohort}: B7e evaluations missing {miss}")
            continue
        P = len(base[0])
        rng = np.random.default_rng(33 if hard else 34)
        draws = [rng.integers(0, P, P) for _ in range(a.draws)]
        print(f"\n== {cohort}: base pass@1 {100 * base[0].mean() / base[1]:.1f}")
        for s in SEEDS:  # reproducibility: B7e step 25 minus B7 step 25
            (ce, n), (cb, _) = seeds[s]["b7e"][25], seeds[s]["b7"][(25, 1.0)]
            d1 = [100 * (ce[i].mean() - cb[i].mean()) / n for i in draws[:1000]]
            d16 = [100 * (pk(ce[i], n, 16) - pk(cb[i], n, 16)) for i in draws[:1000]]
            f1, f16 = 100 * (ce.mean() - cb.mean()) / n, 100 * (pk(ce, n, 16) - pk(cb, n, 16))
            flag = " FLAG" if abs(f1) > 3 or abs(f16) > 1 else ""
            print(f"   reproducibility seed {s}: B7e25 - B7 25 pass@1 {f1:+.2f} [{np.percentile(d1, 5):+.2f}, {np.percentile(d1, 95):+.2f}], "
                  f"pass@16 {f16:+.2f} [{np.percentile(d16, 5):+.2f}, {np.percentile(d16, 95):+.2f}]{flag}")
        pat = "results/worth-p33-sens7-*/PUB_*_hard/eval/step0_T1.00_eval.json" if hard else \
              "results/worth-p28*-public-*/PUB_*_2k/eval/step0_T1.00_eval.json"
        files = sorted(glob.glob(pat))
        print(f"   {len(files)} public checkpoints")
        fams = {"dense": lambda u, i, k: ref(base, seeds, "dense", u, i, k)}
        extra = {"sparse": lambda u, i, k: ref(base, seeds, "sparse", u, i, k),
                 "splice": lambda u, i, k: ref(base, seeds, "splice", u, i, k),
                 "early": lambda u, i, k: ref(base, seeds, "early", u, i, k)}
        if hard:
            fams["law"] = lambda u, i, k: law_ref(base, u, i, k)
        res = {}
        for f in files:
            name = f.split("/")[-3][4:].rsplit("_", 1)[0]
            c, n, d = counts(f)
            if d.get("truncated_rate", 0) > 0.9:
                res[name] = {"_why": "protocol failure (truncation %.0f%%)" % (100 * d["truncated_rate"])}
                continue
            res[name] = {kind: compare(c, n, R, draws) for kind, R in {**fams, **extra}.items()}
            res[name]["_meta"] = (100 * c.mean() / n, d.get("truncated_rate"), d.get("boxed_rate"))
            res[name]["_pass2"] = {kind: compare(c, n, R, draws[:500], 2) for kind, R in (("sparse", extra["sparse"]), ("dense", fams["dense"]))}
        for fam in fams:
            fam_id = {("dense", False): "A1", ("dense", True): "A2", ("law", True): "B2"}[(fam, hard)]
            rows = []
            for name, r in res.items():
                if "_why" in r:
                    rows.append((name, None, r["_why"]))
                else:
                    rows.append((name, r[fam], "insufficient overlap" if fam != "law" else "outside the law's range"))
            vs = verdicts(rows)
            print(f"   family {fam_id} ({cohort} vs {fam} reference):")
            tally = {}
            for name, r in res.items():
                v, q = vs[name]
                tally.setdefault(label(name), {}).setdefault(v, 0)
                tally[label(name)][v] += 1
                if "_why" in r or r[fam] is None:
                    print(f"      {name:52s} {v}")
                    continue
                g = r[fam]
                line = (f"      {name:52s} pass@1 {r['_meta'][0]:5.1f} gain {g['est']:+5.2f} 90% [{g['ci90'][0]:+.2f}, {g['ci90'][1]:+.2f}] "
                        f"drop {100 * g['drop']:.0f}% Holm p {q if q is not None else float('nan'):.3f} -> {v}")
                sp = r["sparse"]
                if fam == "dense" and sp is not None:
                    both = ~np.isnan(g["boots"]) & ~np.isnan(sp["boots"])
                    dd = g["boots"][both] - sp["boots"][both]
                    line += f" | sparse {sp['est']:+5.2f} (change {g['est'] - sp['est']:+.2f} [{np.percentile(dd, 5):+.2f}, {np.percentile(dd, 95):+.2f}])"
                    for kind in ("splice", "early"):
                        line += f" | {kind} {r[kind]['est']:+.2f}" if r[kind] is not None else f" | {kind} n/a"
                    p2 = r["_pass2"]
                    line += "".join(f" | pass@2 {kk} {p2[kk]['est']:+.2f}" if p2[kk] is not None else "" for kk in ("sparse", "dense"))
                print(line + f" [{label(name)}; trunc {r['_meta'][1]}]")
            print(f"      tally by label: {tally}")
        if hard and all(k in res and "_why" not in res[k] for k in ("Qwen25-Math-7B-DeepMath-1024samples-RePO", "Qwen25-Math-7B-DeepMath-1024samples-GRPO")):
            for fam in fams:
                a_, b_ = res["Qwen25-Math-7B-DeepMath-1024samples-RePO"][fam], res["Qwen25-Math-7B-DeepMath-1024samples-GRPO"][fam]
                if a_ is not None and b_ is not None:
                    both = ~np.isnan(a_["boots"]) & ~np.isnan(b_["boots"])
                    dd = a_["boots"][both] - b_["boots"][both]
                    print(f"   RePO minus GRPO (DeepMath), hard cohort, {fam}: {a_['est'] - b_['est']:+.2f} 90% [{np.percentile(dd, 5):+.2f}, {np.percentile(dd, 95):+.2f}]")


if __name__ == "__main__":
    main()
