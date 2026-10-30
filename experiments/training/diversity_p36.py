"""Pre-registration 36's mechanism diagnostics (descriptive, no verdicts): coverage, across-problem spread and
within-problem answer variety at matched pass@1, for every P36 arm and P35 phase-2 control. Written before any P36
result was read.

Metrics as in diversity_decomposition.py (first 32 samples per problem): pass@16 gap, pass@2 gap (at matched pass@1,
minus the change in the across-problem second moment of success), distinct valid incorrect answer classes per
problem, and the collision probability among valid incorrect samples (problems with at least two).
Comparisons: each arm's checkpoints 1-150 against the same-seed B1k_s trajectory (base and steps 25-150), and CWS's
checkpoints against the UNI and HARD trajectories (start and checkpoints), as in analyze_p36.py: eligibility fixed on
the full data, at least two eligible checkpoints per seed, every registered seed pair required, seeds weighted
equally. Bootstrap: analyze_p36.py's schedule (2,000 index draws, seed 36); a seed's draw is unavailable if an
eligible checkpoint leaves the reference's range (fixed checkpoint set). Control: B1k_s against B1k_{next seed}.
Usage: python diversity_p36.py [--draws 2000]
"""

import argparse
import glob

import numpy as np

from diversity_decomposition import P, load, metrics

STEPS = (1, 2, 5, 10, 25, 50, 100, 150)
SEEDS = (8, 9, 10)
SFT_SEED = {8: 3636, 9: 3737, 10: 3838, 11: 3939}
TB = "results/texts_b1k"
P36 = "results/worth-p36-job*"
P35C = "results/worth-p35-controls-s0-20261030-172723"
KEYS = (("p16", "{:+.2f}"), ("p2", "{:+.2f}"), ("D", "{:+.2f}"), ("coll", "{:+.3f}"))


def one(pat):
    f = sorted(glob.glob(pat))
    if len(f) > 1:
        raise SystemExit(f"multiple matches for {pat}")
    return f[0] if f else None


def run(root, name, steps=STEPS):
    paths = [one(f"{root}/{name}/eval/step{st}_T1.00_texts.json.gz") for st in steps]
    return paths if all(paths) else None


def b1k(s):
    return [f"{TB}/BASE15/step0_T1.00_texts.json.gz"] + [f"{TB}/B1k{s}/step{st}_T1.00_texts.json.gz" for st in (25, 50, 100, 150)]


def contrast(arm, refs, seeds, draws):
    """arm, refs: {seed: [texts paths]}. Returns {metric: (estimate, 90% interval)} or a status string."""
    if any(arm.get(s) is None or refs.get(s) is None for s in seeds):
        return "unavailable (missing seeds)"
    full = np.arange(P)
    est, per = {k: [] for k, _ in KEYS}, {k: [] for k, _ in KEYS}
    for s in seeds:
        R = [load(p) for p in refs[s]]
        A = [load(p) for p in arm[s]]
        rx = np.array([metrics(r, full)["p1"] for r in R])
        el = [a for a in A if rx.min() <= metrics(a, full)["p1"] <= rx.max()]
        if len(el) < 2:
            return "insufficient overlap"
        out = {k: np.full(len(draws) + 1, np.nan) for k, _ in KEYS}
        for d, idx in enumerate([full] + draws):
            Rm = [metrics(r, idx) for r in R]
            xs = np.array([m["p1"] for m in Rm])
            o = np.lexsort((np.arange(len(xs)), xs))  # ties by trajectory order (step)
            Am = [metrics(a, idx) for a in el]
            if any(not xs.min() <= m["p1"] <= xs.max() for m in Am):
                continue
            for k, _ in KEYS:
                ys = np.array([m[k] for m in Rm])
                out[k][d] = np.mean([m[k] - np.interp(m["p1"], xs[o], ys[o]) for m in Am])
        for k, _ in KEYS:
            est[k].append(out[k][0])
            per[k].append(out[k][1:])
    res = {}
    for k, _ in KEYS:
        pooled = np.mean(np.vstack(per[k]), 0)
        ok = pooled[np.isfinite(pooled)]
        res[k] = (float(np.mean(est[k])), np.percentile(ok, [5, 95]) if ok.size else (np.nan, np.nan))
    res["unavailable"] = float(np.mean([np.isnan(per["p16"][j]).mean() for j in range(len(seeds))]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(36)
    draws = [rng.integers(0, P, P) for _ in range(a.draws)]  # the same index draws as analyze_p36.py
    arms = {}
    for X in ("CWS", "UNI", "HARD", "TAIL", "TCWS"):
        arms[X] = {s: run(P36, f"P36{X}{SFT_SEED[s]}s{s}") for s in SEEDS}
    arms["TIME"] = {s: run(P36, f"P36TIMEs{s}") for s in SEEDS}
    for X in ("SELF", "TEACHF", "CORR", "H25b"):
        arms[X] = {s: run(P36, f"P35b{X}s{s}") for s in SEEDS}
    arms["H25 (Stage 2)"] = {s: run("results/worth-p30-stage2-*", f"S2H25s{s}") for s in SEEDS}
    refs = {s: b1k(s) for s in SEEDS}

    def own(X, start):  # an arm's trajectory: its 64-sample start, then its checkpoints
        return {s: ([start(s)] + arms[X][s]) if arms[X][s] and start(s) else None for s in SEEDS}

    def p36start(X):
        return lambda s: one(f"{P36}/P36{X}{SFT_SEED[s]}_start_1k/eval/step0_T1.00_texts.json.gz")

    rows = [("control: B1k seed vs next seed", {s: b1k(s)[1:] for s in SEEDS}, {s: b1k({8: 9, 9: 10, 10: 8}[s]) for s in SEEDS})]
    rows += [(f"{X} vs B1k", arms[X], refs) for X in arms]
    rows += [(f"CWS vs {X} trajectory", arms["CWS"], own(X, p36start(X))) for X in ("UNI", "HARD")]
    rows += [("CWS vs HSELF trajectory", arms["CWS"], own("SELF", lambda s: one(f"{P35C}/P35SELF_start_1k/eval/step0_T1.00_texts.json.gz")))]
    print(f"{'comparison':32s} {'pass@16 gap':>22s} {'pass@2 gap (spread)':>22s} {'distinct wrong':>22s} {'wrong collision':>24s}  unavailable")
    for label, arm, rf in rows:
        r = contrast(arm, rf, SEEDS, draws)
        if isinstance(r, str):
            print(f"{label:32s} {r}")
            continue
        cells = [f"{fmt.format(r[k][0])} [{fmt.format(r[k][1][0])}, {fmt.format(r[k][1][1])}]" for k, fmt in KEYS]
        print(f"{label:32s} " + " ".join(f"{c:>22s}" for c in cells) + f"  {100 * r['unavailable']:.1f}%", flush=True)


if __name__ == "__main__":
    main()
