"""Registered analysis of Pre-registration 35, phase 1 (controls of the H25 start; seed 8; written before any run,
revised after Codex's pre-launch review).

For each start X in {SELF, TEACHF, FMT, CORR, H25b} and, for comparison, H25 (Stage 2):
- start diagnostics at 1,024 and 2,048 tokens (pass@1/16/64, truncated, boxed, mean length), always reported;
- start offset: pass@16 minus B1k8's trajectory at the start's pass@1 at 1,024 tokens and at 2,048 tokens (B1k8's
  2,048-token trajectory from the registered check), so the budget comparison uses the same reference seed; the mean
  over seeds 8-10 at 1,024 tokens is also reported (Stage 2's step-0 definition);
- early-RL gap (seed 8): mean over eligible checkpoints 1-50 of pass@16 minus B1k8 at matched pass@1, 90%/95% intervals
  from 2,000 shared problem-bootstrap draws (seed 35), the 10% support rule, at least two eligible checkpoints;
  per-checkpoint diagnostics (truncated, boxed, mean length);
- registered readings, each used only when its estimate is supported (status ok): phase-2 trigger (gap >= +0.96, and
  SELF/TEACHF together); FMT's manipulation check (start boxed rate above and truncation below the base's at 1,024
  tokens) gating any reading of FMT; CORR and H25b within 0.5 points of H25 seed 8;
- writes 50 random FMT-start answers (seed 35) for the registered extraction audit.
Usage: python analyze_p35.py   (2,000 draws; --draws N only for exploratory runs)
"""

import argparse
import glob
import gzip
import json
import random

import numpy as np

import analyze_p30 as A

P35 = "results/worth-p35-controls-s0-*"
CHECK = "results/worth-p30-check2k-s0-*"
CTRL = ("SELF", "TEACHF", "FMT", "CORR", "H25b")
STEPS = (1, 2, 5, 10, 25, 50)


def diag(d):
    return (f"truncated {100 * d.get('truncated_rate', np.nan):.1f}%, boxed "
            f"{100 * d['boxed_rate'] if d.get('boxed_rate') is not None else float('nan'):.1f}%, "
            f"mean length {d.get('mean_length_tokens', d.get('mean_length', float('nan')))}")


def offset(start, ref_pts, full):
    c, n, _ = start
    r = A.interp_ref(ref_pts)(A.pk(c, n, 1, full), full)
    return None if r is None else 100 * (A.pk(c, n, 16, full) - r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    full = np.arange(A.P)
    rng = np.random.default_rng(35)
    draws = [rng.integers(0, A.P, A.P) for _ in range(a.draws)]
    refs = {s: A.b1k(s) for s in A.SEEDS}
    base1k = A.one(f"{A.S1}/BASE_test/eval/step0_T1.00_eval.json")
    base2k = A.one(f"{CHECK}/BASE15_2k/eval/step0_T1.00_eval.json", need=False)
    b2k = [(0, base2k)] + [(st, A.one(f"{CHECK}/B1k8_2k/eval/step{st}_T1.00_eval.json", need=False)) for st in (25, 50, 100, 150)]
    have2k = base2k is not None and all(p is not None for _, p in b2k)
    starts = {c: (f"{P35}/P35{c}_start_1k", f"{P35}/P35{c}_start_2k", f"P35{c}s8", P35, f"{P35}/SFT35_{c}") for c in CTRL}
    starts["H25"] = (f"{A.S1B}/H025_test", f"{CHECK}/H25_2k", "S2H25s8", A.S2, f"{A.S1B}/SFT1b_H")
    print(f"Pre-registration 35, phase 1{'' if a.draws == 2000 else f' (EXPLORATORY: {a.draws} draws)'}")
    print(f"Base at 1,024 tokens (Stage 1, 64 samples): pass@1 {100 * A.pk(base1k[0], base1k[1], 1, full):.1f}, {diag(base1k[2])}")
    est, status, manip = {}, {}, None
    for X, (d1k, d2k, run, root, sftdir) in starts.items():
        print(f"\n== {X}")
        meta = sorted(glob.glob(f"{sftdir}/start_25000/start_meta.json"))
        if meta:
            m = json.load(open(meta[0]))
            print(f"   SFT: presented {m['presented_tokens']}, supervised {m['retained_tokens']}, updates {m['steps']}, "
                  f"problems {m['problems']}, solutions {m['solutions']}, dropped at load {m['dropped_at_load']}")
        s1 = A.one(f"{d1k}/eval/step0_T1.00_eval.json", need=False)
        if s1 is None:
            print("   start evaluation at 1,024 tokens missing")
            continue
        c, n, d = s1
        o8 = offset(s1, refs[8][0], full)
        om = [offset(s1, refs[s][0], full) for s in A.SEEDS]
        om = [x for x in om if x is not None]
        print(f"   start, 1,024 tokens: pass@1 {100 * A.pk(c, n, 1, full):.1f}, pass@16 {100 * A.pk(c, n, 16, full):.1f}, "
              f"pass@64 {100 * A.pk(c, n, 64, full):.1f}, {diag(d)}; offset vs B1k8 "
              f"{'n/a' if o8 is None else f'{o8:+.2f}'}; mean over B1k seeds {np.mean(om):+.2f}" if om else "")
        if X == "FMT":
            bt, bb = base1k[2].get("truncated_rate"), base1k[2].get("boxed_rate")
            ok = (d.get("truncated_rate") is not None and bt is not None and d["truncated_rate"] < bt
                  and d.get("boxed_rate") is not None and bb is not None and d["boxed_rate"] > bb)
            manip = ok
            print(f"   FMT manipulation check (boxed up and truncation down vs the base): {'passed' if ok else 'FAILED'}")
            tx = sorted(glob.glob(f"{d1k}/eval/step0_T1.00_texts.json.gz"))
            if tx:
                samples = json.load(gzip.open(tx[0], "rt"))["samples"]
                pick = random.Random(35).sample(range(len(samples)), 50)
                json.dump([samples[k] for k in pick], open("results_p26/p35_fmt_extraction_audit_sample.json", "w"))
                print("   wrote results_p26/p35_fmt_extraction_audit_sample.json (50 random FMT-start answers for the audit)")
        s2 = A.one(f"{d2k}/eval/step0_T1.00_eval.json", need=False)
        if s2 is not None and have2k:
            o2 = offset(s2, b2k, full)
            c2, n2, d2 = s2
            print(f"   start, 2,048 tokens: pass@1 {100 * A.pk(c2, n2, 1, full):.1f}, pass@16 {100 * A.pk(c2, n2, 16, full):.1f}, "
                  f"{diag(d2)}; offset vs B1k8 at 2,048 {'n/a (outside support)' if o2 is None else f'{o2:+.2f}'}"
                  f" (at 1,024: {'n/a' if o8 is None else f'{o8:+.2f}'})")
        pts = {}
        for st in STEPS:
            r = A.one(f"{root}/{run}/eval/step{st}_T1.00_eval.json", need=False)
            if r is not None:
                pts[st] = r
        if len(pts) < len(STEPS):
            print(f"   early RL: missing checkpoints {[st for st in STEPS if st not in pts]}")
            continue
        print("   checkpoints (pass@1/pass@16; diagnostics): " + "; ".join(
            f"{st}: {100 * A.pk(p[0], p[1], 1, full):.1f}/{100 * A.pk(p[0], p[1], 16, full):.1f} ({diag(p[2])})" for st, p in pts.items()))
        lo, hi = A.support(refs[8][0], None, full)
        el = [st for st in STEPS if lo <= A.pk(pts[st][0], pts[st][1], 1, full) <= hi]
        if len(el) < 2:
            status[X] = "insufficient overlap"
            print(f"   early RL: insufficient overlap (eligible {el})")
            continue
        R = A.interp_ref(refs[8][0])
        fullg = A.gaps_for(pts, R, full, el, None, "full")
        per = [A.gaps_for(pts, R, i, el, None, k) for k, i in enumerate(draws)]
        t = A.summarize(f"{X} early-RL gap (seed 8)", {s: per for s in A.SEEDS}, {s: fullg for s in A.SEEDS}, "obs")
        status[X] = t.get("status")
        if "est" not in t:
            print(f"   early RL: {t['status']}")
            continue
        est[X] = t["est"]
        print(f"   early-RL gap vs B1k8 (checkpoints {el}): {t['est']:+.2f}, 90% [{t['ci90'][0]:+.2f}, {t['ci90'][1]:+.2f}], "
              f"95% [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}], draws affected {100 * t['drop'][8]:.1f}% -> {t['status']}")
    print("\nPhase-1 readings (registered; descriptive; only supported estimates are used)")
    h = est.get("H25") if status.get("H25") == "ok" else None
    for X in CTRL:
        if status.get(X) != "ok":
            print(f"   {X}: no supported early-RL gap ({status.get(X, 'missing')}); no reading")
            continue
        g = est[X]
        trig = g >= 0.96 or X in ("SELF", "TEACHF")
        line = f"   {X}: early-RL gap {g:+.2f}; phase 2 {'triggered' if trig else 'not triggered'}"
        if X == "FMT":
            line += ("; finishing/format explains at least half of H25's effect" if g >= 0.96 and manip else
                     "; manipulation check failed: no reading" if not manip else "; below half of H25's effect")
        if X in ("CORR", "H25b") and h is not None:
            line += f"; within 0.5 of H25 seed 8 ({h:+.2f}): {abs(g - h) <= 0.5}"
        print(line)


if __name__ == "__main__":
    main()
