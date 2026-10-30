"""Registered analysis of Pre-registration 35, phase 1 (controls of the H25 start; seed 8; written before any run).

For each start X in {SELF, TEACHF, FMT, CORR, H25b} and, for comparison, H25 (Stage 2):
- start offset at 1,024 tokens: pass@16 minus B1k_s's trajectory at the start's pass@1, mean over seeds 8-10
  (64-sample start counts; Stage 2's step-0 definition);
- start offset at 2,048 tokens: against B1k8's 2,048-token trajectory (base and steps 25-150 at 2,048 tokens, from the
  registered 2,048-token check);
- early-RL gap (seed 8): mean over eligible checkpoints 1-50 of pass@16 minus B1k8 at matched pass@1, with 90% and 95%
  intervals from 2,000 shared problem-bootstrap draws (seed 35), the 10% support rule and at least two eligible
  checkpoints; for H25 the Stage 2 seed-8 run is used;
- the phase-1 readings of the registration (FMT >= +0.96 with its manipulation check; CORR and H25b within 0.5 of
  H25 seed 8); diagnostics: boxed, truncated, mean length; SFT metadata.
Usage: python analyze_p35.py   (2,000 draws; --draws N only for exploratory runs)
"""

import argparse
import glob
import json

import numpy as np

import analyze_p30 as A

P35 = "results/worth-p35-controls-s0-*"
CHECK = "results/worth-p30-check2k-s0-*"
CTRL = ("SELF", "TEACHF", "FMT", "CORR", "H25b")
STEPS = (1, 2, 5, 10, 25, 50)


def one(pat, need=True):
    return A.one(pat, need)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    full = np.arange(A.P)
    rng = np.random.default_rng(35)
    draws = [rng.integers(0, A.P, A.P) for _ in range(a.draws)]
    refs = {s: A.b1k(s) for s in A.SEEDS}
    base2k = one(f"{CHECK}/BASE15_2k/eval/step0_T1.00_eval.json", need=False)
    b2k = [(0, base2k)] + [(st, one(f"{CHECK}/B1k8_2k/eval/step{st}_T1.00_eval.json", need=False)) for st in (25, 50, 100, 150)]
    have2k = base2k is not None and all(p is not None for _, p in b2k)
    starts = {c: (f"{P35}/P35{c}_start_1k", f"{P35}/P35{c}_start_2k", f"P35{c}s8", P35) for c in CTRL}
    starts["H25"] = (f"{A.S1B}/H025_test", f"{CHECK}/H25_2k", "S2H25s8", A.S2)
    print(f"Pre-registration 35, phase 1{'' if a.draws == 2000 else f' (EXPLORATORY: {a.draws} draws)'}")
    gaps = {}
    for X, (d1k, d2k, run, root) in starts.items():
        s1 = one(f"{d1k}/eval/step0_T1.00_eval.json", need=False)
        if s1 is None:
            print(f"== {X}: start evaluation missing")
            continue
        c, n, d = s1
        offs = []
        for s in A.SEEDS:
            r = A.interp_ref(refs[s][0])(A.pk(c, n, 1, full), full)
            offs.append(None if r is None else 100 * (A.pk(c, n, 16, full) - r))
        ok = [o for o in offs if o is not None]
        line = (f"== {X} start, 1,024 tokens: pass@1 {100 * A.pk(c, n, 1, full):.1f}, pass@16 {100 * A.pk(c, n, 16, full):.1f}, "
                f"pass@64 {100 * A.pk(c, n, 64, full):.1f}, truncated {100 * d.get('truncated_rate', np.nan):.1f}%, "
                f"boxed {100 * d['boxed_rate'] if d.get('boxed_rate') is not None else float('nan'):.1f}%; "
                f"offset vs B1k (mean of seeds) {np.mean(ok):+.2f}" if ok else " offset n/a")
        print(line)
        s2 = one(f"{d2k}/eval/step0_T1.00_eval.json", need=False)
        if s2 is not None and have2k:
            c2, n2, d2 = s2
            r2 = A.interp_ref(b2k)(A.pk(c2, n2, 1, full), full)
            print(f"   2,048 tokens: pass@1 {100 * A.pk(c2, n2, 1, full):.1f}, pass@16 {100 * A.pk(c2, n2, 16, full):.1f}, "
                  f"truncated {100 * d2.get('truncated_rate', np.nan):.1f}%; offset vs B1k8 at 2,048 "
                  f"{'n/a (outside support)' if r2 is None else f'{100 * (A.pk(c2, n2, 16, full) - r2):+.2f}'}")
        pts = {}
        for st in STEPS:
            r = one(f"{root}/{run}/eval/step{st}_T1.00_eval.json", need=False)
            if r is not None:
                pts[st] = r
        if len(pts) < len(STEPS):
            print(f"   early RL: missing checkpoints {[st for st in STEPS if st not in pts]}")
            continue
        R = A.interp_ref(refs[8][0])
        lo, hi = A.support(refs[8][0], None, full)
        el = [st for st in STEPS if lo <= A.pk(pts[st][0], pts[st][1], 1, full) <= hi]
        if len(el) < 2:
            print(f"   early RL: insufficient overlap (eligible {el})")
            continue
        fullg = A.gaps_for(pts, R, full, el, None, "full")
        per = [A.gaps_for(pts, R, i, el, None, k) for k, i in enumerate(draws)]
        t = A.summarize(f"{X} early-RL gap (seed 8)", {8: per, 9: per, 10: per}, {8: fullg, 9: fullg, 10: fullg}, "obs")
        if "est" not in t:
            print(f"   early RL: {t['status']}")
            continue
        gaps[X] = t["est"]
        print(f"   early-RL gap vs B1k8 (seed 8, checkpoints {el}): {t['est']:+.2f}, 90% [{t['ci90'][0]:+.2f}, {t['ci90'][1]:+.2f}], "
              f"95% [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}], draws affected {100 * t['drop'][8]:.1f}% "
              f"({'not interpreted (support)' if t['drop'][8] > 0.10 else 'ok'})")
        print("   per checkpoint: " + ", ".join(f"{st}: {g['obs']:+.1f} ({g['u']:.1f}/{g['v']:.1f})" for st, g in fullg.items() if g))
        meta = sorted(glob.glob(f"{root}/SFT35_{X}/start_25000/start_meta.json")) if X != "H25" else []
        if meta:
            m = json.load(open(meta[0]))
            print(f"   SFT: presented {m['presented_tokens']}, supervised {m['retained_tokens']}, steps {m['steps']}, problems "
                  f"{m['problems']}, solutions {m['solutions']}, dropped {m['dropped_at_load']}")
    print("\nPhase-1 readings (registered; descriptive)")
    h = gaps.get("H25")
    for X in CTRL:
        if X not in gaps:
            print(f"   {X}: no early-RL gap")
            continue
        g = gaps[X]
        trig = g >= 0.96
        print(f"   {X}: early-RL gap {g:+.2f}; phase 2 {'triggered' if trig or X in ('SELF', 'TEACHF') else 'not triggered'}"
              + (f"; within 0.5 of H25 seed 8 ({h:+.2f}): {abs(g - h) <= 0.5}" if h is not None and X in ("CORR", "H25b") else ""))


if __name__ == "__main__":
    main()
