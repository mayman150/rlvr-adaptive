"""Descriptive diagnostic: do the SFT/OPD starts move the curve through more diverse answers or through a more even
spread of success across problems? Against B1k8's T = 1 trajectory (base and steps 25-150; first 32 samples per
problem for every point), at the checkpoint's pass@1: pass@16 gap (coverage), pass@2 gap (at matched pass@1 equal to
minus the change in the across-problem variance of success), and the change in distinct valid incorrect answer
classes per problem (P32's outcome classes). No intervals; one reference seed. Usage: python diag_start_spread.py
"""
import gzip
import json

import numpy as np

import analyze_p30 as A
from analyze_p32 import outcome

full = np.arange(511)


def metrics(path, k=32):
    d = json.load(gzip.open(path, "rt"))
    per = {}
    for s in d["samples"]:
        per.setdefault(s["problem"], []).append(s)
    corr, dist = np.zeros(511, int), np.zeros(511)
    for i in range(511):
        S = per[i][:k]
        corr[i] = sum(bool(s["reward"]) for s in S)
        dist[i] = len({outcome(s["text"], s["reward"]) for s in S if not s["reward"]} - {("none",)})
    return corr, dist


ref = [f"{A.S1}/BASE_test/eval/step0_T1.00_texts.json.gz"] + [f"/tmp/p32/texts/B1k8/step{st}_T1.00_texts.json.gz" for st in (25, 50, 100, 150)]
R = [metrics(p) for p in ref]


def interp(u, f):
    xs = np.array([A.pk(c, 32, 1, full) for c, _ in R])
    ys = np.array([f(c, d) for c, d in R])
    o = np.argsort(xs)
    return float(np.interp(u, xs[o], ys[o])) if xs.min() <= u <= xs.max() else None


for name, path in (("H25 start", f"{A.S1B}/H025_test/eval/step0_T1.00_texts.json.gz"), ("R25 start", f"{A.S1B}/R025_test/eval/step0_T1.00_texts.json.gz"),
                   ("OPD25 start", f"{A.S2}/OPD25_test/eval/step0_T1.00_texts.json.gz"),
                   ("H25 seed 8 step 25", f"{A.S2}/S2H25s8/eval/step25_T1.00_texts.json.gz"), ("H25 seed 8 step 50", f"{A.S2}/S2H25s8/eval/step50_T1.00_texts.json.gz")):
    c, d = metrics(path)
    u = A.pk(c, 32, 1, full)
    parts = []
    for key, f, scale in (("pass@16", lambda c_, d_: A.pk(c_, 32, 16, full), 100), ("pass@2", lambda c_, d_: A.pk(c_, 32, 2, full), 100),
                          ("distinct wrong classes", lambda c_, d_: d_.mean(), 1)):
        r = interp(u, f)
        parts.append(f"{key} {'n/a' if r is None else f'{scale * (f(c, d) - r):+.2f}'}")
    print(f"{name} (pass@1 {100 * u:.1f}): " + ", ".join(parts))
