"""Pre-registration 36: the coverage-targeted SFT stream's problem weights.

For a problem with estimated base success p, one more unit of success raises pass@k by k (1 - p)^(k - 1) (the
derivative of 1 - (1 - p)^k): the marginal pass@k value. p is estimated from the base's 64 samples per training problem
(Pre-registration 30, Stage 0) by the posterior mean under a Beta prior fitted by moments to all counts (law.fit_prior),
so problems with zero observed successes get a finite estimate. Weights w = 16 (1 - p)^15 are written for every
problem with at least one archived teacher solution (Amendment 3's pool); sft_start.py --weights draws problems with
replacement in proportion to w.
Outputs: data/p36_sel_all.json (questions), data/p36_weights_pmv16.json ({question: weight}), a summary on stdout.
"""
import json
from pathlib import Path

import numpy as np

import law

HERE = Path(__file__).resolve().parent
st = json.loads((HERE / "data" / "p30_stage0.json").read_text())
arch = {d["question"]: d for d in json.loads((HERE / "data" / "p30_archive.json").read_text())}
c = np.array([r["c"] for r in st])
a, b = law.fit_prior(c, 64)
p = (c + a) / (64 + a + b)
w = 16 * (1 - p) ** 15
sel = [r["question"] for r in st if arch[r["question"]]["responses"]]
wmap = {r["question"]: float(w[i]) for i, r in enumerate(st) if arch[r["question"]]["responses"]}
(HERE / "data" / "p36_sel_all.json").write_text(json.dumps(sel))
(HERE / "data" / "p36_weights_pmv16.json").write_text(json.dumps(wmap))
ws = np.array([wmap[q] for q in sel])
cs = np.array([st[i]["c"] for i, r in enumerate(st) if arch[r["question"]]["responses"]])
print(f"prior Beta({a:.3f}, {b:.3f}); {len(sel)} problems with a teacher solution")
for lab, m in (("c=0", cs == 0), ("c=1-4", (cs >= 1) & (cs <= 4)), ("c=5-16", (cs >= 5) & (cs <= 16)), ("c=17-32", (cs >= 17) & (cs <= 32)), ("c>32", cs > 32)):
    print(f"   {lab}: {int(m.sum())} problems, {100 * ws[m].sum() / ws.sum():.1f}% of draws (uniform: {100 * m.mean():.1f}%)")
