"""Stage B analysis: structure check, shape of |grad p|^2 against p, and F0, F1.

Bootstrap intervals resample prompts. Difficulty p is estimated from each prompt's
own answers, so bins are noisy; this pilot reports that dilution rather than
correcting it.
"""

import argparse
import json
from pathlib import Path

import numpy as np

SHAPES = {"p^2 (1-p)^2": lambda p: (p * (1 - p)) ** 2, "p (1-p)": lambda p: p * (1 - p),
          "constant": lambda p: np.ones_like(p)}
BINS = [(0.0, 0.0), (1e-9, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1 - 1e-9), (1.0, 1.0)]


def boot(values, rng, n=2000):
    values = np.asarray(values, float)
    if len(values) < 2:
        return float(values.mean()) if len(values) else None, None, None
    means = [rng.choice(values, len(values)).mean() for _ in range(n)]
    return float(values.mean()), float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("summary")
    parser.add_argument("--block", default="combined")
    args = parser.parse_args()
    data = json.loads(Path(args.summary).read_text())
    rng = np.random.default_rng(0)
    recs = data["records"]
    est = [r["estimates"][args.block] for r in recs]
    p = np.array([e["p"] for e in est])
    print(f"{len(recs)} prompts, {data['answers']} answers each, block {args.block}")
    print("difficulty distribution (share correct):", np.round(np.sort(p), 2).tolist())

    # 1. Structure: three pair-type estimates of G2 on mixed-outcome prompts.
    mixed = [e for e in est if 0 < e["p"] < 1 and e["pair11"] is not None and e["pair00"] is not None]
    g11 = [e["pair11"] * e["p"] ** 2 for e in mixed]
    g00 = [e["pair00"] * (1 - e["p"]) ** 2 for e in mixed]
    g10 = [-e["pair10"] * e["p"] * (1 - e["p"]) for e in mixed]
    glpo = [e["G2_lpo"] for e in mixed]
    print(f"\n1. Structure check on {len(mixed)} mixed-outcome prompts (theory: all four agree)")
    for name, vals in (("from correct pairs", g11), ("from wrong pairs", g00),
                       ("from mixed pairs", g10), ("pair statistic", glpo)):
        m, lo, hi = boot(vals, rng)
        if m is None or lo is None:
            print(f"   G2 {name:20s} not enough prompts")
            continue
        print(f"   G2 {name:20s} {m:10.4g}   90% interval [{lo:.4g}, {hi:.4g}]")

    # 2. Shape of G2 against p.
    print("\n2. |grad p|^2 against difficulty (c = 1/2 statistic, valid for all prompts)")
    centers, means = [], []
    for lo_b, hi_b in BINS:
        sel = [e for e in est if lo_b <= e["p"] <= hi_b]
        if not sel:
            continue
        m, lo, hi = boot([e["G2_half"] for e in sel], rng)
        pc = float(np.mean([e["p"] for e in sel]))
        centers.append(pc), means.append(m)
        label = f"p = {lo_b:g}" if lo_b == hi_b else f"{lo_b:.2f} < p < {hi_b:.2f}"
        print(f"   {label:20s} n={len(sel):2d}  G2 {m:10.4g}  [{lo}, {hi}]")
    centers, means = np.array(centers), np.array(means)
    print("   fit of the binned means to each shape (relative residual, lower is better):")
    for name, f in SHAPES.items():
        x = f(np.clip(centers, 0.02, 0.98))
        k = float(x @ means / (x @ x))
        resid = float(np.linalg.norm(means - k * x) / np.linalg.norm(means))
        print(f"     {name:12s} residual {resid:.3f}")

    # 3. F0, F1, lengths, and the noise ratio R = F / G2.
    print("\n3. Per-answer gradient size")
    F1 = [e["F1"] for e in est if e["F1"] is not None]
    F0 = [e["F0"] for e in est if e["F0"] is not None]
    for name, vals in (("F1 (correct)", F1), ("F0 (wrong)", F0)):
        m, lo, hi = boot(vals, rng)
        if m is None or lo is None:
            print(f"   {name:14s} not enough prompts")
            continue
        print(f"   {name:14s} mean {m:.4g}  [{lo:.4g}, {hi:.4g}]")
    lens = [(np.mean([L for L, r in zip(rec["lengths"], rec["rewards"]) if r == 1] or [np.nan]),
             np.mean([L for L, r in zip(rec["lengths"], rec["rewards"]) if r == 0] or [np.nan])) for rec in recs]
    print(f"   mean length: correct {np.nanmean([a for a, _ in lens]):.0f} tokens, wrong {np.nanmean([b for _, b in lens]):.0f} tokens")
    g2_mean = float(np.mean([e["G2_half"] for e in est]))
    if g2_mean > 0:
        print(f"   noise ratio R = mean F / mean G2 = {np.mean(F1 + F0) / g2_mean:.3g}")
    out = {"block": args.block, "p": p.tolist(), "bins": {"centers": centers.tolist(), "means": means.tolist()},
           "F1_mean": float(np.mean(F1)) if F1 else None, "F0_mean": float(np.mean(F0)) if F0 else None,
           "G2_mean": g2_mean}
    Path(args.summary).with_name(f"analysis_{args.block}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
