"""Pre-registered matched-pass@1 comparison (Pre-registrations 5, 6, 7).

For each run R and reference run P (same seed), interpolate P's pass@32 at R's pass@1 along
P's trajectory (sorted by pass@1); average R - P over R's points inside P's pass@1 range.
"""
import json
import re
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
ARM = {"PSR": "psr", "NSR": "nsr", "PSRv": "psr", "NSRv": "nsr", "B25": "shift", "B50": "shift", "B75": "shift", "PSRlow": "psr"}


def points(version, name, seed):
    best = None
    for job in RES.glob(f"worth-{version}-{name}-s{seed}-*"):
        f = job / name / f"{ARM[name]}_passk.json"
        if f.exists():
            pts = sorted((100 * p["pass_at_k"]["1"], 100 * p["pass_at_k"]["32"], p["step"]) for p in json.loads(f.read_text()))
            if best is None or len(pts) > len(best):
                best = pts
    return best or []


def interp(ref, x):
    ref = sorted(ref)
    for (x0, y0, _), (x1, y1, _) in zip(ref, ref[1:]):
        if x0 <= x <= x1 and x1 > x0:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def compare(rp, pp):
    diffs = []
    for x, y, step in rp:
        yi = interp(pp, x)
        if yi is not None:
            diffs.append((step, round(x, 1), round(y - yi, 2)))
    mean = sum(d for *_, d in diffs) / len(diffs) if diffs else None
    return mean, diffs


def main():
    pairs = [("v5", "NSR", "v5", "PSR", (3, 4)), ("v6", "NSRv", "v6", "PSRv", (1, 2)),
             ("v7", "B25", "v5", "PSR", (3, 4)), ("v7", "B50", "v5", "PSR", (3, 4)), ("v7", "B75", "v5", "PSR", (3, 4)),
             ("v5", "NSR", "v7", "PSRlow", (3, 4))]
    for rv, rn, pv, pn, seeds in pairs:
        for seed in seeds:
            rp, pp = points(rv, rn, seed), points(pv, pn, seed)
            if not rp or not pp:
                continue
            mean, diffs = compare(rp, pp)
            m = f"{mean:+.2f}" if mean is not None else "n/a"
            print(f"{rn:6s} vs {pn:6s} seed {seed}: mean {m}  points {diffs}  ({len(rp)} vs {len(pp)} evals)")


if __name__ == "__main__":
    main()
