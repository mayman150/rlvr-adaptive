"""Matched-pass@1 comparisons shared by the analyses of Pre-registrations 20-23 (Pre-registration 12's interpolation with
the guards added in Pre-registration 21).

An arm's checkpoint at pass@1 x is compared with the reference's pass@16 interpolated at x along the reference's points
(the base model and its checkpoints, sorted by pass@1, ties by step; the envelope also takes the maximum with the curve
through the reference's step-150 temperatures). Eligibility (checkpoints inside the reference's pass@1 range) is fixed
on the full data; checkpoints are averaged within a seed and seeds weighted equally; every bootstrap draw (over problems,
shared across arms and seeds) recomputes the curves; a resampled point outside the range takes the nearest end value
and is counted as clamped.
"""

import numpy as np

import analyze_v12 as A


def curve_fn(points, w):
    """points: list of (counts, tiebreak). Returns (value(x) or None outside, lo, hi)."""
    px, py = A.curve(points, w)
    return (lambda x: float(np.interp(x, px, py)) if px.min() <= x <= px.max() else None), px.min(), px.max()


def reference(traj, temps=None):
    """traj(seed) -> list of (counts, tiebreak); temps(seed) -> list of (counts, temperature) or None (trajectory only)."""
    def ref(seed, w):
        parts = [curve_fn(traj(seed), w)] + ([curve_fn(temps(seed), w)] if temps else [])

        def value(x):
            vals = [v(x) for v, _, _ in parts if v(x) is not None]
            return max(vals) if vals else None
        return value, min(p[1] for p in parts), max(p[2] for p in parts)
    return ref


def compare(arm_points, ref, seeds, n, draws=2000, rng=None):
    """arm_points(seed) -> list of (step, counts). Returns estimate, per-seed estimates and 90% intervals, pooled 90/95%
    intervals, two-sided bootstrap p-value, eligible checkpoints and clamping per seed, or {"verdict": ...}."""
    rng = rng or np.random.default_rng(0)
    ones = np.ones(n)
    elig = {}
    for s in seeds:
        _, lo, hi = ref(s, ones)
        elig[s] = [(st, c) for st, c in arm_points(s) if lo <= A.met(c, ones)[0] <= hi]
    if any(len(v) < 2 for v in elig.values()):
        return {"verdict": "insufficient overlap", "eligible": {s: [st for st, _ in v] for s, v in elig.items()}}

    def gaps(w, clamp=None):
        out = {}
        for s in seeds:
            value, lo, hi = ref(s, w)
            g = []
            for st, c in elig[s]:
                p1, p16 = A.met(c, w)
                v = value(p1)
                if v is None:
                    v = value(min(max(p1, lo), hi))
                    if clamp is not None:
                        clamp[s] = clamp.get(s, 0) + 1
                g.append(p16 - v)
            out[s] = float(np.mean(g))
        return out
    full = gaps(ones)
    clamp, boot = {}, []
    for _ in range(draws):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        boot.append(gaps(w, clamp))
    pooled = np.array([np.mean(list(b.values())) for b in boot])
    per = {s: np.array([b[s] for b in boot]) for s in seeds}
    return {"estimate": float(np.mean(list(full.values()))), "per_seed": full,
            "ci90": np.percentile(pooled, [5, 95]), "ci95": np.percentile(pooled, [2.5, 97.5]),
            "per_seed_ci90": {s: np.percentile(per[s], [5, 95]) for s in seeds},
            "p_two_sided": float(min(1.0, 2 * min((pooled <= 0).mean(), (pooled >= 0).mean()))),
            "eligible": {s: [st for st, _ in v] for s, v in elig.items()},
            "clamped_frac": {s: clamp.get(s, 0) / (draws * len(elig[s])) for s in seeds}}


def holm(tests, alpha=0.05):
    ps = sorted((t.get("p_two_sided", 1.0), k) for k, t in tests.items())
    out, stop = {}, False
    for i, (p, k) in enumerate(ps):
        ok = not stop and p <= alpha / (len(ps) - i)
        stop = stop or not ok
        out[k] = (min(1.0, p * (len(ps) - i)), ok)
    return out


def equivalent(t, margin=2.0):
    """Pre-registration 21's average equivalence: pooled and each seed's 90% interval within [-margin, margin]."""
    return "estimate" in t and t["ci90"][0] >= -margin and t["ci90"][1] <= margin and \
        all(lo >= -margin and hi <= margin for lo, hi in t["per_seed_ci90"].values())


def departure(t, holm_ok, margin=2.0):
    signs = {np.sign(v) for v in t.get("per_seed", {}).values()}
    return "estimate" in t and abs(t["estimate"]) >= margin and holm_ok and len(signs) == 1


def moves(t, holm_ok, margin=2.0):
    """Superiority: at least +margin, 95% interval above 0, positive in every seed, Holm-significant."""
    return "estimate" in t and t["estimate"] >= margin and t["ci95"][0] > 0 and all(v > 0 for v in t["per_seed"].values()) and holm_ok


def clamped_ok(t, limit=0.10):
    return all(v <= limit for v in t.get("clamped_frac", {}).values())


def show(name, t, verdict=""):
    if "estimate" not in t:
        return f"   {name}: {t['verdict']} {t.get('eligible', '')}"
    ps = ", ".join(f"{s}: {v:+.2f} [{t['per_seed_ci90'][s][0]:+.2f}, {t['per_seed_ci90'][s][1]:+.2f}]" for s, v in t["per_seed"].items())
    cl = ", ".join(f"{s}: {100 * v:.1f}%" for s, v in t["clamped_frac"].items())
    return (f"   {name}: {t['estimate']:+.2f}, 90% [{t['ci90'][0]:+.2f}, {t['ci90'][1]:+.2f}], 95% [{t['ci95'][0]:+.2f}, "
            f"{t['ci95'][1]:+.2f}]; per seed {ps}; clamped {cl}" + (f" -> {verdict}" if verdict else ""))
