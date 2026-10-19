"""Pre-registration 11, enlarged: the same rule on 511 problems (the original 256 without the
audit-excluded item, plus 256 further problems). See PREREGISTRATION_11.md (Result and enlargement).

Usage: python analyze_v11e.py [--draws 2000]
"""

import argparse
import json
import re

import numpy as np

import analyze_v11 as A

KEYS = [(0, 0, 1.0)] + [(s, st, 1.0) for s in A.SEEDS for st in (50, 100, 150)] + \
       [(s, 150, t) for s in A.SEEDS for t in (1.07, 1.15, 1.3)]


def load_new():
    out = {}
    for job in sorted(A.RES.glob("worth-v11e-*"), key=lambda j: j.name[-15:]):
        m = re.match(r"worth-v11e-(BASE|PSRa|PSRb)-s(\d+)-", job.name)
        if not m:
            continue
        for f in (job / "eval").glob("step*_T*_eval.json"):
            step, t = re.match(r"step(\d+)_T([\d.]+)_eval", f.name).groups()
            d = json.loads(f.read_text())
            if "counts" in d and d.get("n") == A.N and len(d["counts"]) == 256 and all(isinstance(c, int) and 0 <= c <= A.N for c in d["counts"]):
                out[(int(m.group(2)), int(step), round(float(t), 2))] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    old, new = A.load(), load_new()
    missing = [k for k in KEYS if k not in old or k not in new]
    if missing:
        print(f"verdict: incomplete evaluation; missing {missing}")
        return
    probs = json.loads((A.RES.parent / "data" / "eval_frontier_256.json").read_text())
    corr = json.loads((A.RES.parent / "data" / "label_corrections.json").read_text())
    keep = np.array([not any(q["question"].startswith(p) for p in corr["exclude_prefixes"]) for q in probs])
    res = {k: {"counts": list(np.asarray(old[k]["counts"])[keep]) + list(new[k]["counts"])} for k in KEYS}
    A.N_PROBLEMS = len(res[KEYS[0]]["counts"])
    print(f"combined problems: {A.N_PROBLEMS}")
    rng = np.random.default_rng(0)
    ones = np.ones(A.N_PROBLEMS)
    A.HIGH_T = (1.07, 1.15, 1.3)
    qualifying = {}
    for seed in A.SEEDS:
        px, _ = A.trajectory(res, seed, ones)
        qualifying[seed] = [t for t in A.HIGH_T if px.min() <= A.metrics(res[(seed, 150, t)]["counts"], ones)[0] <= px.max()]
    print("qualifying temperatures:", qualifying)
    if any(not q for q in qualifying.values()):
        print("verdict: insufficient overlap")
        return
    full = A.diffs(res, ones, qualifying)
    per_seed = {s: float(np.mean(v)) for s, v in full.items()}
    est = float(np.mean(list(per_seed.values())))
    boot, boot_pts, outside = [], [], []
    for _ in range(args.draws):
        w = np.bincount(rng.integers(0, A.N_PROBLEMS, A.N_PROBLEMS), minlength=A.N_PROBLEMS).astype(float)
        d = A.diffs(res, w, qualifying, outside)
        boot.append(np.mean([np.mean(v) for v in d.values()]))
        boot_pts.append(np.concatenate([d[s] for s in A.SEEDS]))
    lo90, hi90 = np.percentile(boot, [5, 95])
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    flat, bp = np.concatenate([full[s] for s in A.SEEDS]), np.array(boot_pts)
    se = bp.std(axis=0, ddof=1)
    q = np.percentile(np.max(np.abs(bp - flat) / np.where(se > 0, se, np.inf), axis=1), 90)
    band = [(x - q * e, x + q * e) for x, e in zip(flat, se)]
    crossing = any(lo > 2 for lo, _ in band) and any(hi < -2 for _, hi in band)
    if est >= 2 and lo95 > 0:
        verdict = "separable"
    elif lo90 >= -2 and hi90 <= 2:
        verdict = "average agreement"
    elif est <= -2 and hi95 < 0:
        verdict = "below"
    else:
        verdict = "inconclusive"
    if crossing:
        verdict += "; CROSSING"
    print(f"pooled D {est:+.2f}, 90% [{lo90:+.2f}, {hi90:+.2f}], 95% [{lo95:+.2f}, {hi95:+.2f}]; per seed "
          + ", ".join(f"{s}: {v:+.2f}" for s, v in per_seed.items()))
    labels = [(s, t) for s in A.SEEDS for t in qualifying[s]]
    print("simultaneous 90% band: " + "; ".join(f"s{s} T{t}: {x:+.1f} [{lo:+.1f}, {hi:+.1f}]" for (s, t), x, (lo, hi) in zip(labels, flat, band)))
    print(f"bootstrap points outside the range: {100 * np.mean(outside):.1f}%")
    print(f"verdict: {verdict}")
    print("branch for Pre-registration 12: " + ("S" if verdict.startswith("separable") else "M"))


if __name__ == "__main__":
    main()
