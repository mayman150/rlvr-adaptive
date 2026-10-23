"""Pre-registration 16 pilot analysis (seed 12, 255 validation problems). Usage: python analyze_v16.py
Results: results/worth-v16pe-*/<ARM>/eval/step<N>_T<t>_eval.json (and BASE in B's lane).
"""

import json
import re

import numpy as np

import analyze_v12 as A
import analyze_v14 as V

CANDIDATES = ["GC", "PG", "AR", "MXGC"]  # selection order for ties: simpler first
REFS = ["B", "S32", "SD", "MXR"]


def load():
    out = {}
    for job in sorted(A.RES.glob("worth-v16pe-*"), key=lambda j: j.name[-15:]):
        for arm_dir in job.iterdir():
            for f in (arm_dir / "eval").glob("step*_T*_eval.json") if (arm_dir / "eval").exists() else []:
                step, t = re.match(r"step(\d+)_T([\d.]+)_eval", f.name).groups()
                d = json.loads(f.read_text())
                if "counts" in d and d.get("n") == A.N:
                    key = ("BASE", 0, 0, 1.0) if arm_dir.name == "BASE" else (arm_dir.name, 12, int(step), round(float(t), 2))
                    out[key] = np.array(d["counts"])
    return out


def main():
    res = load()
    if ("BASE", 0, 0, 1.0) not in res:
        print("base evaluation missing"); return
    base = res[("BASE", 0, 0, 1.0)]
    env = lambda r, s, w: A.reference(r, s, w, True)
    fresh = lambda: np.random.default_rng(0)
    print(f"{len(base)} validation problems; arms evaluated: {sorted({k[0] for k in res if k[0] != 'BASE'})}")
    rows = {}
    for arm in CANDIDATES + REFS[1:]:
        if not all((arm, 12, st, 1.0) in res for st in A.STEPS):
            print(f"   {arm}: incomplete"); continue
        t = A.compare(res, arm, env, (12,), 2000, fresh())
        d1 = (res[(arm, 12, 150, 1.0)].mean() - res[("B", 12, 150, 1.0)].mean()) / 32 * 100
        A.show(arm, t, f"step-150 pass@1 minus B {d1:+.1f}")
        rows[arm] = (t.get("estimate"), d1, t)
    print("Selection (Pre-registration 16): eligible = gain >= +1.5, step-150 pass@1 >= B - 2, >= 2 points in range")
    elig = {a: g for a, (g, d1, t) in rows.items() if a in CANDIDATES and g is not None and g >= 1.5 and d1 >= -2}
    if all(a in rows for a in CANDIDATES):
        if elig:
            top = max(elig.values())
            tied = [a for a in CANDIDATES if a in elig and elig[a] >= top - 0.5]
            print(f"   SELECTED: {tied[0]} (gain {elig[tied[0]]:+.2f}; tied within 0.5: {tied})")
        else:
            print("   no candidate qualifies")
    else:
        print(f"   pending: {[a for a in CANDIDATES if a not in rows]}; qualifying so far: {elig}")
    if "MXGC" in rows and "MXR" in rows:
        r = A.compare(res, "MXGC", V.traj_ref("MXR"), (12,), 2000, fresh())
        A.show("MXGC vs MXR trajectory", r, "reported")
    strata = {"never": base == 0, "rare": (base >= 1) & (base <= 4)}
    for arm in ["B"] + [a for a in rows]:
        k = (arm, 12, 150, 1.0)
        if k in res:
            c = res[k]
            print(f"   {arm} step 150 pass@16 by base stratum: " + ", ".join(
                f"{s} {100 * np.mean([1 - A.comb(32 - int(x), 16) / A.comb(32, 16) for x in c[m]]):.1f}" for s, m in strata.items()))


if __name__ == "__main__":
    main()
