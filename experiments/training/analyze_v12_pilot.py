"""Pre-registration 12 pilot: choose each arm's strength (seed 11, 100 steps, validation set).

Rule (PREREGISTRATION_12.md, amendment 1): among strengths whose step-100 validation pass@1 is
within 3 points of B's, the higher matched-accuracy gain (pass@16 minus B's trajectory, base and
steps 50, 100, at the same pass@1); ties to the weaker strength; if none is within 3 points, the
one closer to B's pass@1; an inactive intervention gets the stronger setting.
"""

import json
from math import comb
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"
N, K = 32, 16
# Each tuple lists the weaker strength first.
GROUPS = {"EF": ("EF6", "EF8"), "A": ("A1", "A3"), "E": ("E1", "E3"), "CM": ("CM1", "CM4"), "M": ("M1", "M4")}


def lane(name):
    for job in sorted(RES.glob("worth-v12p-J*-s11-*"), key=lambda j: j.name[-15:]):
        f = job / name / "baseline_passk.json"
        if f.exists():
            return {p["step"]: np.array(p["counts"]) for p in json.loads(f.read_text())}, json.loads((job / name / "baseline_train.json").read_text())
    return None, None


def metrics(c):
    return 100 * float(np.mean(c / N)), 100 * float(np.mean([1 - comb(N - int(x), K) / comb(N, K) for x in c]))


def active(name, train):
    if name.startswith("EF"):
        return any(r.get("floor", {}).get("tries", 0) > 0 for r in train)
    if name.startswith(("CM", "M")):
        return any(r.get("memory", {}).get("questions", 0) > 0 for r in train)
    if name.startswith(("A", "E")):
        return any((r.get("entropy_coef") or 0) > 0 for r in train)
    return True


def main():
    b, _ = lane("B")
    if not b or not {0, 50, 100} <= set(b):
        print("B incomplete:", sorted(b) if b else None)
        return
    traj = sorted(metrics(b[s]) + (s,) for s in (0, 50, 100))
    px, py = np.array([t[0] for t in traj]), np.array([t[1] for t in traj])
    b1 = metrics(b[100])[0]
    print(f"B: " + ", ".join(f"step {s}: pass@1 {x:.1f} pass@16 {y:.1f}" for x, y, s in traj))
    choice = {}
    for g, names in GROUPS.items():
        rows = []
        for name in names:
            pts, train = lane(name)
            if not pts or 100 not in pts:
                print(f"{g}: {name} incomplete")
                break
            p1, p16 = metrics(pts[100])
            gain = p16 - float(np.interp(p1, px, py)) if px.min() <= p1 <= px.max() else float("nan")
            rows.append((name, p1, p16, gain, active(name, train), np.mean([r["first_block_reward"] for r in train[-20:]])))
        else:
            for r in rows:
                print(f"   {g} {r[0]}: pass@1 {r[1]:.1f} (B {b1:.1f}), pass@16 {r[2]:.1f}, matched gain {r[3]:+.1f}, active {r[4]}, "
                      f"train acc last 20 {100 * r[5]:.1f}")
            act = [r for r in rows if r[4]]
            if not act:
                pick = rows[-1][0]  # inactive -> stronger
            else:
                near = [r for r in act if abs(r[1] - b1) <= 3 and not np.isnan(r[3])]
                if near:
                    best = max(r[3] for r in near)
                    pick = next(r[0] for r in near if r[3] == best)  # first listed = weaker on ties
                else:
                    pick = min(act, key=lambda r: abs(r[1] - b1))[0]
            choice[g] = pick
            print(f"{g}: choose {pick}")
    print("choices:", choice, "(M uses CM's lambda in the primary runs)")


if __name__ == "__main__":
    main()
