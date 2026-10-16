"""Compare the arms of finished training runs and apply the pre-registered rule.

Usage: python analyze_train.py results/<job-1>/training [results/<job-2>/training ...]
Each directory holds calibration.json, config.json, and per-arm *_train.json and *_eval.json.
"""

import json
import sys
from pathlib import Path


def load(run):
    run = Path(run)
    out = {"run": run.parent.name, "w": json.loads((run / "config.json").read_text())["w_used"]}
    for arm in ("baseline", "scaled"):
        train = json.loads((run / f"{arm}_train.json").read_text())
        evals = json.loads((run / f"{arm}_eval.json").read_text())
        last = [r["mean_reward"] for r in train[-50:]]
        out[arm] = {"final_accuracy": evals[-1]["accuracy"], "final_step": evals[-1]["step"],
                    "evals": {e["step"]: e["accuracy"] for e in evals},
                    "late_reward": sum(last) / len(last), "steps": len(train)}
    return out


def main():
    runs = [load(r) for r in sys.argv[1:]]
    print(f"{'run':40s} {'w':>6s} {'acc base':>9s} {'acc scaled':>11s} {'diff':>7s} {'late rew base':>14s} {'late rew scaled':>16s}")
    for r in runs:
        b, s = r["baseline"], r["scaled"]
        print(f"{r['run']:40s} {r['w']:6.3f} {b['final_accuracy']:9.4f} {s['final_accuracy']:11.4f} "
              f"{100 * (s['final_accuracy'] - b['final_accuracy']):+6.1f}pt {b['late_reward']:14.3f} {s['late_reward']:16.3f}")
        steps = sorted(set(b["evals"]) & set(s["evals"]))
        print("   accuracy by step:", ", ".join(f"{k}: {b['evals'][k]:.3f} vs {s['evals'][k]:.3f}" for k in steps))
    if len(runs) < 2:
        print("\nPre-registered rule needs both seeds; verdict deferred.")
        return
    diffs = [r["scaled"]["final_accuracy"] - r["baseline"]["final_accuracy"] for r in runs]
    rew = [r["scaled"]["late_reward"] - r["baseline"]["late_reward"] for r in runs]
    primary = all(d >= 0.02 for d in diffs)
    secondary = all(x > 0 for x in rew) and all(d >= 0 for d in diffs)
    verdict = "PROMISING" if (primary or secondary) else "NO EVIDENCE"
    print(f"\nPrimary (scaled at least 2 points better in every seed): {primary}")
    print(f"Secondary (late reward higher in every seed, accuracy not lower): {secondary}")
    print(f"Mean accuracy difference {100 * sum(diffs) / len(diffs):+.1f} points over {len(runs)} seeds")
    print(f"Pre-registered verdict: {verdict}")


if __name__ == "__main__":
    main()
