"""Pre-registration 30 (Amendment 4) and 34: frozen forecasts for the advanced starts, written before any Stage 2 run.

For every start X and the base B, on validation and test: the law's conditional curve f_X(u) (predicted pass@16 at
pass@1 u; law.py, tau = -0.20, prior and initial state from the first 32 of the 64 samples per problem, strict
reachability with tolerance 0.005), the uniform log-odds shift (beta 1) and affine map (beta 2.10, Pre-registration
17) from the same state, and the differenced forecast f_X(u) - f_B(u); a zero-shift predictor is f_X = f_B.
Validation grid: u = 0.40, 0.42, ..., 0.56, kept if inside B1k8's validation T = 1 trajectory range and at least
0.005 above both laws' initial pass@1 and below their final recorded value; the start's forecast shift is the
equal-weight mean of f_X - f_B over the kept grid ("no common support" if empty).
Features for rank comparison: changes from the base in pass@1, pass@32 and pass@64 (all 64 samples), validation
human-solution NLL (p30_nll.py output), and a power-map forecast: pass@16 at pass@1 u predicted as the start's own
64-sample pass@(a * 16^0.845) with a solving pass@a = u (a >= 1; beta 0.845 fitted on Pre-registration 21's Dr. GRPO
checkpoints by boptr.py), differenced against the base's.
Usage: python p30_forecast.py --start NAME=VAL_TEXTS,TEST_TEXTS ... --base VAL_TEXTS,TEST_TEXTS --nll FILE [--out F]
"""

import argparse
import glob
import gzip
import json
from math import comb
from pathlib import Path

import numpy as np

import law

HERE = Path(__file__).resolve().parent
TAU, BETA_AFF, BETA_PM, TOL = -0.20, 2.10, 0.845, 0.005
UGRID = [round(x, 2) for x in np.arange(0.20, 0.951, 0.01)]
VGRID = [round(x, 2) for x in np.arange(0.40, 0.561, 0.02)]


def first_k_counts(path, k=32):
    d = json.load(gzip.open(path, "rt"))
    n_prob = max(s["problem"] for s in d["samples"]) + 1
    seen, c64, ck = np.zeros(n_prob, int), np.zeros(n_prob, int), np.zeros(n_prob, int)
    for s in d["samples"]:  # samples are stored in sample order within each problem
        i = s["problem"]
        c64[i] += bool(s["reward"])
        if seen[i] < k:
            ck[i] += bool(s["reward"])
        seen[i] += 1
    assert (seen == 64).all(), path
    return ck, c64


def pass_k(c, n, k):
    return float(np.mean([1 - comb(n - int(x), k) / comb(n, k) for x in c]))


def law_curve(c32):
    th0 = law.initial_theta(c32, 32, law.fit_prior(c32, 32))
    tr = law.trajectory(th0, TAU)
    states, means = tr
    out = {}
    for u in UGRID:
        if means[0] + TOL <= u <= means[-1] - TOL:
            p = law.at_pass1(tr, u, strict=True)
            out[u] = {"law": law.pass_k_true(p, 16), "uniform": law.pass_k_true(law.uniform_shift(th0, u), 16),
                      "affine": law.pass_k_true(law.uniform_shift(th0, u, BETA_AFF), 16)}
        else:
            out[u] = None
    return out, float(means[0]), float(means[-1])


def power_map(c64, u):
    ks = np.unique(np.round(np.geomspace(1, 64, 96)).astype(int))
    vals = np.array([pass_k(c64, 64, int(k)) for k in ks])
    if not vals[0] <= u <= vals[-1]:
        return None
    a = float(np.exp(np.interp(u, vals, np.log(ks))))
    n = a * 16 ** BETA_PM
    return float(np.interp(np.log(n), np.log(ks), vals)) if 1 <= n <= 64 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", nargs="+", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--nll", nargs="*", default=[])
    ap.add_argument("--out", default=str(HERE / "FORECAST_30.json"))
    a = ap.parse_args()
    b1k8 = [np.array(json.load(open(f))["counts"]).mean() / 32
            for f in glob.glob(str(HERE / "results/worth-v15v-all-*/B1k8val/eval/step*_T1.00_eval.json"))]
    b0 = np.array(json.load(open(glob.glob(str(HERE / "results/worth-v15v-all-*/BASE15val/eval/step0_T1.00_eval.json"))[0]))["counts"]).mean() / 32
    vlo, vhi = min(b1k8 + [b0]), max(b1k8 + [b0])
    nll = {}
    for f in a.nll:
        nll.update({k: v["mean_token_nll"] for k, v in json.load(open(f)).items()})
    bval, btest = a.base.split(",")
    base = {"val": first_k_counts(bval), "test": first_k_counts(btest)}
    bcurve = {s: law_curve(base[s][0]) for s in base}
    out = {"tau": TAU, "beta_affine": BETA_AFF, "beta_power_map": BETA_PM, "b1k8_validation_range": [vlo, vhi],
           "base": {"files": [bval, btest], "nll": nll.get("base")}, "starts": {}}
    for spec in a.start:
        name, files = spec.split("=")
        fv, ft = files.split(",")
        rec = {"files": [fv, ft], "nll": nll.get(name)}
        for sset, f in (("val", fv), ("test", ft)):
            c32, c64 = first_k_counts(f)
            curve, m0, m1 = law_curve(c32)
            bc, bm0, bm1 = bcurve[sset]
            b32, b64 = base[sset]
            rows = []
            for u in UGRID:
                x, y = curve[u], bc[u]
                pm_x, pm_b = power_map(c64, u), power_map(b64, u)
                rows.append({"u": u, "f_X": x, "f_B": y,
                             "law_gap": None if x is None or y is None else x["law"] - y["law"],
                             "power_map_gap": None if pm_x is None or pm_b is None else pm_x - pm_b})
            feats = {"pass@1": 100 * (c64.mean() - b64.mean()) / 64,
                     "pass@32": 100 * (pass_k(c64, 64, 32) - pass_k(b64, 64, 32)),
                     "pass@64": 100 * (pass_k(c64, 64, 64) - pass_k(b64, 64, 64))}
            rec[sset] = {"law_initial_pass1": m0, "law_final_pass1": m1, "curve": rows, "feature_changes": feats,
                         "start_pass@1": 100 * c64.mean() / 64}
            if sset == "val":
                kept = [r for r in rows if r["u"] in VGRID and vlo <= r["u"] <= vhi and r["law_gap"] is not None]
                rec["validation_grid"] = [r["u"] for r in kept]
                rec["forecast_shift"] = (100 * float(np.mean([r["law_gap"] for r in kept]))) if kept else None
                rec["support"] = "ok" if kept else "no common support"
        out["starts"][name] = rec
        print(f"{name}: val start pass@1 {rec['val']['start_pass@1']:.1f}; grid {rec['validation_grid']}; "
              f"forecast shift {rec['forecast_shift'] if rec['forecast_shift'] is None else round(rec['forecast_shift'], 2)}; "
              f"test features {({k: round(v, 2) for k, v in rec['test']['feature_changes'].items()})}; NLL {rec['nll']}")
    Path(a.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
