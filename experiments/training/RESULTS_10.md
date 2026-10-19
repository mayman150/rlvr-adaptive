# Pre-registration 10: results (October 19, 2026)

All 14 runs complete (seeds 5, 6, 7: PSR, NSR, PSR+N, NSR×2; seeds 5, 6: PSR×2). Full output:
`RESULTS_10_analysis_output.txt`. Verdicts below were applied by the script and checked
independently against the pre-registered rules (Codex, read-only).

| Rule | Result | Verdict |
| --- | --- | --- |
| 1. Manipulation check (PSR+N / same-policy NSR shadow, steps 0-49) | 1.02, 90% [1.01, 1.05] | pass |
| 2. Primary: phi(PSR+N), pooled, steps 25-49 | -0.11, 90% [-0.22, -0.05]; per seed +0.12, -0.16, -0.40 | trace-matched sign noise reproduces NSR's early accuracy lag; heterogeneity flagged (seed 7) |
| 3a. phi(NSR×2) | +1.16, 95% [1.07, 1.26] | doubling the learning rate recovers at least half of NSR's lag |
| 3b. Gap at matched 2e-6 (seeds 5, 6) | 4.7 points, 95% [2.7, 6.3]; 0.31 of the 1e-6 gap | NSR still lags at 2e-6 |
| 4. Average matched-accuracy equivalence vs PSR (pass@16, 128 problems) | PSR+N -0.37 [-1.13, +0.46]; NSR (5-7) -0.20 [-0.97, +0.55]; NSR×2 -0.03 [-0.88, +0.88]; NSR (3-7, secondary) -0.43 [-1.12, +0.21] | average equivalence for all; simultaneous bands exceed ±3, so checkpointwise equivalence is unresolved |
| 5. F0 / F1 persistence above 1.2 | 2.7-3.4 at step 0, about 1.4-1.6 at steps 50-150 | inconclusive (attenuation; no interval wholly below 1.2) |

Reported without a rule: once F0 and F1 are re-measured and interpolated, the theorem's
second moment matches the measured one within 0.69 to 1.18 in every 50-step window (0.18 to
0.89 with the calibration F's alone). Raw NSR / PSR second moment about 6; after division by
Adam's previous per-coordinate scale 1.04 to 1.32; lagged projection of Adam's update on the
population-gradient reference, NSR / PSR, 0.73 to 0.87 (an unnormalized projection, not a
cosine or accuracy progress). No formatting flags; clipping never active.

Accepted wording (Codex): Across three fresh Qwen2.5-Math-1.5B seeds, same-policy
second-moment matching passed, and injected sign noise reproduced NSR's early accuracy lag
within the preregistered pooled margin (phi = -0.11, 90% interval [-0.22, -0.05]). PSR+N was
slightly slower than NSR, and seed heterogeneity limits claims of uniform agreement. Doubling
NSR's learning rate recovered the lag relative to PSR at its original rate, while NSR still
lagged at the matched doubled rate. PSR+N and NSR satisfied average matched-accuracy pass@16
equivalence against PSR, while checkpointwise equivalence remained unresolved. Gradient
asymmetry attenuated, persistence was inconclusive, and updated gradient-norm measurements
improved second-moment agreement. These conditional results show that added noise can
reproduce the lag here, without uniquely identifying scalar noise size as its mechanism.

Codex's updated score: 5/10 (borderline reject), about 6/10 with a clear OLMo replication.
