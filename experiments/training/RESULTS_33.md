# Results of Pre-registration 33: sensitivity of Pre-registration 28's 7B readings

Job worth-p33-sens7-s0-20261029-064449 (us-west-2); all 42 steps exited 0 (the job's status.json). Analysis:
analyze_p33.py (committed before the results were fetched; two later commits changed only its speed, with identical
values). Analysis output: results_p26/p33_analysis.txt. Gains are pass@16 minus the reference at the checkpoint's
pass@1, in percentage points; 90% intervals from 2,000 problem-bootstrap draws per cohort shared across comparisons
on that cohort, conditional on the trained reference seeds. Holm correction is applied within each of the three
primary families. Departure requires |gain| >= 2 and Holm-adjusted p <= 0.05; equivalence requires the 90% interval
within +/-2, pointwise and unadjusted; otherwise the verdict is inconclusive. Comparisons with more than 10% of
bootstrap draws outside support are not interpreted; truncation above 90% is a protocol failure. Wording checked by
Codex (codex_results33_check.md).

## Reproducibility (B7e step 25 minus B7 step 25)

| Cohort | Seed | pass@1 | pass@16 | Flag |
|---|---|---|---|---|
| 511 | 8 | -0.91 [-1.51, -0.32] | -0.39 [-1.30, +0.46] | no |
| 511 | 9 | +0.07 [-0.49, +0.67] | +0.34 [-0.60, +1.35] | no |
| hard | 8 | -1.52 [-2.40, -0.65] | -0.55 [-2.24, +1.19] | no |
| hard | 9 | +0.38 [-0.43, +1.20] | +1.38 [-0.29, +3.09] | yes |

A cohort is flagged when either seed's absolute step-25 point difference exceeds 3 points in pass@1 or 1 point in
pass@16; the interval endpoints do not determine the flag. Only hard-cohort seed 9 is flagged (pass@16 +1.38), so
the dense, splice and supported B7e-only early readings receive equal weight in the hard-cohort discussion. Dense and
splice point gains differ by up to 0.37 points (LUFFY +2.24 vs +2.50, random-reward reproduction +1.64 vs +2.01,
ExGRPO-LUFFY continual +1.41 vs +1.43; the other three coincide at the printed precision); the early reference
coincides with the dense gains for the five checkpoints it covers and does not cover the continual checkpoint.
Sensitivity intervals and Holm p-values are not printed, so sensitivity verdicts are not given; the random-reward
splice estimate reaches the 2-point size threshold, which alone does not establish a departure.

## A1: 511 test problems against the dense reference

| Checkpoint (label) | pass@1 | Dense gain [90%] | Holm p | Dense verdict | Sparse gain |
|---|---|---|---|---|---|
| LUFFY on-policy baseline (correctness-only) | 46.9 | +0.12 [-1.10, +1.31] | 1.000 | equivalent | +0.71 |
| One-shot RLVR pi1 (correctness-only) | 48.4 | +1.31 [+0.42, +2.24] | 0.077 | inconclusive | +1.82 |
| ExGRPO Zero (correctness-only) | 49.5 | +0.65 [-0.42, +1.63] | 1.000 | equivalent | +1.09 |
| LUFFY (external) | 56.4 | -0.40 [-1.53, +0.69] | 1.000 | equivalent | -0.30 |
| Random-reward reproduction (random reward) | 57.3 | +0.62 [-0.14, +1.40] | 0.985 | equivalent | +0.71 |
| ExGRPO-LUFFY continual (external lineage) | 62.8 | +0.52 [-0.50, +1.58] | 1.000 | equivalent | +0.51 |
| SimpleRL-Zero (correctness-only) | 68.8 | -7.46 [-9.22, -5.91] | <0.001 | departure down | -7.46 |
| SimpleRL-Zoo (correctness-only) | 70.3 | -0.89 [-1.74, -0.27] | 0.142 | equivalent | -0.89 |

Six checkpoints (Oat-Zero, RAFT++, the ground-truth reproduction, both DeepMath models, ReLIFT) are reported as having
insufficient overlap and are not interpreted. Tally: 6 equivalent, 1 inconclusive, 1 departure down, 6 insufficient
overlap; no upward departure. For the five checkpoints at pass@1 46.9-57.3%, the paired dense-minus-sparse gain
changes range from -0.59 to -0.10 points; all five 90% change intervals include zero (one-shot RLVR: -0.51
[-1.22, +0.20]). Among the three higher-pass@1 scorable checkpoints, SimpleRL-Zero and SimpleRL-Zoo have changes
+0.00 [+0.00, +0.00] and ExGRPO-LUFFY continual +0.01 [-0.51, +0.49].

## A2: hard cohort (253 problems) against the dense reference

| Checkpoint | pass@1 | Dense gain [90%] | Holm p | Dense verdict | Sparse gain (dense minus sparse [90%]) |
|---|---|---|---|---|---|
| LUFFY on-policy baseline | 30.2 | -0.22 [-2.61, +2.08] | 0.844 | inconclusive | +1.85 (-2.06 [-3.23, -0.82]) |
| One-shot RLVR pi1 | 29.6 | +1.82 [+0.20, +3.45] | 0.272 | inconclusive | +3.92 (-2.09 [-3.27, -0.87]) |
| ExGRPO Zero | 31.2 | +2.63 [+0.62, +4.66] | 0.150 | inconclusive | +4.65 (-2.02 [-3.15, -0.79]) |
| LUFFY | 37.0 | +2.24 [+0.36, +4.06] | 0.255 | inconclusive | +3.47 (-1.23 [-2.25, -0.24]) |
| Random-reward reproduction | 38.9 | +1.64 [-0.07, +3.32] | 0.345 | inconclusive | +2.47 (-0.82 [-1.81, +0.03]) |
| ExGRPO-LUFFY continual | 44.1 | +1.41 [-0.35, +4.07] | 0.356 | inconclusive | +1.43 (-0.02 [-0.51, +0.06]) |

Eight checkpoints are reported as having insufficient overlap and are not interpreted. Tally: 6 inconclusive, 8
insufficient overlap; no departure in either direction and no equivalence verdict. The paired dense-minus-sparse
changes, in table order, are -2.06, -2.09, -2.02, -1.23, -0.82 and -0.02 points (reductions of 0.02-2.09); their 90%
intervals exclude zero for the first four and include zero for the random-reward reproduction and the continual
checkpoint.

## B2: hard cohort against the frozen law from the base's hard-cohort counts

All 14 checkpoints are scorable against the frozen law (tau = -0.20). Three correctness-only checkpoints have verdict
"departure down": SimpleRL-Zero -13.66 [-17.48, -10.16] (Holm p < 0.001; 41% truncation), Oat-Zero -5.81
[-8.86, -2.92] (Holm p 0.024) and the ground-truth reproduction -5.70 [-8.68, -2.90] (Holm p 0.013). The other eleven
are inconclusive, with estimates from -4.26 to -0.83; none is positive. The paired difference in frozen-law gains,
DeepMath RePO minus GRPO on the hard cohort, is +0.48 [-1.81, +2.92] (90%).

| Training-data label | A1 | A2 | B2 |
|---|---|---|---|
| correctness-only (LUFFY's on-policy baseline) | 1 equivalent | 1 inconclusive | 1 inconclusive |
| correctness-only | 2 equivalent; 1 inconclusive; 1 departure down; 5 insufficient overlap | 2 inconclusive; 7 insufficient overlap | 6 inconclusive; 3 departure down |
| external | 1 equivalent; 1 insufficient overlap | 1 inconclusive; 1 insufficient overlap | 2 inconclusive |
| random reward | 1 equivalent | 1 inconclusive | 1 inconclusive |
| external lineage (start is not the base) | 1 equivalent | 1 inconclusive | 1 inconclusive |

Additional paired gain-change intervals, support/drop rates and truncation rates for scorable comparisons are in the
analysis output. Boxed rates, requested by the registration, are absent from that output.

## Diagnostic (not a verdict): pass@2 at matched pass@1 on the 511 problems

Replacing the sparse reference with the dense composite reference changed the matched-pass@1 pass@2 gain from +1.14
to -0.78 for LUFFY's on-policy baseline and from +1.10 to -0.65 for ExGRPO Zero. These are descriptive sensitivity
estimates; the composite reference changes both the early knots and the step-25 replicate.

## Reading

None of the 28 interpreted comparisons in the three primary families met the registered upward-departure criterion.
Across those families there are 6 equivalent, 18 inconclusive and 4 departure-down verdicts, plus 14 comparisons with
insufficient overlap; no protocol-failure verdict is reported. The absence of upward departures does not establish
general equivalence. Using the dense composite reference reduced the point gains for the five 511-problem
checkpoints at pass@1 46.9-57.3% by 0.10-0.59 points, and for the six scorable hard-cohort checkpoints by 0.02-2.09
points; these estimates show sensitivity to reference construction, which changes both the early knots and the
step-25 replicate. On the hard cohort, dense and splice gains are positive for five of six scorable checkpoints, and
all six dense verdicts are inconclusive; frozen-law gains are negative for all fourteen checkpoints, with three
downward departures. For the six checkpoints covered by both, dense-minus-law gains range from about +2.24 to +4.09
points (splice-minus-law +2.26 to +4.27, from rounded estimates); paired intervals for these between-reference
contrasts are not reported. Scope: two reference seeds, one budget (2,048 tokens), one protocol.
