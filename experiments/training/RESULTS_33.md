# Results of Pre-registration 33: sensitivity of Pre-registration 28's 7B readings

Job worth-p33-sens7-s0-20261029-064449 (us-west-2); all 42 steps exited 0. Analysis: analyze_p33.py (committed before
the results were fetched; two later commits changed only its speed, with identical values); output:
results_p26/p33_analysis.txt. Gains are pass@16 minus the reference at the checkpoint's pass@1, in percentage
points; 90% intervals from 2,000 problem-bootstrap draws per cohort, conditional on the trained reference seeds.
Families and verdicts as registered (departure: |gain| >= 2 and Holm-adjusted p <= 0.05; equivalence: 90% interval
within +/-2, pointwise and unadjusted; otherwise inconclusive).

## Reproducibility (B7e step 25 minus B7 step 25)

| Cohort | Seed | pass@1 | pass@16 | Flag |
|---|---|---|---|---|
| 511 | 8 | -0.91 [-1.51, -0.32] | -0.39 [-1.30, +0.46] | no |
| 511 | 9 | +0.07 [-0.49, +0.67] | +0.34 [-0.60, +1.35] | no |
| hard | 8 | -1.52 [-2.40, -0.65] | -0.55 [-2.24, +1.19] | no |
| hard | 9 | +0.38 [-0.43, +1.20] | +1.38 [-0.29, +3.09] | yes (pass@16 > 1) |

The flag on the hard cohort means the primary (dense) and the splice readings carry equal weight there; for the six
scorable hard-cohort checkpoints the two agree within 0.26 points.

## A1: 511 test problems against the dense reference

| Checkpoint (label) | pass@1 | Dense gain [90%] | Holm p | Verdict | Sparse gain |
|---|---|---|---|---|---|
| LUFFY on-policy baseline (correctness-only) | 46.9 | +0.12 [-1.10, +1.31] | 1.000 | equivalent | +0.71 |
| One-shot RLVR pi1 (correctness-only) | 48.4 | +1.31 [+0.42, +2.24] | 0.077 | inconclusive | +1.82 |
| ExGRPO Zero (correctness-only) | 49.5 | +0.65 [-0.42, +1.63] | 1.000 | equivalent | +1.09 |
| LUFFY (external) | 56.4 | -0.40 [-1.53, +0.69] | 1.000 | equivalent | -0.30 |
| Random reward, lr 5e-7 (random reward) | 57.3 | +0.62 [-0.14, +1.40] | 0.985 | equivalent | +0.71 |
| ExGRPO-LUFFY continual (external lineage) | 62.8 | +0.52 [-0.50, +1.58] | 1.000 | equivalent | +0.51 |
| SimpleRL-Zero (correctness-only) | 68.8 | -7.46 [-9.22, -5.91] | <0.001 | departure down | -7.46 |
| SimpleRL-Zoo (correctness-only) | 70.3 | -0.89 [-1.74, -0.27] | 0.142 | equivalent | -0.89 |

Six checkpoints (Oat-Zero, RAFT++, the ground-truth reproduction, both DeepMath models, ReLIFT) lie above the
reference's pass@1 range: insufficient overlap. Tally: 6 equivalent, 1 inconclusive, 1 departure down, 6 insufficient
overlap; no upward departure. The dense knots lowered every gain in the former gap (pass@1 46-57) by 0.1-0.6 points
(paired changes, e.g. one-shot RLVR -0.51 [-1.22, +0.20]); checkpoints above 62 are unchanged.

## A2: hard cohort (253 problems) against the dense reference

| Checkpoint (label) | pass@1 | Dense gain [90%] | Holm p | Verdict | Sparse gain (paired change) |
|---|---|---|---|---|---|
| LUFFY on-policy baseline | 30.2 | -0.22 [-2.61, +2.08] | 0.844 | inconclusive | +1.85 (-2.06 [-3.23, -0.82]) |
| One-shot RLVR pi1 | 29.6 | +1.82 [+0.20, +3.45] | 0.272 | inconclusive | +3.92 (-2.09 [-3.27, -0.87]) |
| ExGRPO Zero | 31.2 | +2.63 [+0.62, +4.66] | 0.150 | inconclusive | +4.65 (-2.02 [-3.15, -0.79]) |
| LUFFY | 37.0 | +2.24 [+0.36, +4.06] | 0.255 | inconclusive | +3.47 (-1.23 [-2.25, -0.24]) |
| Random reward, lr 5e-7 | 38.9 | +1.64 [-0.07, +3.32] | 0.345 | inconclusive | +2.47 (-0.82 [-1.81, +0.03]) |
| ExGRPO-LUFFY continual | 44.1 | +1.41 [-0.35, +4.07] | 0.356 | inconclusive | +1.43 (-0.02 [-0.51, +0.06]) |

Eight checkpoints lie above the hard-cohort reference's pass@1 range. Tally: 6 inconclusive, 8 insufficient overlap;
no departure in either direction. Against the sparse reference, these hard-cohort gains would have been 1.2-2.1
points larger: the original reference's chord overstated them.

## B2: hard cohort against the frozen law from the base's hard-cohort counts

All 14 checkpoints are scorable. Three depart downward: SimpleRL-Zero -13.66 [-17.48, -10.16] (41% truncated),
Oat-Zero -5.81 [-8.86, -2.92] (Holm p 0.024), the ground-truth reproduction -5.70 [-8.68, -2.90] (Holm p 0.013).
The other eleven are inconclusive, with estimates from -4.26 to -0.83; none is positive. DeepMath RePO minus GRPO on
the hard cohort: +0.48 [-1.81, +2.92].

## Diagnostic (not a verdict): pass@2 at matched pass@1 on the 511 problems

Under the dense reference, the pass@2 gains of checkpoints in the former gap fell, for example LUFFY's on-policy
baseline from +1.14 to -0.78 and ExGRPO Zero from +1.10 to -0.65: much of the spread difference reported in
diag_pass2_public.txt for these checkpoints came from the sparse reference.

## Reading

No interpretable comparison met the registered upward-departure criterion in any family; this is a scoped robustness
statement that coexists with inconclusive, unsupported and failed comparisons and does not establish equivalence.
Densifying the early 7B reference lowered gains in the former pass@1 gap, slightly on the 511 problems (0.1-0.6
points) and substantially on the hard cohort (about 1-2 points), which confirms that the sparse chord biased those
readings upward. On the hard cohort, the empirical reference (positive point estimates for five of six scorable
checkpoints, all inconclusive) and the frozen law (negative estimates for all fourteen) disagree by roughly 3-4
points at matched pass@1, so the hard cohort does not discriminate sharply between them at this precision. Scope:
two reference seeds, one budget (2,048 tokens), one protocol.
