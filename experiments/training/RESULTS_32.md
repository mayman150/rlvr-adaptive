# Results of Pre-registration 32: outcome-based exploration on the matched-pass@1 curve

Job worth-p32-oe-s0-20261029-005906 (us-west-2), all six runs and their evaluations exit 0. Analysis: analyze_p32.py
(frozen before results), output results_p26/p32_analysis.txt. Notation: OU = UCB-Con (within-group rarity bonus,
c = 0.2, b0 = 0.5), OB = Batch (repetition penalty, c = 0.2); B1k = Dr. GRPO seeds 8-10; gains in percentage points
of pass@16 at matched pass@1, checkpoint-averaged within seed, seeds weighted equally; intervals from 2,000 problem
draws, conditional on the trained seeds.

## Registered verdicts

| Arm | pass@16 gain vs B1k envelope | 90% | 95% | per seed | Holm p | Verdict |
|---|---|---|---|---|---|---|
| OU | +0.40 | [+0.07, +0.70] | [+0.02, +0.76] | +0.56, +0.35, +0.27 | 0.072 | average equivalence |
| OB | +0.01 | [-0.32, +0.35] | [-0.38, +0.39] | +0.10, -0.08, +0.01 | 0.960 | average equivalence |

Clamping at most 9.0% (OB seed 10), below the 10% guard. Against B1k's trajectory alone the estimates are the same
(+0.40, +0.01). For scale: one B1k seed against the other seeds' trajectories gives +0.12 [+0.04, +0.23]
(diag_pass2.txt), part of which is interpolation bias, so OU's +0.40 is not wholly a method effect.

## Manipulation check (test texts; not multiplicity-adjusted)

| Arm | distinct valid incorrect classes per problem, minus B1k at matched pass@1 | 90% | collision probability change |
|---|---|---|---|
| OU | +0.057 | [+0.014, +0.101] | -0.003 [-0.007, +0.000] |
| OB | -0.007 | [-0.049, +0.034] | -0.002 [-0.006, +0.001] |

OU raised the registered diversity metric, by about 1% of B1k's level (6.2 distinct incorrect classes per problem
at step 150 for seed 8). For OB an increase has not been demonstrated, so its curve result is not read as a test of
more diverse answers. The no-parsed-box share at step 150 is unchanged (11.6% OU, 11.4% OB, 11.5% B1k).

## Against the reported magnitude (secondary)

pass@32 at matched pass@1: OU +0.31 [-0.18, +0.71], OB -0.18 [-0.73, +0.32] (90%). Both upper bounds are below
+1.7 (Batch minus GRPO in the paper's Table 4), so that fixed magnitude is excluded for our registered contrast; this
does not establish a cross-study difference in effects. Raw step-150 differences from B1k: pass@1 +0.28 (OU),
+0.08 (OB); pass@32 +0.39, -0.13.

## Reported

- Training: OU gave nonzero bonuses to 86 of 128 answers per step (43 positive; mean +0.018); 4.2 of 16 groups per
  step were all correct and skipped. OB gave 100 of 128 nonzero (all negative; mean -0.098, including all-correct
  groups). Distinct outcomes per group 2.96 (OU) and 3.00 (OB); in all-wrong groups 5.28 and 5.40. Training reward
  tracked B1k throughout (no slowdown).
- pass@16 at step 150 by base-count stratum, arm minus B1k (seeds 8, 9, 10): OU c = 0: +2.0, +3.2, -0.3; c = 1-4:
  +0.9, +1.3, -2.1; c >= 5: -0.1, -0.2, +0.2. OB c = 0: +2.6, +0.8, +0.2; c = 1-4: -0.7, +1.5, -3.1; c >= 5: -0.0,
  +0.1, +0.3. Single-checkpoint strata are noisy (74 problems with c = 0) and are descriptive only.

## Reading

Neither outcome-based bonus moves the matched-pass@1 curve in this adaptation; both are within +/-2 of Dr. GRPO.
OU measurably but slightly increased test-time wrong-answer diversity, with a small pass@16 estimate (+0.40) that is
below the registered margin and partly explained by the interpolation bias seen between Dr. GRPO seeds. OB did not
change measured diversity. The scope limits stated in the registration apply: no historical exploration (each problem
visited once), a coefficient transferred rather than strength-matched, Dr. GRPO instead of GRPO with KL, and 1.5B
instead of 7B.
