# Results of Pre-registration 32: outcome-based exploration on the matched-pass@1 curve

Job worth-p32-oe-s0-20261029-005906; all six training runs and their evaluations exited 0.
Analysis: analyze_p32.py (committed before the results were fetched); output: results_p26/p32_analysis.txt.
Notation: OU = UCB-Con (within-group rarity bonus, c = 0.2, b0 = 0.5); OB = Batch (repetition penalty, c = 0.2);
B1k = Dr. GRPO seeds 8-10. Curve contrasts are in percentage points, averaged over eligible checkpoints within each
seed, then equally over seeds. Registered intervals use 2,000 shared bootstrap resamples of the 511 test problems,
with eligibility fixed on the full data, conditional on the trained seeds. Wording checked by Codex.

## Registered verdicts

| Arm | pass@16 gain vs B1k envelope | Unadjusted 90% CI | Unadjusted 95% CI | Per seed: estimate [90% CI], seeds 8, 9, 10 | Holm-adjusted p | Verdict |
|---|---|---|---|---|---|---|
| OU | +0.40 | [+0.07, +0.70] | [+0.02, +0.76] | +0.56 [+0.09, +1.03]; +0.35 [-0.16, +0.89]; +0.27 [-0.31, +0.81] | 0.072 | checkpoint-average equivalence within +/-2 points |
| OB | +0.01 | [-0.32, +0.35] | [-0.38, +0.39] | +0.10 [-0.39, +0.58]; -0.08 [-0.62, +0.49]; +0.01 [-0.61, +0.59] | 0.960 | checkpoint-average equivalence within +/-2 points |

All pooled and per-seed 90% intervals lie within +/-2. OU's positive unadjusted 95% interval does not establish the
registered gain verdict: its estimate is below +2 and its Holm-adjusted p exceeds 0.05. (Holm p-values test against
zero; they are not equivalence p-values.)

Maximum per-seed fractions of bootstrap comparison points clamped were 4.9% for OU and 9.0% for OB, below the 10%
guard. Against B1k's trajectory alone, point estimates round to +0.40 for OU (90% [+0.08, +0.70]) and +0.01 for OB
(90% [-0.31, +0.35]). The cross-seed Dr. GRPO trajectory diagnostic gives +0.12 with interval [+0.04, +0.23]
(diag_pass2.txt): a nonzero matched contrast between seeds of the same method; it supplies no quantified bias
correction for OU or OB.

## Manipulation check (test texts; not multiplicity-adjusted)

| Arm | Distinct valid incorrect classes per problem, minus B1k's same-seed trajectory at matched pass@1 | 90% CI | Conditional collision probability change (90% CI) |
|---|---|---|---|
| OU | +0.057 | [+0.014, +0.101] | -0.003 [-0.007, +0.000] |
| OB | -0.007 | [-0.049, +0.034] | -0.002 [-0.006, +0.001] |

OU meets the registered rule for an increase in distinct valid incorrect classes: its pooled, unadjusted 90%
interval is above zero. The increase is small; for scale, B1k seed 8 at step 150 has 6.18 distinct valid incorrect
classes per problem (computed from its test texts with analyze_p32.text_metrics). OB does not demonstrate an
increase, so its curve result is not read as a test of more diverse answers. Matching aggregate pass@1 does not
match the number of incorrect samples per problem; the conditional collision intervals reach or cross zero at the
reported precision. Step-150 no-parsed-box shares were 11.6% for OU, 11.4% for OB, and 11.5% for B1k; no interval
or equivalence test for these rates is reported.

## Against the reported magnitude (secondary)

pass@32 at matched pass@1: OU +0.31 [-0.18, +0.71], OB -0.18 [-0.73, +0.32] (90%). Both upper bounds are below
+1.7 (Batch minus GRPO in the paper's Table 4), so that fixed magnitude is excluded for our registered contrast; this
does not establish a cross-study difference in effects. Raw step-150 differences from B1k: pass@1 +0.28 (OU),
+0.08 (OB); pass@32 +0.39, -0.13.

## Reported

- Training (450 steps per arm across three seeds; 128 answers in 16 groups per step): OU averaged 86.4/128 nonzero
  bonuses and 42.9/128 positive bonuses; the mean applied exploration term c B_i over all answers was +0.0184.
  All-correct groups averaged 4.22/16, all 4.22 skipped. OB averaged 99.6/128 nonzero bonuses, all negative, and
  0.0/128 positive; mean c B_i was -0.0983. All-correct groups averaged 3.94/16, with 0.00 skipped. Distinct
  outcomes per group were 2.96 (OU) and 3.00 (OB); in all-wrong groups, 5.28 and 5.40. Training no-parsed-box
  counts averaged 13.4/128 (OU) and 13.9/128 (OB) per step. Mean first-block probe entropy was 0.3842 (OU) and
  0.3890 (OB). Training reward over steps 100-150 was 0.72-0.78 for every OU and OB seed, against 0.76 for B1k seed
  9 (the only B1k training log available locally; descriptive).
- Raw step-150 pass@16 differences in percentage points, arm minus its corresponding B1k seed, stratified by the
  base model's correct count c among 32 samples (entries ordered seeds 8, 9, 10): OU c = 0 (74 problems): +2.0,
  +3.2, -0.3; c = 1-4 (162): +0.9, +1.3, -2.1; c >= 5 (275): -0.1, -0.2, +0.2. OB c = 0: +2.6, +0.8, +0.2;
  c = 1-4: -0.7, +1.5, -3.1; c >= 5: -0.0, +0.1, +0.3. Descriptive only.

## Reading

Both arms meet the registered checkpoint-average equivalence rule within +/-2 percentage points, conditional on these
trained seeds; neither meets the registered gain or loss rule. Whole-trajectory equivalence remains unestablished.
OU's pass@16 estimate is +0.40 points (Holm-adjusted p = 0.072), and it demonstrates a small increase in the
registered distinct-class metric under the unadjusted 90% rule. OB does not demonstrate an increase in that metric;
its diversity effect remains uncertain. The cross-seed diagnostic does not quantify interpolation bias in either
arm's estimate. These findings concern the preregistered adaptation: each training problem is visited at most once;
c is transferred without strength matching; OU uses b0 = 0.5 rather than the paper's MATH value of 1; outcome
equivalence is narrower than math_verify; and training uses Dr. GRPO instead of GRPO with KL, and 1.5B instead of
7B.
