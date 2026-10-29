All transcribed estimates, intervals, p-values, raw differences, strata, and rounded training statistics match their sources. Both primary equivalence verdicts, OU’s manipulation-check verdict, OB’s “increase not demonstrated” verdict, and both +1.7 exclusions follow the registered rules.

The 6.2-class reference value—and the derived “about 1%”—cannot be verified from the supplied sources. The main problems are overclaims and omitted diagnostics.

No files changed. All line numbers below refer to the original [RESULTS_32.md](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/RESULTS_32.md).

1. **Lines 3–7:** The supplied sources do not document the analyzer’s freeze timing. Specify eligible checkpoints and bootstrap resampling.

   ```markdown
   Job worth-p32-oe-s0-20261029-005906; all six training runs and their evaluations exited 0.
   Analysis: analyze_p32.py; output: results_p26/p32_analysis.txt.
   Notation: OU = UCB-Con (within-group rarity bonus, c = 0.2, b0 = 0.5);
   OB = Batch (repetition penalty, c = 0.2); B1k = Dr. GRPO seeds 8-10.
   Curve contrasts are in percentage points, averaged over eligible checkpoints within each seed,
   then equally over seeds. Registered intervals use 2,000 shared bootstrap resamples of
   the 511 test problems, with eligibility fixed on the full data, conditional on the trained seeds.
   ```

2. **Lines 11–14:** Both labels are correct shorthand. Use the full registered label and show the per-seed intervals required for equivalence. Holm p-values test against zero; they are not equivalence p-values.

   ```markdown
   | Arm | pass@16 gain vs B1k envelope | Unadjusted 90% CI | Unadjusted 95% CI | Per seed: estimate [90% CI], seeds 8, 9, 10 | Holm-adjusted p | Verdict |
   |---|---|---|---|---|---|---|
   | OU | +0.40 | [+0.07, +0.70] | [+0.02, +0.76] | +0.56 [+0.09, +1.03]; +0.35 [-0.16, +0.89]; +0.27 [-0.31, +0.81] | 0.072 | checkpoint-average equivalence within +/-2 points |
   | OB | +0.01 | [-0.32, +0.35] | [-0.38, +0.39] | +0.10 [-0.39, +0.58]; -0.08 [-0.62, +0.49]; +0.01 [-0.61, +0.59] | 0.960 | checkpoint-average equivalence within +/-2 points |
   ```

   All pooled and per-seed 90% intervals lie within ±2. OU’s positive unadjusted 95% interval does not establish the registered gain verdict: its estimate is below +2 and its Holm-adjusted p-value exceeds 0.05.

3. **Lines 16–18:** Trajectory-only estimates agree at the printed precision, but their intervals differ. The diagnostic does not quantify how much of OU’s estimate is interpolation bias.

   ```markdown
   Maximum per-seed fractions of bootstrap comparison points clamped were 4.9% for OU and 9.0%
   for OB, below the 10% guard. Against B1k's trajectory alone, point estimates round to
   +0.40 for OU (90% [+0.08, +0.70]) and +0.01 for OB (90% [-0.31, +0.35]).
   The cross-seed Dr. GRPO trajectory diagnostic gives +0.12 with reported interval
   [+0.04, +0.23] (diag_pass2.txt). This illustrates a nonzero matched contrast between
   seeds using the same training method; it supplies no quantified bias correction for OU or OB.
   ```

4. **Lines 22–29:** Remove the unsupported relative magnitude, distinguish distinct-class counts from conditional diversity, and replace “unchanged” parsing rates with descriptive reporting.

   ```markdown
   | Arm | Distinct valid incorrect classes per problem, minus B1k's same-seed trajectory at matched pass@1 | 90% CI | Conditional collision probability change (probability units; 90% CI) |
   |---|---|---|---|
   | OU | +0.057 | [+0.014, +0.101] | -0.003 [-0.007, +0.000] |
   | OB | -0.007 | [-0.049, +0.034] | -0.002 [-0.006, +0.001] |

   OU meets the registered rule for an increase in distinct valid incorrect classes:
   its pooled, unadjusted 90% interval is above zero. OB does not demonstrate an increase,
   so its curve result is not read as a test of more diverse answers. Matching aggregate
   pass@1 does not match the number of incorrect samples per problem; the conditional
   collision intervals reach or cross zero at the reported precision.
   Step-150 no-parsed-box shares were 11.6% for OU, 11.4% for OB, and 11.5% for B1k;
   no interval or equivalence test for these rates is reported.
   ```

5. **Lines 40–43:** The existing rounded numbers are correct. Add the preregistered OB all-correct group statistics, training no-parsed-box counts, and first-block probe entropy. Remove “tracked B1k throughout (no slowdown),” which the supplied sources do not substantiate.

   ```markdown
   - Training (450 steps per arm across three seeds; 128 answers in 16 groups per step):
     OU averaged 86.4/128 nonzero bonuses and 42.9/128 positive bonuses; the mean applied
     exploration coefficient c B_i over all answers was +0.0184. All-correct groups
     averaged 4.22/16, all 4.22 skipped. OB averaged 99.6/128 nonzero bonuses, all negative,
     and 0.0/128 positive; mean c B_i was -0.0983. All-correct groups averaged 3.94/16,
     with 0.00 skipped. Distinct outcomes per group were 2.96 (OU) and 3.00 (OB);
     in all-wrong groups, 5.28 and 5.40. Training no-parsed-box counts averaged
     13.4/128 (OU) and 13.9/128 (OB) per step. Mean first-block probe entropy
     was 0.3842 (OU) and 0.3890 (OB).
   ```

6. **Line 44:** Define the stratum count and make the same-seed, raw comparison explicit. All stratum values on lines 44–46 are correct.

   ```markdown
   - Raw step-150 pass@16 differences in percentage points, arm minus its corresponding B1k seed, stratified by the base model's correct count c among 32 samples (entries ordered seeds 8, 9, 10): OU c = 0: +2.0, +3.2, -0.3; c = 1-4:
   ```

7. **Lines 50–55:** “Neither moves the curve” overstates checkpoint-average equivalence; “OB did not change measured diversity” overstates failure to demonstrate an increase. Remove the bias attribution and change “visited once” to “at most once.”

   ```markdown
   Both arms meet the registered checkpoint-average equivalence rule within +/-2 percentage
   points, conditional on these trained seeds; neither meets the registered gain or loss rule.
   Whole-trajectory equivalence remains unestablished. OU's pass@16 estimate is +0.40 points
   (Holm-adjusted p = 0.072), and it demonstrates a small increase in the registered
   distinct-class metric under the unadjusted 90% rule. OB does not demonstrate an increase
   in that metric; its diversity effect remains uncertain. The cross-seed diagnostic does
   not quantify interpolation bias in either arm's estimate.
   These findings concern the preregistered adaptation: each training problem is visited
   at most once; c is transferred without strength matching; OU uses b0 = 0.5 rather than
   the paper's MATH value of 1; outcome equivalence is narrower; and training uses
   Dr. GRPO instead of GRPO with KL, and 1.5B instead of 7B.
   ```