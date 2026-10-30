All experimental numbers and verdict labels match the analysis output, including the step-150 ranges. Reporting `<0.001` for the output’s rounded `0.000` is appropriate. The corrections concern precision, missing reporting, and overclaims. Line numbers below refer to the original results document.

1. **[Lines 3–10](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/RESULTS_30_stage2.md:3): distinguish registered settings from verified execution, specify nominal dose and averaging.** Neither comparison file substantiates the job’s 31 successful steps or the analysis commit/review chronology. Amendment 4 explicitly calls 25K nominal and directs actual counts to metadata. Replace with:

   ```text
   Analysis output: results_p26/p30_stage2_analysis.txt. The preregistration specifies 2,000 joint
   problem-bootstrap draws. R25 / H25 denote starts at the nominal 25K presented-token SFT dose,
   using SFT seed 3030 and 262 problems per selection from Amendment 3's teacher pool. R is a
   subject-stratified random selection among problems with surviving verifier-correct teacher
   solutions; H comprises such problems with base success count c <= 4 of 64. Amendment 4 chose
   25K through its validation-only fallback, minimizing the summed distances of R's and H's
   validation pass@1 from 0.40; this is recorded as a deviation from the primary dose rule.

   Observed gaps, in percentage points, are pass@16 minus the same-seed T = 1 B1k trajectory
   interpolated at the checkpoint's observed pass@1, averaged over eligible checkpoints within
   seed and then equally over seeds 8, 9 and 10. Full-data eligible checkpoints are 1, 2, 5, 10,
   25 and 50; steps 100 and 150 lie beyond B1k's support. Intervals condition on the trained
   starts and seeds. P34 is unavailable in this output because all seed-10 checkpoints are missing.
   ```

2. **[Lines 14 and 21](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/RESULTS_30_stage2.md:14): make units and seed order explicit.** Replace line 14 with:

   ```text
   | Start | pass@1 (%) | pass@16 (%) | pass@64 (%) | Truncated | Gap vs B1k at matched pass@1 (pp; seeds 8 / 9 / 10) |
   ```

   Replace line 21 with:

   ```text
   | Contrast | Estimate (pp) | 90% CI | 95% CI | Per-seed estimates (pp; 8 / 9 / 10) | Draws affected (8 / 9 / 10) | Holm p | Verdict |
   ```

3. **[Line 25](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/RESULTS_30_stage2.md:25): “as above” is ambiguous because the immediately preceding row is R25.** Replace that cell with:

   ```text
   7.2%, 8.3%, 6.7%
   ```

   For consistency with the explicit seed order, replace line 27’s `0%` cell with:

   ```text
   0.0%, 0.0%, 0.0%
   ```

4. **[Lines 30–35](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/RESULTS_30_stage2.md:30): state the operational rules, include the required residual threshold p-value, and distinguish support failure from an unscorable forecast.** The forecast interval’s confidence level is unspecified in [analysis line 18](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/results_p26/p30_stage2_analysis.txt:18). The rules are in [Readings](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_30.md:79). Replace with:

   ```text
   H25's mean observed gap is positive: its 95% interval is above 0, all three seed means are
   positive, and Holm p = 0.002. "Curve raised" additionally requires an estimated mean gap
   >= +2 points; the estimate +1.92 therefore receives the registered verdict "inconclusive".
   The one-sided bootstrap p-value for |gap| > 2 is 0.560 for the trajectory contrast and
   0.559 for the envelope contrast.

   The law forecast a mean H25 gap of +6.69 (reported interval [+5.23, +8.13], confidence level
   unspecified in the output), overestimating the observed gap. The residual -4.77 meets
   "forecast fails (bias detected)": |residual| >= 2 and the two-sided test of zero residual
   has Holm p <= 0.05 (here <0.001). This operational label uses the estimated magnitude and
   a test of zero bias; the separately reported one-sided p-value for |residual| > 2 is 0.003.

   The registered analysis requires at least two eligible checkpoints per seed. Checkpoints
   leaving reference or relevant law support are dropped within bootstrap draws; more than
   10% affected draws makes that seed's verdict "not interpreted". R25's trajectory and
   envelope contrasts exceed this limit in every seed. Its forecast residual is separately
   "unscorable": the full-data eligible seed-10, step-2 checkpoint has no forecast.
   The H25-minus-R25 targeting comparison is consequently not interpreted. The registered
   targeting rule requires an estimated difference >= 2 points with a 95% interval above 0.
   ```

   The output establishes the affected-draw rates; it does not identify which checkpoints were dropped in each draw. Avoid attributing all losses specifically to step 50.

5. **[Lines 37–43](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/RESULTS_30_stage2.md:37): qualify preservation as descriptive and distinguish checkpoint estimates from intervals.** Every listed checkpoint number is correct. However, H25 seed 9 at step 5 has a pointwise 90% interval crossing zero, and no interval for post-RL minus step-0 offset is reported. Replace with:

   ```text
   ## Per-checkpoint gaps (descriptive point estimates; full pointwise 90% intervals in the analysis output)

   H25, steps 1, 2, 5, 10, 25, 50: seed 8 +2.0, +2.1, +1.9, +1.5, +1.6, +2.0;
   seed 9 +1.4, +2.1, +1.2, +2.2, +2.1, +2.0;
   seed 10 +2.0, +2.7, +1.8, +1.9, +2.0, +2.1.
   R25 point estimates range from +0.5 to +1.5 across these checkpoints and seeds.
   H25's seed-9, step-5 pointwise 90% interval is [-0.2, +2.6]; the aggregate positive-gap
   result does not establish a positive gap at every individual checkpoint.

   Descriptively, the post-RL point estimates are close to each start's step-0 offset.
   Preservation or equivalence of these offsets has not been tested. The primary matched
   analysis covers early RL through step 50 and does not establish persistence through
   step 150. Beyond B1k's support, the descriptive seed-8 comparison with P27's LH1
   reference gives H25 +1.5 at step 100 and +1.1 at step 150; R25 +1.0 and +0.6, respectively.
   ```

6. **[Lines 47–49](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/RESULTS_30_stage2.md:47): identify percentages and the source of the zero-count stratum.** The ranges are all correct. Replace with:

   ```text
   Across R25 and H25 and seeds 8-10, step-150 pass@1 is 64.4-65.4%, versus 57.5-58.1% for
   B1k; pass@16 is 87.7-88.9%, versus 86.4-87.2%. For the SFT starts, truncation is
   19.6-22.4% and boxed-answer rate is 87.2-87.9%. On the 51 test problems with zero successes
   in the Stage 1 base's 64-sample evaluation, pass@16 is 27.9-30.9% for the SFT starts and
   15.2-23.9% for B1k. These endpoint comparisons are descriptive and unmatched in pass@1.
   ```

7. **[Lines 53–59](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/RESULTS_30_stage2.md:53): the law forecast the direction, but overestimated the magnitude; avoid implying checkpoint-wide significance or established preservation.** The asserted three-way-review agreement is also unsupported by the supplied comparison files. Replace with:

   ```text
   In this regime (1.5B, MATH, 1,024 tokens, Dr. GRPO, three RL seeds from one SFT start per
   selection), H25 has a positive mean pass@16 gap over same-seed Dr. GRPO-from-base at
   matched pass@1 across eligible checkpoints 1-50: +1.92 points, 95% interval [+0.68, +3.12],
   Holm p = 0.002. Its registered "curve raised" verdict is inconclusive because the estimate
   is below +2 points. The checkpoint point estimates are descriptively close to the start's
   pre-RL offset; preservation of that offset has not been established by a paired comparison.

   The frozen per-problem law predicted a positive mean shift but overestimated it
   (+6.69 forecast versus +1.92 observed), yielding the registered verdict "forecast fails
   (bias detected)". Remaining questions concern the roles of truncation, formatting and
   reasoning content, reproducibility across SFT seeds, and persistence beyond the primary
   reference's support. The 2,048-token sensitivity check is preregistered; self-SFT and
   format-only controls and a second SFT seed would be additional experiments.
   ```

8. **After [line 59](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/RESULTS_30_stage2.md:59), disclose incomplete registered reporting.** [Readings 98–103](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_30.md:98) and [Amendment 4](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_30.md:191) retain readouts absent from both supplied results files. Add:

   ```text
   Reporting scope: neither this document nor the supplied analysis output reports the
   registered T = 1.15 and 1.3 endpoint evaluations, the 2,048-token sensitivity results,
   complete checkpoint finishing/boxing/truncation/length series, each start's own zero-count
   stratum, accuracy at equal RL-generated tokens, total teacher/SFT/RL compute, or the
   predictor comparisons and rank agreements. Their completion status is not established here.
   ```

No files modified.