# Results of Pre-registration 30, Stage 2 (Amendment 4: 25K starts)

Job worth-p30-stage2-s0-20261029-182830 (us-west-2); all 31 steps exited 0 (the job's status.json). Analysis:
analyze_p30.py; output: results_p26/p30_stage2_analysis.txt (2,000 joint problem-bootstrap draws, as registered).
R25 / H25 denote starts at the nominal 25K presented-token SFT dose, using SFT seed 3030 and 262 problems per
selection from Amendment 3's teacher pool. R is a subject-stratified random selection among problems with surviving
verifier-correct teacher solutions; H comprises such problems with base success count c <= 4 of 64. Amendment 4 chose
25K through its validation-only fallback, minimizing the summed distances of R's and H's validation pass@1 from
0.40; this is recorded as a deviation from the primary dose rule.

Observed gaps, in percentage points, are pass@16 minus the same-seed T = 1 B1k trajectory interpolated at the
checkpoint's observed pass@1, averaged over eligible checkpoints within seed and then equally over seeds 8, 9 and
10. Full-data eligible checkpoints are 1, 2, 5, 10, 25 and 50; steps 100 and 150 lie beyond B1k's support. Intervals
condition on the trained starts and seeds. P34 is unavailable in this output because all seed-10 checkpoints were
missing when it ran. Wording checked by Codex (codex_results30_stage2_check.md).

## Starts (step 0; 64 samples; not part of the post-RL estimates)

| Start | pass@1 (%) | pass@16 (%) | pass@64 (%) | Truncated | Gap vs B1k at matched pass@1 (pp; seeds 8 / 9 / 10) |
|---|---|---|---|---|---|
| R25 | 47.5 | 86.6 | 92.0 | 45.5% | +0.85 / +0.73 / +0.95 |
| H25 | 37.8 | 86.2 | 92.8 | 47.3% | +1.90 / +1.69 / +1.72 |

## Registered verdicts (Holm over R25 and H25 within each family)

| Contrast | Estimate (pp) | 90% CI | 95% CI | Per-seed estimates (pp; 8 / 9 / 10) | Draws affected (8 / 9 / 10) | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| H25 observed gap vs B1k trajectory | +1.92 | [+0.89, +2.93] | [+0.68, +3.12] | +1.84, +1.84, +2.09 | 7.2%, 8.3%, 6.7% | 0.002 | inconclusive |
| R25 observed gap vs B1k trajectory | +1.03 | [-0.06, +2.14] | [-0.26, +2.37] | +0.94, +1.12, +1.02 | 38.2%, 50.4%, 72.4% | 1.000 | not interpreted (support) |
| H25 forecast residual | -4.77 | [-6.30, -3.22] | [-6.57, -2.88] | -4.86, -4.86, -4.60 | 7.2%, 8.3%, 6.7% | <0.001 | forecast fails (bias detected) |
| R25 forecast residual | - | - | - | - | - | - | unscorable |
| H25 observed gap vs B1k envelope | +1.92 | [+0.89, +2.93] | [+0.68, +3.13] | +1.84, +1.84, +2.09 | 0.0%, 0.0%, 0.0% | 0.002 | inconclusive |
| R25 observed gap vs B1k envelope | +1.03 | [-0.05, +2.12] | [-0.26, +2.37] | +0.94, +1.12, +1.02 | 12.2%, 20.9%, 41.7% | 1.000 | not interpreted (support) |

H25's mean observed gap is positive: its 95% interval is above 0, all three seed means are positive, and Holm
p = 0.002. "Curve raised" additionally requires an estimated mean gap >= +2 points; the estimate +1.92 therefore
receives the registered verdict "inconclusive". The one-sided bootstrap p-value for |gap| > 2 is 0.560 for the
trajectory contrast and 0.559 for the envelope contrast.

The law forecast a mean H25 gap of +6.69 (reported interval [+5.23, +8.13], 90%), overestimating the observed gap.
The residual -4.77 meets "forecast fails (bias detected)": |residual| >= 2 and the two-sided test of zero residual has
Holm p <= 0.05 (here < 0.001); the separately reported one-sided p-value for |residual| > 2 is 0.003.

The registered analysis requires at least two eligible checkpoints per seed. Checkpoints leaving reference or law
support are dropped within bootstrap draws; more than 10% affected draws makes that seed's verdict "not interpreted".
R25's trajectory and envelope contrasts exceed this limit in every seed. Its forecast residual is separately
"unscorable": the full-data eligible seed-10, step-2 checkpoint has no forecast. The H25-minus-R25 targeting
comparison is consequently not interpreted (the registered rule requires an estimated difference >= 2 points with a
95% interval above 0).

## Per-checkpoint gaps (descriptive point estimates; full pointwise 90% intervals in the analysis output)

H25, steps 1, 2, 5, 10, 25, 50: seed 8 +2.0, +2.1, +1.9, +1.5, +1.6, +2.0; seed 9 +1.4, +2.1, +1.2, +2.2, +2.1, +2.0;
seed 10 +2.0, +2.7, +1.8, +1.9, +2.0, +2.1. R25 point estimates range from +0.5 to +1.5 across these checkpoints and
seeds. H25's seed-9, step-5 pointwise 90% interval is [-0.2, +2.6]; the aggregate positive-gap result does not
establish a positive gap at every individual checkpoint.

Descriptively, the post-RL point estimates are close to each start's step-0 offset. Preservation or equivalence of
these offsets has not been tested. The primary matched analysis covers early RL through step 50 and does not
establish persistence through step 150. Beyond B1k's support, the descriptive seed-8 comparison with P27's LH1
reference gives H25 +1.5 at step 100 and +1.1 at step 150; R25 +1.0 and +0.6.

## Step 150 (T = 1; descriptive, unmatched in pass@1)

Across R25 and H25 and seeds 8-10, step-150 pass@1 is 64.4-65.4%, versus 57.5-58.1% for B1k; pass@16 is
87.7-88.9%, versus 86.4-87.2%. For the SFT starts, truncation is 19.6-22.4% and the boxed-answer rate is 87.2-87.9%.
On the 51 test problems with zero successes in the Stage 1 base's 64-sample evaluation, pass@16 is 27.9-30.9% for the
SFT starts and 15.2-23.9% for B1k. These endpoint comparisons are descriptive and unmatched in pass@1.

## Reading

In this regime (1.5B, MATH, 1,024 tokens, Dr. GRPO, three RL seeds from one SFT start per selection), H25 has a
positive mean pass@16 gap over same-seed Dr. GRPO-from-base at matched pass@1 across eligible checkpoints 1-50: +1.92
points, 95% interval [+0.68, +3.12], Holm p = 0.002. Its registered "curve raised" verdict is inconclusive because the
estimate is below +2 points. The checkpoint point estimates are descriptively close to the start's pre-RL offset;
preservation of that offset has not been established by a paired comparison.

The frozen per-problem law predicted a positive mean shift but overestimated it (+6.69 forecast versus +1.92
observed), yielding the registered verdict "forecast fails (bias detected)". Remaining questions concern the roles
of truncation, formatting and reasoning content, reproducibility across SFT seeds, and persistence beyond the
primary reference's support. The 2,048-token sensitivity check is preregistered; self-SFT and format-only controls
and a second SFT seed are additional experiments (Pre-registration 35).

Reporting scope: this document does not yet report the registered T = 1.15 and 1.3 endpoint evaluations, the
2,048-token sensitivity results, complete checkpoint finishing/boxing/truncation/length series, each start's own
zero-count stratum, accuracy at equal RL-generated tokens, total teacher/SFT/RL compute, or the predictor
comparisons and rank agreements; these are to be added.
