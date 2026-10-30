The four reproducibility rows, all checkpoint-table numbers and verdicts, B2 estimates, diagnostic examples, and aggregate tallies match the analysis. The corrections concern prose, interpretation, and reporting completeness. No files were modified.

All line numbers below refer to the original [RESULTS_33.md](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/RESULTS_33.md:1).

1. **Lines 3–4: provenance is unverifiable from the two comparison files.** Neither records the job’s 42 exit statuses, commit timing, or two subsequent speed-only changes. Unless supported separately, replace with:

   > Analysis output:

   This connects directly to the existing filename on line 5.

2. **Lines 7–8: state the family boundaries and registered guards explicitly.**

   > Holm correction is applied within each of the three primary families. Departure requires |gain| >= 2 and Holm-adjusted p <= 0.05; equivalence requires the 90% interval within +/-2, pointwise and unadjusted; otherwise the verdict is inconclusive. Comparisons with more than 10% of bootstrap draws outside support are not interpreted; truncation above 90% is a protocol failure.

   At **line 6**, add “shared across comparisons on that cohort” after “2,000 problem-bootstrap draws per cohort.”

3. **Lines 19–20: the maximum disagreement is 0.37, and the flag requires fuller sensitivity reporting.**

   > A cohort is flagged when either seed’s absolute step-25 point difference exceeds 3 percentage points in pass@1 or 1 point in pass@16; the interval endpoints do not determine the flag. Only hard-cohort seed 9 is flagged, because its pass@16 difference is +1.38 points. Accordingly, the dense, splice, and supported B7e-only early readings receive equal weight in the hard-cohort discussion.
   >
   > Dense and splice point gains differ by up to 0.37 points: LUFFY is +2.24 versus +2.50, random reward +1.64 versus +2.01, and ExGRPO-LUFFY continual +1.41 versus +1.43; the other three coincide at the printed precision. The early-reference point gains coincide with the dense gains for the five checkpoints it covers; it does not cover the continual checkpoint. Sensitivity intervals and Holm p-values are not printed, so their verdicts cannot be verified. In particular, the random-reward splice estimate reaches the 2-point effect-size threshold, but that alone does not establish a departure.

   The existing four flag labels are correct.

4. **Lines 35–38: “above the range” is not demonstrated by these sources, and “above 62 are unchanged” is too absolute.**

   > Six checkpoints (Oat-Zero, RAFT++, the ground-truth reproduction, both DeepMath models, ReLIFT) are reported as having insufficient overlap and are not interpreted. Tally: 6 equivalent, 1 inconclusive, 1 departure down, 6 insufficient overlap; no upward departure.
   >
   > For the five checkpoints at pass@1 46.9–57.3%, the reported paired dense-minus-sparse gain changes range from -0.59 to -0.10 points; all five 90% change intervals include zero. One-shot RLVR’s change is -0.51 [-1.22, +0.20]. Among the three higher-pass@1 scorable checkpoints, SimpleRL-Zero and SimpleRL-Zoo have changes +0.00 [+0.00, +0.00], while ExGRPO-LUFFY continual has +0.01 [-0.51, +0.49].

5. **Lines 51–53: the hard-cohort reductions range from 0.02 to 2.09 points.**

   > Eight checkpoints are reported as having insufficient overlap and are not interpreted. Tally: 6 inconclusive, 8 insufficient overlap; no departure in either direction and no equivalence verdict. The reported paired dense-minus-sparse changes, in table order, are -2.06, -2.09, -2.02, -1.23, -0.82 and -0.02 points: reductions of 0.02–2.09 points. Their 90% intervals exclude zero for the first four checkpoints and include zero for random reward and the continual checkpoint.

   Use the printed paired changes: subtracting independently rounded gain columns can differ by 0.01.

6. **Lines 57–60: clarify the training label and that RePO minus GRPO compares law-referenced gains.**

   > All 14 checkpoints are scorable against the frozen law (tau = -0.20). Three correctness-only checkpoints have verdict “departure down”: SimpleRL-Zero -13.66 [-17.48, -10.16] (Holm p <0.001; 41% truncation), Oat-Zero -5.81 [-8.86, -2.92] (Holm p 0.024), and the ground-truth reproduction -5.70 [-8.68, -2.90] (Holm p 0.013). The other eleven are inconclusive, with estimates from -4.26 to -0.83; none is positive. The paired difference in frozen-law gains, DeepMath RePO minus GRPO on the hard cohort, is +0.48 [-1.81, +2.92] (90%).

   The existing 41% is correctly rounded from 41.156%.

7. **Lines 64–66: retain the diagnostic numbers without attributing the difference solely to the sparse chord.**

   > On the 511 problems, replacing the sparse reference with the dense composite reference changed the matched-pass@1 pass@2 gain from +1.14 to -0.78 for LUFFY’s on-policy baseline and from +1.10 to -0.65 for ExGRPO Zero. These are descriptive sensitivity estimates; the composite reference changes both the early knots and the step-25 replicate.

8. **Lines 70–77: remove the implied observed failures, confirmed-bias claim, and unsupported precision conclusion.**

   > None of the 28 interpreted comparisons in the three primary families met the registered upward-departure criterion. Across those families, there are 6 equivalent, 18 inconclusive and 4 departure-down verdicts, plus 14 comparisons with insufficient overlap. No protocol-failure verdict is reported. The absence of upward departures does not establish general equivalence.
   >
   > Using the dense composite reference reduced the point gains for the five 511-problem checkpoints at pass@1 46.9–57.3% by 0.10–0.59 points, and for the six scorable hard-cohort checkpoints by 0.02–2.09 points. These estimates show sensitivity to reference construction, which changes both the early knots and the step-25 replicate.
   >
   > On the hard cohort, dense and splice gains are positive for five of six scorable checkpoints; all six dense verdicts are inconclusive. Frozen-law gains are negative for all fourteen checkpoints, with three downward departures. For the six checkpoints covered by both comparisons, dense-minus-law gains range approximately from +2.24 to +4.09 points; splice-minus-law gains range from +2.26 to +4.27, calculated from rounded estimates. Paired intervals for these between-reference contrasts are not reported. Scope: two reference seeds, one budget (2,048 tokens), one protocol.

9. **Line 61: insert the registered label tallies to make the summary self-contained.** Preserve the analysis’s separate baseline label:

   | Training-data label | A1 | A2 | B2 |
   |---|---|---|---|
   | correctness-only (LUFFY’s on-policy baseline) | 1 equivalent | 1 inconclusive | 1 inconclusive |
   | correctness-only | 2 equivalent; 1 inconclusive; 1 departure down; 5 insufficient overlap | 2 inconclusive; 7 insufficient overlap | 6 inconclusive; 3 departure down |
   | external | 1 equivalent; 1 insufficient overlap | 1 inconclusive; 1 insufficient overlap | 2 inconclusive |
   | random reward | 1 equivalent | 1 inconclusive | 1 inconclusive |
   | external lineage (start is not the base) | 1 equivalent | 1 inconclusive | 1 inconclusive |

   Also insert:

   > Additional paired gain-change intervals, support/drop rates and truncation rates for scorable comparisons are available in the analysis output. Boxed rates, requested by the preregistration, are absent from that output.

10. **Small wording corrections:**

   - **Lines 24 and 42:** replace `Verdict` with `Dense verdict`.
   - **Line 42:** replace `Checkpoint (label)` with `Checkpoint`; replace `Sparse gain (paired change)` with `Sparse gain (dense minus sparse [90%])`.
   - **Lines 30 and 48:** the exact `lr 5e-7` value is not established by the truncated analysis identifier or preregistration. Source-limited replacements for the checkpoint cells are `Random reward reproduction (random reward)` and `Random reward reproduction`, respectively; retain the learning rate only with separate checkpoint-identity evidence.