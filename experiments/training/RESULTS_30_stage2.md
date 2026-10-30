# Results of Pre-registration 30, Stage 2 (Amendment 4: 25K starts)

Job worth-p30-stage2-s0-20261029-182830 (us-west-2), all 31 steps exited 0. Analysis: analyze_p30.py (committed before
any Stage 2 evaluation was inspected and revised after Codex's pre-run review, still before inspection); output:
results_p26/p30_stage2_analysis.txt (2,000 draws). Notation: R25 / H25 = SFT starts from 25K presented teacher tokens
on the random (R) and hard (H) selections; the dose is Amendment 4's registered fallback (no dose met the primary
validation rule), chosen after Stage 1 results were inspected. Gaps are pass@16 minus the same-seed Dr. GRPO-from-base
trajectory (B1k) at the checkpoint's pass@1, in percentage points, over RL checkpoints 1-50 that lie inside B1k's
pass@1 support (steps 100 and 150 lie beyond it); intervals condition on the trained starts and seeds. P34 (the OPD
start) is reported separately once its seed 10 finishes.

## Starts (step 0; 64 samples; not part of the post-RL estimates)

| Start | pass@1 | pass@16 | pass@64 | Truncated | Gap vs B1k at matched pass@1 (seeds 8 / 9 / 10) |
|---|---|---|---|---|---|
| R25 | 47.5 | 86.6 | 92.0 | 45.5% | +0.85 / +0.73 / +0.95 |
| H25 | 37.8 | 86.2 | 92.8 | 47.3% | +1.90 / +1.69 / +1.72 |

## Registered verdicts (Holm over R25 and H25 within each family)

| Contrast | Estimate | 90% | 95% | Per seed | Draws affected | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| H25 observed gap vs B1k trajectory | +1.92 | [+0.89, +2.93] | [+0.68, +3.12] | +1.84, +1.84, +2.09 | 7.2%, 8.3%, 6.7% | 0.002 | inconclusive |
| R25 observed gap vs B1k trajectory | +1.03 | [-0.06, +2.14] | [-0.26, +2.37] | +0.94, +1.12, +1.02 | 38.2%, 50.4%, 72.4% | 1.000 | not interpreted (support) |
| H25 forecast residual | -4.77 | [-6.30, -3.22] | [-6.57, -2.88] | -4.86, -4.86, -4.60 | as above | <0.001 | forecast fails (bias detected) |
| R25 forecast residual | - | - | - | - | - | - | unscorable (a full-data eligible checkpoint has no forecast) |
| H25 observed gap vs B1k envelope | +1.92 | [+0.89, +2.93] | [+0.68, +3.13] | +1.84, +1.84, +2.09 | 0% | 0.002 | inconclusive |
| R25 observed gap vs B1k envelope | +1.03 | [-0.05, +2.12] | [-0.26, +2.37] | +0.94, +1.12, +1.02 | 12.2%, 20.9%, 41.7% | 1.000 | not interpreted (support) |

H25: a positive gap is detected (95% interval above 0, every seed positive, Holm p 0.002), but the estimate is below
the registered +2 threshold for "curve raised", so the verdict is inconclusive; the one-sided bootstrap probability
on the side of |gap| <= 2 is 0.56, so the magnitude is unresolved against the threshold. The frozen law forecast a
gap of +6.69 [+5.23, +8.13] for H25 and failed (residual -4.77). R25's step-50 checkpoints sit at the edge of B1k's
pass@1 range, so too many bootstrap draws lose them; its contrasts are not interpreted, and the targeting comparison
(H25 minus R25) is therefore not interpreted.

## Per-checkpoint gaps (pointwise 90% bands; descriptive)

H25, steps 1, 2, 5, 10, 25, 50: seed 8 +2.0, +2.1, +1.9, +1.5, +1.6, +2.0; seed 9 +1.4, +2.1, +1.2, +2.2, +2.1, +2.0;
seed 10 +2.0, +2.7, +1.8, +1.9, +2.0, +2.1 (bands roughly +/-1.4). R25: +0.5 to +1.5 at every checkpoint in all seeds.
The post-RL gaps are close to each start's own offset; the data support early preservation of the offset (through
step 50), not persistence through step 150. Beyond B1k's support, seed 8 against P27's single-seed LH1 run
(descriptive): H25 +1.5 (step 100), +1.1 (step 150); R25 +1.0, +0.6.

## Step 150 (T = 1; descriptive, unmatched in pass@1)

pass@1 64.4-65.4 for the SFT starts against 57.5-58.1 for B1k; pass@16 87.7-88.9 against 86.4-87.2; truncation
19.6-22.4%; boxed 87.2-87.9%. pass@16 on the 51 test problems with zero base successes in 64 samples: 27.9-30.9 for
the SFT starts against 15.2-23.9 for B1k, not matched in pass@1.

## Reading

In this regime (1.5B, MATH, 1,024 tokens, Dr. GRPO, three RL seeds from one start per selection), RL from a 25K-token
teacher-SFT start on hard problems keeps a positive pass@16 gap over Dr. GRPO-from-base at matched pass@1 through
step 50, approximately equal to the start's own pre-RL offset; its size (+1.92) is inconclusive against the registered
2-point threshold. The frozen per-problem law, fitted on RL from the base, did not forecast this shift. Open
questions agreed in the three-way review (paper/debate_2026_10_30/): whether the offset reflects finishing or format
rather than reasoning content (the registered 2,048-token check; self-SFT and format-only controls), whether it
survives a second SFT seed, and whether it persists beyond step 50 (a longer reference).
