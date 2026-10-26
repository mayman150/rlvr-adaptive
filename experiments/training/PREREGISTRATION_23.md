# Pre-registration 23 (pilot): minimum-help credit on the policy's own prefixes

Declared October 26, 2026, before any run of this design; method proposed by Codex (ranked first of five);
revised after Codex's code and design review, before launch. A one-seed pilot on validation problems: it ranks and
decides whether a registered three-seed test follows; it cannot establish an effect.

## Why

Rollout allocation (Pre-registration 18) doubled the share of rare problems with a sampled success and did not move
the matched-pass@1 curve: more binary outcomes on rare problems are not the missing signal. On a problem whose answers
all fail, Dr. GRPO's update is zero, and with one success it only separates that answer from the rest. Minimum-help
credit ranks the policy's own failed attempts by how much external help they need to be completed, and credits only
the policy-generated tokens, so the policy is not asked to reproduce human text (the failure of Pre-registration 19).
Closest prior work: Rewarding Progress (2410.08146; a prover's progress as process reward), VinePPO (2410.01679;
continuation-based values), KnowRL (2604.12627; minimal sufficient guidance). Difference: the measured assistance
requirement supervises the unassisted prefix that preceded the intervention.

## Method (`train_grpo.py --mh-mode min`)

Pre-registration 21's recipe (Dr. GRPO, 16 x 8 rollouts, 1,024 tokens, AdamW 1e-6, 150 steps, budget 128, seed 8)
plus, at steps t < 100: among the step's problems with at most 1 success in their 8 answers and a retained human
solution (`data/human_prefix_train.json`; the loader drops human solutions longer than 1,023 tokens or failing
re-verification, counts reported), up to 4 problems (random); 4 of each problem's failed answers (random); each
answer's prefix of L = min(256, tokens before its first stop token and wholly before its first box) tokens (skipped if
L < 8). Each prefix is continued with hints of 0, 32 and 64 tokens of the problem's human solution, cut wholly before
its box and before its first "= answer", inserted as "(Hint: ...)" after the prefix; 2 continuations per level, within
1,024 tokens in total (levels with fewer than 16 tokens left are skipped); no gradient through continuations. Value of
a prefix: 1, 2/3, 1/3 for the smallest hint level with a verified success, 0 if none; centred within its problem's
prefixes; problems whose prefixes all have the same value get no credit. Additional loss:
-c_t sum_i a_i sum_{s < L_i} log pi(y_is) / (128 x 1,024), c_t = 2 (1 - t/100). The ordinary Dr. GRPO loss is
unchanged.

**Control AVH (`--mh-mode avg`):** identical, except the value is the mean success over the prefix's 6 continuations
(a continuation-value process reward with the same hints, queries and schedule).

## Pilot (seed 8; Pre-registration 16's validation problems)

Arms MH and AVH; reference B1k seed 8 (Pre-registration 21). Evaluation on the 255 validation problems
(`data/val_frontier_256.json`, not the 511 test problems), 32 samples at T = 1, 1,024 tokens: MH and AVH at steps
50, 100, 150; B1k at steps 25, 50, 100, 150 and at T = 1.15 and 1.3 at step 150 (envelope; base model and B1k's steps
as interpolation knots).

**Estimator.** Pre-registration 12's interpolation: the mean, over the arm's checkpoints inside the reference's pass@1
range, of the arm's pass@16 minus the reference's interpolated pass@16 at the arm's pass@1; 2,000 bootstrap draws over
problems shared across arms; at least two overlapping checkpoints, else "insufficient overlap"; more than 10% of
bootstrap points clamped makes the comparison uninterpretable. Uncertainty is conditional on this seed.

**Dose floor.** At least 150 credited prefixes (nonzero centred value) over steps 0-99; otherwise the pilot is
"inconclusive: insufficient credit".

**Selection rule.** An arm X in {MH, AVH} qualifies if (1) its gain over B1k's envelope is at least +1.5, (2) its
step-150 pass@1 is at most 2 points below B1k's, and (3) at least 30% of its credited problem-steps show a contrast
between prefix values. MH is the candidate for the three-seed test if it qualifies and MH minus AVH's trajectory is
at least +0.5; AVH is the candidate if AVH qualifies and MH does not (or MH minus AVH is below +0.5 while AVH
qualifies). Otherwise the method is dropped and reported.

**Reported.** Eligible and credited problems per step, credited prefixes and tokens, credit mass and the variance of
centred values in each arm, assisted success by hint level, duplicate hint levels (human solutions shorter than the
larger hint), query tokens and truncation (the extra generation cost), unassisted pass@16 on problems with 0 and 1-4
base successes, and the repeatability of prefix values (prefixes re-probed with fresh continuations in an offline
audit of 50 credited prefixes).

## Cost

2 training runs (about 8 L40S-hours each with the extra continuations) and 12 checkpoint evaluations on 255 problems
(about 1.2 L40S-hours each): about 30 L40S-hours.
