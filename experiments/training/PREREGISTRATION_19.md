# Pre-registration 19: does external information move the frontier where self-generated information does not?

Declared October 23, 2026, before any run of this design. Reviewed by Codex before launch.

## What was known

At matched pass@1, every intervention tested that uses only the model's own information (advantage
reweighting, entropy terms, replay of own or base-model successes, groups of 32) lies on or near Dr.
GRPO's trajectory; a preliminary 450-step Dr. GRPO run (seed 13; steps 200 and 250 evaluated) reaches
pass@1 51.7 / pass@16 78.6, the point the base-success warm start (S32) and groups of 32 reach at step
150 (S32 about +1.9 above that trajectory, G32 -0.35, both provisional). Thesis: interventions that
reuse the policy's own information change the progress rate, not the curve; only information the base
model does not have should move it.

## Data (revised after Codex's review, before launch)

Human-written MATH solutions for all 3,920 training problems (EleutherAI/hendrycks_math, matched by
problem text; all 3,920 verified by our checker with a parseable final box).
- `data/human_positive.json` (HP): solutions for the 3,433 problems with at least one base success in
  32 samples (mean 380 characters).
- `data/human_swap.json` (HQ): the same number of problems, 3,433, in which 487 randomly chosen
  problems with c > 0 (seed 1919) are replaced by the 487 problems with no base success in 32 samples
  (mean 433 characters; the c = 0 solutions average 759, the replaced ones 387).
- `data/human_solvable.json` (HS): solutions for the 3,235 problems with a retained base success (the
  same problems as S32).
"c = 0" means no success observed in 32 samples, not that the base model cannot solve the problem.

## Arms (seeds 8, 9, 10; S32's procedure: supervised stage at AdamW 1e-5, batches of 8, 2.46 million response tokens, gradient clipping 1, optimizer reset; then Pre-registration 12's Dr. GRPO for 150 steps)

HQ, HP, HS as above. References: S32 (Pre-registration 14); B (Pre-registration 12); the long Dr.
GRPO trajectories (seed 13 on AWS; 14 and 15 on Tamia if available). The retained solutions after
the trainer's checks (16 calibration problems withheld per seed; length and re-verification filter),
their counts by c stratum, response tokens and epochs are logged and reported.

## Evaluation

The 511 problems, 32 samples, steps 0, 50, 100, 150 at T = 1; greedy accuracy on 3,382 test problems
at step 150. Pre-registration 12's interpolation, 2,000 shared bootstrap draws over problems.

## Decision rules

1. **Information on problems the base did not solve moves the curve:** HQ against HP's trajectory
   (including HP's post-fine-tuning step 0) at matched pass@1, at least +1 pp with the 95% interval above
   0. HQ and HP have the same number of problems and token budget and differ only in whether 487 slots
   hold c = 0 or c > 0 problems. pass@16 in the c = 0 stratum, HQ minus HP at step 150, with its 90%
   interval. Not interpretable (reported as such) if fewer than two HQ checkpoints per seed fall in HP's
   pass@1 range, if more than 10% of bootstrap points are clamped, or if fewer than 400 of the 487 c = 0
   solutions survive the trainer's filters. Remaining confounds, stated: the swapped solutions are longer
   (fewer epochs at the fixed token budget: about 12%).
2. **External versus self-generated text on the same problems (reported):** HS against S32's
   trajectory; length differs (human 374 vs base 1,110 characters).
3. **Against the long Dr. GRPO curve (descriptive):** HQ, HP, HS and S32 against the long trajectory at
   matched pass@1 (one or three reference seeds), without a verdict.
Thesis prediction: rule 1 holds; HP, HS and S32 lie near the long Dr. GRPO curve.

## Cost

About $135 training (nine runs) and $90 evaluation.
