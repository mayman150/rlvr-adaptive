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

## Data

Human-written MATH solutions for all 3,920 training problems (EleutherAI/hendrycks_math, matched by
problem text; all 3,920 verified by our checker with a parseable final box; mean 427 characters
against 1,110 for the base successes).
- `data/human_all.json`: solutions for all 3,920 problems, including the 487 the base model never
  solved in 32 samples.
- `data/human_solvable.json`: solutions only for the 3,235 problems with a retained base success (the
  same problems as S32's base-success data).

## Arms (seeds 8, 9, 10; S32's procedure: supervised stage at AdamW 1e-5, batches of 8, 2.46 million response tokens, gradient clipping 1, optimizer reset; then Pre-registration 12's Dr. GRPO for 150 steps)

- **HA:** supervised stage on `human_all.json`.
- **HS:** supervised stage on `human_solvable.json`.
References: S32 (Pre-registration 14; base successes on the same 3,235 problems); B (Pre-registration
12); the long Dr. GRPO trajectories (seed 13 on AWS; 14 and 15 on Tamia if available).

## Evaluation

The 511 problems, 32 samples, steps 0, 50, 100, 150 at T = 1; greedy accuracy on 3,382 test problems
at step 150. Pre-registration 12's interpolation, 2,000 shared bootstrap draws over problems.

## Decision rules

1. **New information moves the curve:** HA against HS's trajectory at matched pass@1 at least +1 pp
   with the 95% interval above 0 (the two differ only by solutions to the 487 problems the base never
   solved). pass@16 in the c = 0 stratum, HA minus HS at step 150, reported with its 90% interval.
2. **External versus self-generated text on the same problems (reported):** HS against S32's
   trajectory; the length difference (427 vs 1,110 characters) is a stated confound.
3. **Against the long Dr. GRPO curve (descriptive):** HA, HS and S32 against the long trajectory at
   matched pass@1 (seed 13, plus Tamia seeds if available); with one reference seed this is reported
   without a verdict.
Thesis prediction: rule 1 holds; HS and S32 lie near the long Dr. GRPO curve.

## Cost

About $90 training (six runs) and $60 evaluation.
