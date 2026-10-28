# Pre-registration 26 (diagnostics and pilots): updates off the shared direction

Declared October 27, 2026, before any run of this design; reviewed by Codex before the pilots launch.

## Why

Every tested intervention changes only the speed along Dr. GRPO's pass@16-vs-pass@1 curve (1.5B at 512 and 1,024 tokens,
7B at 2,048). The starvation explanation is refuted (allocation, hints, minimum-help credit). Independent seeds improve the
same problems (two-seed mixtures give no pass@16 gain, -0.3 to +0.1), and RLVR weight updates are reported to be close to
rank-1 and nearly linear (AlphaRL 2510.00553; RELEX 2605.21468). Hypothesis: training moves along one shared direction, so
methods that reweight or resample can only change speed; a method must change the direction of the update.

## Step 1: diagnostics (no training; `direction_diag.py`)

(a) For the 1,024-token runs B1k8, PSR1k8, NSR1k8, AC1k8, HG1k8, HGm1k8, PG1k8, MH8, AVH8 (step 150): the rank-1 energy
share of each run's weight update, the cosine between runs' updates and the alignment of their top singular directions;
along B1k8's trajectory (steps 25-150), the cosine between updates (linearity). Two checkpoints from B1k8's final update,
W0 + rank-1 part ("par") and W0 + remainder ("perp"), evaluated on the 511 test problems (vLLM, 32 samples, 1,024 tokens),
pass@1/pass@16 overall and by base-count stratum.
(b) At B1k8 steps 50 and 150: CountSketch (2^20) per-problem Dr. GRPO gradients for 32 mixed problems; the Gram matrix's
eigenvalue shares and mean pairwise cosine.
**Readings fixed in advance:** the shared-direction hypothesis is supported by (a) if the runs' updates have pairwise
cosine >= 0.5 and "par" recovers at least 80% of B1k8's pass@1 gain over the base; off-direction updates have room only
if "perp" retains a pass@16 gain at matched pass@1 or a rare-stratum gain (reported). Whitening has room if the top Gram
eigenvalue share at step 150 is at least 0.3 (one direction dominates) - otherwise problem gradients are already
decorrelated and whitening is predicted to change little.

## Step 2: pilots (seed 8; Pre-registration 21's recipe at 1,024 tokens; 255 validation problems)

- **OD (off-direction updates, `--od-beta 0.25 --od-warmup 10`):** for every 2-D non-embedding weight matrix, the top
  singular pair (u, v) of the cumulative update W_t - W_0 is tracked by two warm-started power iterations per step;
  after each AdamW step the realized update Delta becomes Delta - 0.75 u (u^T Delta v) v^T, rescaled to the norm of Delta
  (the update keeps its size; 75% of its component along the dominant direction is moved into the other directions).
- **PW (problem-space whitening, `--pw-lambda 1.0`):** per step, the Dr. GRPO gradient of each problem with a nonzero
  advantage (one backward pass per problem), CountSketch to 2^20 dimensions, Gram matrix G; problem weights
  alpha = (G + tr(G)/m I)^{-1} 1, negatives set to 0, rescaled to mean 1; the update uses advantages times alpha.
Reference: B1k seed 8 on the validation problems (Pre-registration 23's vLLM evaluation, envelope with T = 1.15, 1.3).
Evaluation: 32 samples, T = 1, 1,024 tokens, steps 50, 100, 150 (vLLM).

**Selection rule (Pre-registration 23's estimator and guards).** An arm qualifies for a three-seed test if its gain over
B1k's envelope at matched pass@1 is at least +1.5, its step-150 pass@1 is at most 2 points below B1k's, and its mechanism
is active: OD's logged pre-projection dominant share of the update is at least 0.05 on average after warm-up (there was
a dominant component to move), PW's alpha has max/min above 1.5 in at least half of the steps (the weights differ from
uniform). If none qualifies, both methods are dropped and reported, and the diagnostics are reported as the mechanism.

## Cost

Diagnostics: one L40S job (about 1 hour) and two vLLM evaluations. Pilots: two training runs (OD about 1.1x and PW about
2x Dr. GRPO's step time) and 6 vLLM evaluations: about 25 L40S-hours, $60-100.

## Amendment 1 (October 27, before any pilot and before any diagnostic output): Codex's code review, the teacher arm

**Corrections.** (i) Off-direction updates and whitening now reject every other intervention (entropy terms, floors,
replay, TAR, SFT, shaping, gating, shadow statistics) and each other, validate their settings, and are set up after any
SFT phase. (ii) Degenerate cases: a zero cumulative update skips the projection for that matrix that step; a lost
power-iteration vector is restarted and the matrix is normally projected in the same step; a zero or non-finite Gram
trace gives uniform weights; undefined cosines (zero vectors) are logged as 1 (inactive). (iii) The description of OD
in Step 2 overstated it: OD removes 75% of the update's component along the tracked direction and rescales the update to
its original norm, so it amplifies the update's existing other directions; an update parallel to the tracked u v^T
is returned unchanged. (iv) The gradient diagnostic left the model in evaluation mode after generation (no checkpointing) and ran
all answers in one pass; fixed, with microbatches and released references.

**Diagnostics (changes to Step 1).**
- Two rank notions are reported separately: the per-matrix rank-1 energy share (matrix rank, AlphaRL's measure) and the
  eigenvalue shares of the Gram matrix of B1k8's updates at steps 25, 50, 100, 150 (trajectory rank, RELEX's measure; a
  one-dimensional trajectory can have full-rank matrices).
- Seed-to-seed reference: B1k9 and B1k10 (step 150) are added, so each method-vs-B1k8 cosine is read against the cosine
  between Dr. GRPO seeds.
- A third checkpoint, "other" (base matrices, trained embeddings, norms and biases), controls for the non-matrix
  parameters that "par" and "perp" keep. The par reading becomes: par recovers at least 80% of B1k8's pass@1 gain over
  "other" (gain over the base also reported); the reading is not interpretable unless B1k8's gain over "other" is
  positive with a 90% interval above 0. The cosine and norm ratio between the realized (bf16-rounded) and the intended
  matrix updates, aggregated over matrices, are reported (vLLM evaluates in bf16). "perp" is compared with B1k8's
  1,024-token test trajectory and temperature envelope (Pre-registration 21's evaluation) at matched pass@1, with
  Pre-registration 21's estimator; its pass@16 departure and rare-stratum gain are reported, not gated.
- Gradients: two independent groups of 8 answers per problem (problems mixed in both), sketched over the 2-D non-embedding
  matrices only (the whitening pilot's sketch). Sampling noise inflates the diagonal of a one-group Gram matrix; the
  cross-group matrix S_A S_B^T (symmetrized) estimates the Gram matrix of the expected gradients given that both groups
  are mixed (conditional on mixing; not an unconditional RL direction). It can be indefinite, so eigenvalues are
  reported as signed shares of the sum of absolute eigenvalues, with the negative share. The whitening reading uses the
  cross-group top share at step 150 (>= 0.3: one direction dominates), and is not interpretable if the negative share
  exceeds 0.2 (noise-dominated). The norm-free (cosine) spectra and each problem's cross-group cosine are reported.

**Mechanism gates (changes to the selection rule).** Both previous gates can pass without a change of direction, so each
arm must also rotate its update: OD's logged realized cosine between the projected and the original update averages at
most 0.98 after warm-up; PW's cosine between the whitened and the plain combined gradient (sketch space) averages at
most 0.98. Steps without a projection count as cosine 1 and share 0. Reported for every arm: the cosine between its
step-150 update and B1k8's, against the seed-to-seed cosines. If no arm qualifies, the methods are reported as failed
pilots; that outcome does not by itself establish the shared-direction mechanism.

## Step 2b: residual teacher distillation (third pilot arm; approved October 27)

- **Teacher solutions.** Teacher: Qwen2.5-Math-7B after Dr. GRPO (B7, seed 8, step 150; Pre-registration 20). Problems:
  the 1,668 training problems with base count at most 4 of 32 (`discovery_qwen_train32.json`). 8 samples each at T = 1
  and 1,024 tokens (vLLM, seed 2030). Retained per problem: the first verified sample by sample index, canonicalized as
  in Pre-registration 14 (retokenized for the 1.5B, at most 1,023 tokens, verifies, has a final box). The teacher was
  trained on these problems; evaluation problems are disjoint from training.
- **TD (teacher distillation):** Dr. GRPO plus, at each step, the gradient of -lambda sum log pi(y_T | x) / N over the
  step's problems that have a teacher solution, with lambda = 1 (a teacher solution counts as one answer with advantage
  1) and N Dr. GRPO's constant normalizer. Each problem is seen at most once in 150 steps.
- **RTD (residual teacher distillation):** TD with the teacher gradient of each 2-D non-embedding matrix replaced by
  (I - u u^T) G (I - v v^T), rescaled to the norm of G, where (u, v) is the top singular pair of the cumulative update
  W_t - W_0 (the direction OD tracks); computed before the reinforcement gradient is added; no projection during the
  first 10 steps. Embeddings, norms and biases receive the plain teacher gradient. The projection acts on gradients;
  Adam's diagonal scaling means the resulting update is not exactly orthogonal to (u, v).
- **Selection.** TD qualifies by Step 2's rule (gain >= +1.5 over B1k's envelope at matched pass@1, step-150 pass@1 at
  most 2 below B1k's). RTD qualifies only if, in addition, it beats TD at matched pass@1 by at least +1 (TD's
  trajectory as the reference) and its logged removed share of the teacher gradient averages at least 0.05 after
  warm-up; RTD must itself also meet both of Step 2's conditions against B1k, with the estimator's guards. Otherwise the
  projection is dropped and only TD can advance.
- **Scope of the claim.** RTD residualizes the teacher gradient against the run's own tracked dominant update direction
  (which includes the teacher's own contribution from step 0), removes whole row and column subspaces of each matrix,
  leaves the other parameters unprojected, and acts before Adam's momentum and scaling. A positive RTD result supports
  "residualized teacher supervision helps", not that the update leaves a shared RL direction.
- Reported: teacher success rate and coverage of the rare problems, solutions per step, pass@16 by base-count stratum.

## Cost (revised)

Diagnostics: one L40S job (about 3 hours, two rollout groups) and three vLLM evaluations. Teacher solutions: one
8 x A100 job, about 1 hour ($32). Pilots: four training runs (OD, PW, TD, RTD) and 12 vLLM evaluations: about 45
L40S-hours, $120-170 in total.
