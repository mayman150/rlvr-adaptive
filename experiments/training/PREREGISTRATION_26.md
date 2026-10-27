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
