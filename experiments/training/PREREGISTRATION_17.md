# Pre-registration 17: does a per-question learning law predict the accuracy-diversity frontier?

Declared October 22, 2026. Code frozen in `law.py` and `analyze_law.py` at this file's commit,
before the law is applied to any test set below. Designed after an exploratory, in-sample fit on
Pre-registration 12's Dr. GRPO runs (disclosed: RMSE about 1 point of pass@16 over 9 checkpoints,
with the transfer term not identified: -0.25 or +0.25 depending on the prior for never-solved
questions) and reviewed by Codex.

## The law

theta_x(0) = log((c_x + a)/(n_x - c_x + b)) from the base model's per-question counts, (a, b) a
Beta prior fitted by moments; d theta_x/dt = p_x(1 - p_x) + tau * mean p(1 - p). Each checkpoint
is placed at the time where the predicted mean success equals its observed pass@1; everything else
(pass@k for k = 2...32, difficulty strata, per-question counts) is predicted.

## Fitting (only on Pre-registration 12's B, Qwen2.5-Math-1.5B, seeds 8-10, steps 50/100/150)

- **tau_32:** least squares on pass@16 with the 32-sample base counts only (the general-purpose law).
- **tau_deep:** the same, after adding 256 base samples on the 309 evaluation problems the base
  solves at most 4 of 32 times (deep sampling); this identifies the sign of transfer.
- **beta:** the affine log-odds alternative (theta' = alpha + beta theta), fitted on the same data.
Then frozen. tau_deep's 90% interval: bootstrap over problems (refitting tau in each draw).

## Test sets (no refitting; only each checkpoint's pass@1 is matched)

- **T1, other methods on Qwen:** Pre-registration 12's CM, M, U; Pre-registration 14's D32, U32,
  C05, G32, T32; Pre-registration 15's BBG, MXR, WRF; Pre-registration 16's arms (validation set,
  its own base counts) when available.
- **T2, the warm start:** S32 at steps 50, 100, 150, from S32's own step-0 counts (prior refitted
  on them by the same procedure), tau_32 frozen.
- **T3, longer training:** new Dr. GRPO runs to 450 steps (seed 13 on AWS; 14 and 15 on Tamia),
  checkpoints at 200 to 450 beyond the fitted range, from the same base counts.
- **T4, a second family:** OLMo-2-1B (Pre-registration 9, five rules, two seeds, 256 problems, its
  own base counts), tau_32 transferred unchanged from Qwen; the OLMo-refitted tau is reported.
- **T5, a larger budget:** 1,024-token evaluations (Check 16) from the 1,024-token base counts.

## Criteria (fixed now)

Per test set: **supported** if the law's pass@16 RMSE is at most 2 points, **partial** if 2-3,
**falsified** if above 3. The law must also beat the affine alternative in total per-question
log-likelihood on T1-T4 (reported as the difference, with the number of runs where it wins).
Reported without a rule: pass@k for every k; never/rare/medium strata errors; per-question residual
correlation across seeds (whether residuals are reproducible, as "RLVR is a Kernel" (2609.22547)
argues they are); tau_deep's sign and interval.

## What follows from each outcome

Supported on T1-T4: the frontier is predicted from base difficulty and near-independent
per-question learning; correctness-only rules differ only in speed (the place on the trajectory);
moving the frontier requires changing the starting distribution or transfer. That law would be the
backbone of the paper, and the method would allocate demonstration acquisition where the law
predicts stalls (a separate registration). Falsified on T2 or T4: the law is scoped to the regime
where it holds.
