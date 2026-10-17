# Pre-registration 3: when does rollout collection matter?

Declared October 16, 2026, before any run of this design existed (see this file's commit
time). It follows the seed-1 pilot, where scaling failure advantages gave no accuracy
gain (45.3% versus 47.5%), and replaces the two queued optimal-rule jobs of
Pre-registration 2, which never started and are run here under the same arm definitions.

## Why

The theory predicts training signal per generated answer under two simplifications: prompt
gradients are orthogonal, and the step size is tuned. In real training three things can
hide a better per-answer signal: (1) training is not limited by gradient noise at the
budget used, so a cleaner gradient buys little; (2) what generalizes is the part of each
prompt's gradient shared with the population gradient, which orthogonality ignores; (3)
Adam at a fixed learning rate turns a k-fold noise reduction into roughly a square-root
gain in progress per step. This design measures each of these directly.

## Runs (per seed; seeds 1 and 2)

Common: Qwen2.5-Math-1.5B base, the pilot's data, optimizer, clipping, 150 steps, groups of
8 or blocks of 4 with a cap of 16, loss normalizer budget by 512 tokens.

| Run | Rule | Answers per step |
| --- | --- | --- |
| L32, L64, L128, L256 | Dr. GRPO | 32, 64, 128, 256 (4, 8, 16, 32 prompts) |
| O64, O128 | optimal lookup table (Pre-registration 2 definition) | 64, 128 on average |
| S128 | failures scaled by w = F1 / F0 (pilot definition) | 128 |

L128 and S128 repeat the pilot arms with the new measurements; their difference from the
pilot runs measures run-to-run variation from GPU nondeterminism alone.

## Measurements

- Greedy accuracy on the 364 numeric MATH-500 problems at steps 0, 10, 20, 30, 40, 50, 75,
  100, 125, 150, saved per problem.
- Sampled accuracy (4 answers per problem, temperature 1), the quantity the theory
  optimizes, at steps 50, 100, 150.
- Final greedy accuracy on 3,382 numeric MATH test problems outside MATH-500, per problem.
- Every step: prompts are split at random into halves A and B. With g_A and g_B the arm's
  gradients from each half and ref_B the Dr. GRPO estimator on the first 4 answers of each
  prompt in B (unbiased for the population gradient direction, independent of g_A), we
  log |g_A|^2, |g_B|^2, <g_A, g_B>, <g_A, ref_B>, |ref_B|^2. From these: the squared norm of
  the arm's mean update per prompt s, its per-prompt noise V, its noise scale in answers
  B_noise = n V / s (n = answers per prompt), and its per-answer efficiency along the
  population gradient, Eff = <mu, rho>^2 / (n V). Token entropy of trained answers.

Aggregation: ratio of sums over steps 0 to 49 and 50 to 149, both seeds pooled, with a
bootstrap over steps for intervals.

## Predictions and decisions

1. **Regime.** Report B_noise of L128. If B_noise > 128 answers, training at 128 is
   noise-limited and the ladder should show L256 ahead of L128 in mean greedy accuracy over
   steps 10 to 150 (AUC); if B_noise < 64, the ladder should be flat above 64. A ladder that
   contradicts its own measured noise scale falsifies the noise-scale reading.
2. **Mechanism.** The theory predicts Eff(O128) / Eff(L128) between 1.69 and 2.27 and
   Eff(S128) / Eff(L128) between 1.12 and 1.56. Pass if the optimal ratio is at least 1.3
   and the scaled ratio at least 1.1, pooled over steps 0 to 149 and both seeds.
3. **Payoff.** Primary: final accuracy on the 3,382-problem set, O128 minus L128, paired by
   problem, both seeds. Secondary: O64 against L128 (the compute-saving reading: the
   optimal rule with half the answers matches the baseline), AUC, sampled accuracy.
   **Promising** if O128 beats L128 by at least 1.5 points in both seeds, or O64 is within
   0.5 points of L128 or better in both seeds while L64 is at least 1.5 points below L128.
   **No evidence** otherwise.
4. **Exploration.** If S128's token entropy falls faster than L128's, the pilot's null
   result is consistent with failure down-weighting reducing exploration, which the
   one-step theory does not model.

Two seeds justify a larger study, not a claim.
