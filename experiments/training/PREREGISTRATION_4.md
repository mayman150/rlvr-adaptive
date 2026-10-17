# Pre-registration 4: positive and negative reinforcement share their expected update

Declared October 17, 2026, before any run of this design existed (see this file's commit
time). It follows the literature review: the critical-budget framing is already studied
(Bansal et al., HiLD at ICML 2026; Li et al., arXiv 2608.29296), while no paper measures
gradient norms by correctness or separates the mean and the noise of positive and
negative sample reinforcement.

## Theory being tested

For binary rewards, on-policy sampling, and sequence-level weights, the two-moment
theorem gives, per prompt with n answers:

- PSR (advantage 1 for correct answers, 0 otherwise): mean n grad p, noise term n p F1.
- NSR (advantage -1 for wrong answers, 0 otherwise): mean n grad p, noise term n (1-p) F0.
- Dr. GRPO: mean (n-1) grad p. W-REINFORCE (0.1 for correct, -1 for wrong): mean 1.1 n grad p.

So PSR and NSR have exactly the same expected update. Whatever separates them in
training (Zhu et al. 2025: NSR preserves entropy and pass@k, PSR collapses them) must come
from the noise, which NSR puts on hard prompts and, because F0 is about 2.5 F1, makes
larger per answer, or from implementation choices outside the theorem (token-mean
aggregation, clipping, reuse of stale samples).

## Runs

Qwen2.5-Math-1.5B base, the pilot's data, strictly on-policy (one step per batch, no
ratios or clipping), sequence-sum loss with a constant normalizer, AdamW at 1e-6, 16
prompts by 8 answers, 150 steps, seeds 1 and 2. Arms: drgrpo, psr, nsr, wreinforce
(positive weight 0.1). Generation in batches of 256.

## Measurements

- Every step: split-half gradient statistics with two independent within-prompt
  references (per-prompt mean update projected on the population gradient, per-prompt
  second moment, per-prompt signal).
- Every 5 steps: token entropy and mean token log-probability of the first 4 answers of
  every prompt (an on-policy sample for every arm), split by correctness.
- Greedy accuracy on the 364 numeric MATH-500 problems at steps 0, 25, 50, 75, 100, 125,
  150; sampled accuracy (4 answers) at 50 and 100; pass@k for k up to 64 (64 samples per
  problem) at step 150, and at step 0 in the seed-1 drgrpo run; greedy accuracy on 3,382
  further MATH test problems at step 150. Final models are saved.

## Predictions

1. **Equal means.** The projection of the mean update on the population gradient (q) is
   equal for psr and nsr within sampling error (ratio between 0.7 and 1.4 pooled over
   steps 0 to 49 and both seeds), and drgrpo's is 7/8 of theirs.
2. **Noise as the theorem says.** The measured per-prompt second moment of each arm agrees
   with the theorem's prediction from F0, F1 (calibration) and the observed group counts
   within 30% over steps 0 to 49; nsr's exceeds psr's by the predicted factor.
3. **What differs.** Primary endpoint: pass@64 at step 150, nsr minus psr, both seeds.
   Secondary: probe entropy over steps 100 to 150, greedy accuracy, pass@1.
   - If nsr exceeds psr by at least 3 points of pass@64 in both seeds, or its entropy is
     higher by at least 20% in both seeds, the separation survives strictly on-policy
     training with equal means: it is a noise effect.
   - If neither, the reported separation is not reproduced in this setting, which points
     to implementation choices (to be tested next with token-mean aggregation and clipped
     reuse of samples).

Two seeds justify a larger study, not a claim.
