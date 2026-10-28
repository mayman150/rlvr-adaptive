# Investigation 27 (exploratory, not registered): why do methods share one curve?

October 28, 2026. Existing vLLM evaluations (1.5B, 1,024 tokens, 511 test problems, 32 samples at T = 1).

## Notation

- p_i: a checkpoint's success rate on problem i (successes of 32); pass@1 = mean_i p_i.
- Matched pair: two checkpoints whose pass@1 differ by at most 1.5 points.
- Per-problem RMS difference: sqrt(mean_i (p_i^a - p_i^b)^2), in percentage points. It includes sampling noise of the 32
  samples; every evaluation uses the same per-problem sampling seeds, so noise is not independent across checkpoints.
- c_i: the base model's successes of 32 on problem i.

## Findings

1. Weight space (Pre-registration 26, step 1a): PSR1k8 and NSR1k8 have the least similar cumulative updates of all
   measured pairs (cosine 0.266; Dr. GRPO seeds 0.34).
2. Behaviour, per problem: at matched pass@1, two different methods differ per problem as much as two seeds of the same
   method (mean per-problem RMS difference 6.69 pp for different methods and seeds, 6.63 for different methods with the
   same seed, 6.66 for the same method with different seeds, 7.01 for PSR vs NSR; 282, 141, 55 and 20 pairs).
3. The mean success rate by base count c_i is the same across methods at matched pass@1 (e.g. pass@1 near 58:
   c = 0 about 9%, c = 1-2 about 33%, c = 3-8 about 67%, c = 9-16 about 79%, c = 17-24 about 88%).
4. A one-parameter uniform logit shift of the base rates (smoothed (c_i + 0.5) / 33), fitted to each checkpoint's
   pass@1, overpredicts pass@16 by about 10 points (71 checkpoints, RMSE 10.0): the common path is not a uniform logit
   shift.

## Reading

At matched pass@1, methods are indistinguishable from Dr. GRPO seeds not only in pass@16 but in which problems they
solve and how often, although their weight updates differ (PSR and NSR most of all). The shared object is the
per-problem success profile, a behavioural quantity; many different weight changes produce it.

## Next test (job worth-p27-base-s0-20261028-055034)

Does sharpening the base model alone reach the same profile? The base model at T = 0.4, 0.6, 0.8 (32 samples), and at
T = 1 with 128 samples (also completes the BOPTR baseline). If base-at-lower-temperature lies on Dr. GRPO's curve and
matches its per-problem profile at equal pass@1, the curve is a property of the base model's own distribution, which
RL sharpens; if it falls below the curve, RL does more than temperature-like sharpening.
