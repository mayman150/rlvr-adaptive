# Investigation 27 (exploratory, not registered): why do methods share one curve?

October 28, 2026. Existing vLLM evaluations (1.5B, 1,024 tokens, 511 test problems, 32 samples at T = 1).

## Notation

- \hat p_i: a checkpoint's estimated success rate on problem i (successes of 32); pass@1 = mean_i \hat p_i.
- Matched pair: two checkpoints whose pass@1 differ by at most 1.5 points.
- Per-problem RMS difference: sqrt(mean_i (p_i^a - p_i^b)^2), in percentage points. It includes sampling noise of the 32
  samples; every evaluation uses the same per-problem sampling seeds (distinct from training seeds), so noise is not
  independent across checkpoints and the RMS values carry no noise floor.
- c_i: the base model's successes of 32 on problem i.

## Findings

1. Weight space (Pre-registration 26, step 1a): PSR1k8 and NSR1k8 have the least similar cumulative updates of all
   measured pairs (cosine 0.266; Dr. GRPO seeds 0.34).
2. Behaviour, per problem (descriptive): at matched pass@1, pairs of different methods show similar observed profile
   distances to pairs of training seeds of the same method (mean per-problem RMS difference 6.69 pp for different
   methods and training seeds, 6.63 for different methods with the same training seed, 6.66 for the same method with
   different training seeds (55 pairs pooled over methods; Dr. GRPO alone: 11 pairs, 6.56), 7.01 for PSR vs NSR).
   Paired excess statistic (Codex's suggestion): for each method checkpoint M_s, the Dr. GRPO checkpoint of the same
   training seed B_s and of each other seed B_s' nearest in pass@1 (within 1.5 points); Delta = mean_i [(\hat p^M -
   \hat p^{B_s})^2 - (\hat p^{B_s'} - \hat p^{B_s})^2]; problem bootstrap (2,000 draws, all checkpoints resampled
   together), 90% intervals, as a percentage of Dr. GRPO's seed-to-seed squared distance (42.0 pp^2):
   all methods +2.1% [-1.9%, +6.1%] (78 combinations); PSR1k +7.5% [+2.6%, +12.4%]; NSR1k +7.0% [-0.5%, +14.2%];
   AC1k +4.6% [-0.5%, +9.7%]; HG1k -0.2% [-5.6%, +5.4%]; HGm1k -0.1% [-6.3%, +5.9%]; PG1k -7.1% [-12.2%, -2.1%].
   M_s and B_s share the training data order, which by itself should make them closer than B_s' and B_s; the
   advantage-rule arms (PSR, NSR) nevertheless leave a small method-specific per-problem signature, while arms that
   keep Dr. GRPO's advantages (HG, HGm, PG, AC) stay within training-seed variation.
3. Similar stratum means by base count c_i across methods at matched pass@1 (e.g. pass@1 near 58:
   c = 0 about 9%, c = 1-2 about 33%, c = 3-8 about 67%, c = 9-16 about 79%, c = 17-24 about 88%).
4. This smoothed uniform success-logit-shift predictor fits poorly: a one-parameter shift of the base rates (Jeffreys
   posterior mean (c_i + 0.5) / 33), fitted to each checkpoint's pass@1, overpredicts pass@16 by about 10 points (71
   evaluations including six at T > 1; RMSE 10.0). Integrating the Jeffreys posterior instead of its mean gives RMSE
   5.55 (Codex's recomputation). The project's frozen frontier law (law.py, tau = -0.20) gives 0.82 pp RMSE on the 65
   T = 1 checkpoints: a per-problem law from base counts already predicts the curve.

## Reading

At matched pass@1, methods are close to Dr. GRPO training seeds not only in pass@16 but in which problems they solve
and how often (within about 6% of the seed-level per-problem distance overall; advantage-rule changes leave a small
signature), although their weight updates differ (PSR and NSR most of all). The shared object is approximately the
per-problem success profile, a behavioural quantity that a per-problem law from base counts predicts; many different
weight changes produce it.

## Next test (job worth-p27-base-s0-20261028-055034)

Does sharpening the base model alone reach the same profile? The base model at T = 0.4, 0.6, 0.8 (32 samples), and at
T = 1 with 128 samples (also extends the restricted BOPTR comparison). If base-at-lower-temperature lies on Dr. GRPO's curve and
matches its per-problem profile at equal pass@1, the curve is a property of the base model's own distribution, which
RL's effect is consistent with a temperature surrogate (this cannot by itself establish the mechanism); matching the
curve but not the profiles leaves problem-specific differences; falling below means the tested temperatures give less
coverage than RL; lying above means temperature gives better coverage; no pass@1 overlap is inconclusive. Caveats:
the base truncates 35.8% of samples at T = 1 and 1,024 tokens, so length and truncation are monitored; temperatures
should be chosen on validation problems before comparing held-out profiles.
