# Restricted BOPTR baseline (Sun et al. 2026, arXiv 2609.01274) vs Dr. GRPO's curve

October 28, 2026; `boptr.py`, output `results_p26/boptr_1k.txt`. Not pre-registered (a descriptive comparison
requested by the October 27 review).

## Notation

- pass@k: unbiased estimate from 32 samples per problem (T = 1), 511 test problems, 1.5B, 1,024 tokens.
- BOPTR: an RL checkpoint's pass@k is approximated by the base model's pass@N, N = alpha k^beta. Restricted here to one
  decoder (T = 1 sampling), so N is limited to [1, 32]; the published rule also selects search policies.
- Fit set: Dr. GRPO's evaluations (B1k seeds 8-10, steps 25-150, T = 1, 1.15, 1.3; 18 evaluations). Held out: all
  other 1,024-token arms (PSR1k, NSR1k, AC1k, HG1k, HGm1k, PG1k; 53 evaluations).
- Curve predictor: pass@16 at the checkpoint's pass@1 on the pooled Dr. GRPO points (base included), linearly
  interpolated. BOPTR predictor: beta fitted on the fit set; alpha from base pass@alpha = checkpoint pass@1; prediction
  base pass@(alpha 16^beta).

## Result

- beta = 0.845 (fit RMSE 2.28 pp over k = 2-16 on the fit set).
- Only 8 of 53 held-out evaluations are scorable by BOPTR: the base model's pass@32 (85.5) is below most trained
  checkpoints' pass@16 (about 86-87), so alpha 16^beta exceeds the 32 available base samples.
- On those 8 (PSR1k, NSR1k, PG1k at early steps): pass@16 RMSE 0.63 pp for the curve vs 1.93 pp for BOPTR (mean error
  +0.37 vs +1.81).

## Reading and next step

The restricted BOPTR baseline is mostly unscorable with 32 base samples; where it is scorable, Dr. GRPO's curve predicts
held-out pass@16 about three times more accurately. A full comparison needs base-model evaluations with more samples
(e.g. 256 per problem on the 511 test problems, about one GPU-hour with vLLM).
