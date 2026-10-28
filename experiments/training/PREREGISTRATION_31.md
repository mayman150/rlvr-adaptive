# Pre-registration 31: Flat-RLVR (sharpness-aware RLVR) on the matched-pass@1 curve

Declared October 28, 2026, before any run of this design; reviewed by Codex before launch.

## Why

The October 28 literature review (paper/LIT_2026_10_28.md) named Flat-RLVR (arXiv 2610.29281) as the strongest
reported case of equal rounded aggregate pass@1 with higher pass@256: five-benchmark mean pass@1 16.3 for DAPO and
Flat-DAPO, pass@256 65.2 vs 68.6 (intervals over three training seeds). Its MATH500 numbers show little endpoint
coverage improvement (GRPO 34.6/72.7 vs Flat-GRPO 34.0/72.5; DAPO 35.7/72.1 vs Flat-DAPO 35.7/72.4, pass@1/pass@256);
matched-curve equivalence was not established there. Its setting is close to ours (Qwen2.5-Math-1.5B, MATH training
problems, 1,024 tokens, learning rate 1e-6, no KL).

## Arms (Qwen2.5-Math-1.5B, Dr. GRPO, Pre-registration 21's recipe, seeds 8, 9, 10)

Flat-RLVR's SAM step (Algorithm 1 and Appendix B; Table 11 settings) added to Dr. GRPO: at steps t (1-indexed) with
t mod n = 0, gradient centralization of the Dr. GRPO gradient (each row of every weight gradient with two or more
dimensions has its mean removed), radius rho_t = rho_min + (rho_init - rho_min)(1 + cos(pi t / T)) / 2 with
rho_init = 0.01, rho_min = 0.002, T = 150, multiplied by the batch success rate (focal weight, gamma = 1) and floored
at 0.001; ascent eps = rho g / ||g|| (global L2 norm); the gradient recomputed on the same answers at theta + eps and
centralized; parameters restored; AdamW uses the recomputed gradient (clipped at 1 as usual). At theta + eps the
gradient is that of the paper's surrogate with the policy frozen at theta (Eq. 2: token ratios exp(log pi(theta + eps)
- log pi(theta)), unclipped, summed with Dr. GRPO's constant normalizer), which equals the Dr. GRPO gradient at theta.
SAM with cached original-perturbation restoration (Algorithm 1), not Appendix D's memory approximation. Other steps are
plain Dr. GRPO. The SAM step index t is 1-indexed and logged.
- **FL10:** n = 10 (the paper's setting).
- **FL5:** n = 5 (in the paper's AMC23/AIME25 results, a variant with more coverage and lower pass@1).
Differences from the paper, stated: Dr. GRPO (no std normalization, constant normalizer) instead of GRPO/DAPO; 16
problems x 8 answers per step, 150 steps (15 SAM opportunities for FL10, 30 for FL5), constant learning rate, no
weight decay, AdamW beta2 0.99 instead of 0.999 (Pre-registration 21's recipe). A null therefore concerns this
adaptation and exposure; it cannot refute the paper's broader claim.

## Evaluation and decision (511 test problems, 32 samples, T = 1, 1,024 tokens, vLLM; checkpoints 25, 50, 100, 150)

Reference: B1k seeds 8-10 (trajectory and temperature envelope; the trajectory-only contrast is also reported).
Primary metric: pass@16 in percentage points at matched pass@1, averaged over eligible checkpoints (25, 50, 100, 150)
within each seed, then over seeds; Pre-registration 21's estimator and guards; significance requirements are
Holm-adjusted over the two arms.
- **Moves the curve:** pooled gain >= +2, 95% interval above 0, positive in all three seeds.
- **Average equivalence:** the pooled and each seed's 90% interval within +/-2.
- Otherwise inconclusive; more than 10% clamped in a seed: not interpreted.
Reported: step-150 pass@1 minus B1k's; the SAM steps' radii and batch success rates; pass@16 by base-count stratum;
pass@32 at matched pass@1 (the largest k our 32 samples support; the paper's pass@256 claim is not tested here).

## Cost (estimate)

Six single-GPU runs (SAM adds one backward pass every n steps, about +5-10% time) with their evaluations, on one
8 x A100 machine shared with Pre-registration 30's Stage 0: about 5-6 hours of the machine, about $170-190 in total for
the shared job.
