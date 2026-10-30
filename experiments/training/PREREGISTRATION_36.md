# Pre-registration 36: a teacher-free, coverage-weighted warm start for RLVR

Declared October 30, 2026, after Pre-registration 35's phase 1 and before any run of this design; reviewed by Codex
before launch.

## Why

Pre-registration 35 (phase 1, RL seed 8): SFT on the base's own verified solutions to hard problems (SELF) gave an
early-RL matched pass@16 gap over Dr. GRPO of +2.41 [95%: +1.13, +3.63], as large as teacher solutions on the same
prompts (+2.23) and teacher solutions with corrupted final answers (+2.03), while moving pass@1 only from 21.4 to
23.8. The effect therefore does not require a teacher or correct answer keys. The theory of matched-accuracy coverage
says a gain needs success to move toward problems with low p_x, weighted by the marginal pass@k value
k(1 - p_x)^(k-1). This registration tests a method built on both: sample the SFT stream by that value, using the base's
own rare successes, then run standard RL.

## Arms (Qwen2.5-Math-1.5B; nominal dose 25K presented response tokens; sft_start.py defaults: AdamW 1e-5, batches of
8, clip 1; then Dr. GRPO with Pre-registration 21's recipe, 150 steps)

Self pool (p36_prepare.py; data/p36_prepare_report.json with hashes): 3,795 training problems with at least one
finished, verified base response among Stage 0's 64 samples (27,185 responses, at most 8 per problem, one terminal
format; finished = original sample at most 1,000 tokens). Problems with no verified base response (c = 0 of 64) are
not in the pool.
- **CWS (method):** problems drawn with replacement with weight 16 (1 - p_x)^15, p_x the Beta-prior posterior mean of
  the 64-sample counts (55% of draws on c = 1-4, 43% on c = 5-16; uniform: 9% and 37%).
- **UNI:** the same pool and draw mechanism with uniform weights.
- **TAIL:** UNI with TailSFT (arXiv 2608.25756, sft_start --tail).
- **HSELF:** Pre-registration 35's SELF start (hard problems only, shuffled passes), from its phase 2.
- **TIME (timing ablation):** HSELF's archive and selection applied during RL instead of before it: one SFT batch
  (same optimizer settings, its own AdamW) every 5 RL steps from step 0 until 25K presented tokens
  (train_grpo --isft-*), from the base model.
- **Reference:** Dr. GRPO from the base, B1k seeds 8-10 (and the long B1kL runs of Pre-registration 30, Amendment 5,
  for checkpoints beyond B1k's pass@1 range).
SFT seeds (CWS, UNI, TAIL): 3636, 3737, 3838, each paired with RL seed 8, 9, 10 respectively (SFT and RL variability
both enter the seed-level estimates). HSELF uses one SFT seed (3030) with RL seeds 8-10; TIME uses interleaved-SFT
seed 3030 with RL seeds 8-10.

## Evaluation

Checkpoints 1, 2, 5, 10, 25, 50, 100, 150 on the 511 test problems (32 samples, T = 1, 1,024 tokens); checkpoints
50, 100, 150 also at 2,048 tokens (reference: B1k seeds 8-10 re-evaluated at 2,048 tokens). Every start on the test
problems with 64 samples at 1,024 and 2,048 tokens.

## Readings (fixed now)

- **Each arm against Dr. GRPO:** Pre-registration 30's Stage 2 estimator and rules (same-seed B1k trajectory, eligible
  checkpoints, seeds weighted equally, 2,000 problem-bootstrap draws, support rule; "curve raised" at >= +2 with a 95%
  interval above 0 in the pooled estimate, all seed estimates positive and Holm p <= 0.05 over the five arms).
- **Method against each baseline (primary):** at each CWS checkpoint, pass@16 minus the same-RL-seed baseline
  trajectory (its start and RL checkpoints) interpolated at the checkpoint's pass@1, averaged over eligible checkpoints
  and equally over seeds; baselines UNI, TAIL, HSELF; Holm over the three. **CWS beats a baseline** if the pooled
  difference is at least +0.5 points with a 95% interval above 0, positive in at least two of three seeds, and Holm
  p <= 0.05; **not distinguishable** if the 90% interval lies within +/-0.5; otherwise inconclusive.
- **Timing:** TIME against HSELF by the same matched contrast (HSELF's trajectory as the reference); "before RL is
  better" if HSELF beats TIME by the method rule above.
- **Mechanism diagnostics (not verdicts):** matched pass@2 and wrong-answer collision for every arm
  (diversity_decomposition.py); pass@16 by base-count stratum; 2,048-token gaps; SFT tokens, updates and preparation
  cost (base samples drawn, verifier calls).
- **Decision (week 4):** the paper is a method paper if CWS beats UNI and TAIL; if CWS is not distinguishable from UNI
  but both raise the curve against Dr. GRPO, the claim is "self-SFT before RL" without a targeting benefit.

## Cost (estimate)

Nine SFT starts (minutes each) and twelve RL runs with evaluations (about 6.5 hours each on one A100), packed with
Pre-registration 35's phase 2 (twelve runs) on three 8 x A100 machines: about $600 in total for both.
