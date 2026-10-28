# Core reading: six papers and how they connect to our results

October 28, 2026. Summaries from the arXiv abstracts (fetched today) and our notes; "Connection" refers to our
registered results (RESULTS_18-23, RESULTS_26_interim).

## Notation

- pass@k: probability that at least one of k sampled answers is correct (unbiased estimator from 32 samples).
- Matched pass@1 comparison: an intervention's pass@16 minus Dr. GRPO's pass@16 at the same pass@1, read off Dr.
  GRPO's training trajectory and temperature envelope (Pre-registration 21's estimator).
- "The curve": Dr. GRPO's pass@16-vs-pass@1 trajectory.

## 1. DeepSeekMath (Shao et al., 2024; arXiv 2402.03300)

**What it does.** Continues pre-training a 7B code model using a 120B-token math corpus alongside language and code
data, and introduces GRPO: a PPO variant
without a value network, whose baseline is the mean reward of a group of answers to the same prompt (advantages are
also divided by the group's reward standard deviation).
**Connection.** GRPO is the algorithm family we study. PSR and NSR change how group rewards become advantages; AC
reallocates rollouts across problems while keeping Dr. GRPO's advantages. (The matched-pass@1 analysis itself does not
require group centering.)

## 2. DeepSeek-R1 (DeepSeek-AI, 2025; arXiv 2501.12948)

**What it does.** Shows that reasoning ability can be improved by pure RL with verifiable rewards, without human
reasoning demonstrations; R1-Zero omits any preliminary SFT; the smaller released models are distilled from
DeepSeek-R1.
**Connection.** Our primary arms are R1-Zero-style (RL directly from Qwen2.5-Math-1.5B and 7B base models, no SFT);
separate warm-start and human-solution arms use supervision. Distillation motivates the teacher arms (TD, RTD), whose
outcomes are not yet available.

## 3. Dr. GRPO (Liu et al., 2025; arXiv 2503.20783)

**What it does.** Examines R1-Zero-like training: base models (Qwen2.5 bases reason well even without prompt templates,
suggesting pretraining effects) and GRPO, which it finds has an optimization bias that inflates response length,
especially of incorrect answers. Dr. GRPO removes this bias (and the per-group standard-deviation normalization),
giving a minimalist recipe (43.3% on AIME 2024 with a 7B base).
**Connection.** Dr. GRPO is our baseline: advantage r - mean r, 128 rollouts per step, and a fixed normalizer within
each setting (512, 1,024 or 2,048-token caps). Within each setting, the supported finite-horizon comparisons show
average agreement with Dr. GRPO's curve; attainable ceilings were not measured.

## 4. Does RL really incentivize reasoning beyond the base model? (Yue et al., 2025; arXiv 2504.13837)

**What it does.** Measures pass@k at large k across model families, RL algorithms and benchmarks. RL-trained models
beat their base models at small k, but the base models win at large k; coverage and perplexity analyses suggest RL's
reasoning is bounded by the base model; six popular RLVR algorithms perform similarly; distillation, in contrast, can
introduce new reasoning patterns from the teacher.
**Connection.** Our matched-accuracy evidence is complementary to their "algorithms perform similarly". We compare at
matched pass@1 rather than at a fixed training step, which separates speed along the curve from a change of the curve.
Pre-registration 21 (1.5B, 1,024 tokens) supports four of four average-equivalence contrasts; Pre-registration 20 (7B,
2,048 tokens) supports three of four. Agreement is setting-specific, within +/-2 points over supported checkpoints.
Their distillation result motivates the teacher tests, whose efficacy is not yet known.

## 5. Negative reinforcement (Zhu et al., 2025; arXiv 2506.01347; NeurIPS 2025)

**What it does.** Splits the RLVR signal into reinforcing correct answers (PSR) and penalizing incorrect ones (NSR).
Training with NSR alone improves pass@k across k up to 256, often matching PPO and GRPO; PSR alone improves pass@1 but
degrades higher k through lost diversity. Proposes an objective that upweights NSR.
**Connection.** We test PSR and NSR at matched pass@1. At 7B (2,048 tokens) both are equivalent to Dr. GRPO within
the registered +/-2 margin (PSR -0.09, NSR -0.09 pass@16 points); at 1.5B and 1,024 tokens, NSR vs PSR is -0.25 and
neither departs from Dr. GRPO. Their fixed-step differences are therefore consistent, in our setting, with different
speeds along the same curve rather than different curves. The 7B equivalence results concern the 511-problem test
set; the direct NSR-PSR comparison at 7B is not interpretable (13.7% clamping in seed 9). Both studies include
Qwen2.5-Math-7B, but protocols differ; our results are consistent with speed differences within our setting.

## 6. Budgeted search perspective (Sun et al., 2026; arXiv 2609.01274)

**What it does.** Asks whether an RL model's pass@k curve can be reproduced by the base model under search. Using
paired base/RL checkpoints (SimpleRL-Zoo) and a unified decoding framework, it finds a rule N_base ~ alpha N_RL^beta
(BOPTR) mapping RL operating points to base operating points, with transfer error about 3-5 pass@k points across ten
models and four benchmarks. Reading: much of the RL gain is a change in sampling efficiency toward operating points
the base model can already reach.
**Connection.** BOPTR is a closely related descriptive prior and a novelty risk. They map base and RL models through
decoding budgets; we compare training interventions at matched pass@1 with pre-registered tests and token-budget
checks. Our interventions so far did not move the curve: the allocation forecast failed, minimum-hint credit failed
selection in one validation seed, Pre-registration 26 failed the saved-update shared-direction criterion and its
gradient diagnostic is inconclusive; teacher outcomes are pending. Fitting BOPTR on validation and freezing it on
held-out checkpoints is a proposed comparison.
