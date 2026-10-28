# Pre-registration 30: change the starting model, predict the post-RL curve

Declared October 28, 2026, before any run of this design; reviewed by Codex before launch.

## Why

Interpretable comparisons of near-on-policy RLVR recipes on Qwen2.5-Math-1.5B/7B approximately follow Dr. GRPO's
pass@16-vs-pass@1 curve, and a frozen per-problem law computed from the starting model's success counts predicts that
curve (0.23-0.82 points RMSE; first public 7B checkpoints within about 1 point). If the curve is set by the starting
distribution, changing the start should move it, and the frozen law should predict by how much. Prior work shows that
the start matters (OctoThinker 2506.20512; 2512.07783; TailSFT 2608.25756) and that pre-RL pass@large-k and held-out
loss predict RL outcomes (Quagmires 2510.01624; BOPTR 2609.01274). The open contribution is a prospective,
quantitative forecast of the post-RL curve shift caused by a controlled change of the start. The law predicts pass@k
conditional on pass@1; it does not predict the RL endpoint or the compute to reach it, and no claim of that is made.
A prior failure is known: from the S32 warm start's own counts the law missed with RMSE 3.56 (Pre-registration 17), so
the test can falsify it.

## Stage 0: definitions (evaluation only)

The 1.5B base, 64 samples per training problem (3,920 MATH training problems), T = 1, 1,024 tokens, vLLM; per problem the
success count c and the distribution of final boxed answers. Selections (training problems only; MATH test problems
are disjoint by construction, and exact-duplicate questions between train.json and the evaluation files are removed):
- **R (random):** problems drawn uniformly (seed 3030).
- **H (rare success):** c <= 4 of 64.
- **W (confidently wrong):** c <= 4 and the most frequent final boxed answer is wrong and appears in at least 16 of 64
  samples (behavioural concentration on one wrong answer; token confidence is not used).
Unfinished-heavy problems (at least half the samples without a final box) are reported separately. Subject mix and the
teacher solutions' mean length are reported for each selection.

**Teacher pool:** Qwen2.5-Math-72B-Instruct solutions from Pre-registration 29a (4 samples per training problem at
2,048 tokens), verified, retokenized for the 1.5B and kept if at most 1,023 tokens. One pool serves every selection.

## Stage 1: screening the starts (no RL)

Supervised fine-tuning from the base (Pre-registration 14's warm-start recipe: mean token NLL, AdamW 1e-5, batches of
8, clip 1) on each selection's teacher solutions, cycling through a problem's distinct solutions before repeating, at
two doses of response tokens, 0.5M and 2M (unique examples and repetitions reported). Starts: R-0.5, R-2, H-0.5, H-2,
W-0.5, W-2, and TailSFT-2 (an adaptation of TailSFT on the R pool: sequences the current model already fits are
dropped during SFT; the filter follows the paper and is stated in the code before launch). Plus the base.

Each start is evaluated on the 255 validation and the 511 test problems (32 samples, T = 1, 1,024 tokens), with
truncation, boxed rate and length. **Forecast (frozen before any Stage-2 run):** the law (law.py, tau = -0.20,
law.fit_prior / law.initial_theta on the start's own 32-sample counts) gives the predicted post-RL pass@16 at pass@1
targets 0.45, 0.50, 0.55, 0.58; the predicted shift of start X is the mean over targets of (prediction for X minus
prediction for the base), separately on validation and test. Competing predictors, also frozen: the change in the
start's pass@1, in its pass@32, and in its mean token NLL on the validation problems' human-written solutions.

**Selection for Stage 2 (validation forecasts only):** the base plus up to three starts with the largest absolute
predicted shift; a start advances only if its validation-predicted shift is at least 2 points in absolute value and
its validation pass@1 is within the law's reachable range. If no start reaches 2 points, the start with the largest
predicted shift advances and the test is labelled weak (a near-zero predicted effect gives little evidence).

## Stage 2: RL from the chosen starts (the decisive test)

Dr. GRPO, Pre-registration 21's recipe (1.5B, 1,024 tokens, 16 x 8, AdamW 1e-6, 150 steps), seeds 8, 9, 10, from each
chosen start (the SFT stage reruns inside each training run with the same seed, so the start is identical to the
screened one up to sampling); checkpoints 25, 50, 100, 150 on the 511 test problems (32 samples, T = 1; step 150 also
at T = 1.15, 1.3). Reference: B1k seeds 8-10 (Pre-registration 21), trajectory and temperature envelope.

**Readings fixed in advance** (Pre-registration 21's estimator: matched-pass@1 pass@16 gain against B1k's envelope,
checkpoint-averaged within seed, seeds equally weighted, problem bootstrap with 2,000 draws; at least two eligible
checkpoints per seed, otherwise "insufficient overlap"; intervals condition on the trained seeds):
- **Law supported for start X:** the observed shift and the test-set predicted shift differ by at most 1.5 points and
  the 90% interval of (observed - predicted) lies within +/-2.5.
- **Law falsified for start X:** observed - predicted has |estimate| >= 2 with a 95% interval excluding 0
  (Holm over the chosen starts).
- Otherwise inconclusive.
- **The curve moved:** observed shift >= +2 with a 95% interval above 0 and positive in all seeds (reported for each
  start regardless of the law's verdict).
- **Predictor comparison (reported):** across starts, the rank agreement and absolute error of each predictor (law,
  start pass@1, start pass@32, held-out NLL) against the observed shifts; with at most three starts this is descriptive.
Reported confounds: finishing, boxed rate, truncation and length of every start and checkpoint; pass@16 by base-count
stratum (Stage-0 counts); per-problem profile distance to B1k at matched pass@1 (Investigation 27's statistic).

## Cost (estimates)

Stage 0: 3,920 x 64 samples, about 30 minutes on one 8 x A100 machine (about $15). Stage 1: seven SFT runs and
fourteen evaluations, about 1 hour on one machine (about $30). Stage 2: up to nine RL runs and their evaluations, about
55 GPU-hours (about 7 hours of one machine, about $220). Total about $270.
