# Pre-registration 11: is the shared curve a sharpening curve? (temperature diagnostic)

Declared October 19, 2026, before any evaluation of this design (see this file's commit time).
Evaluation only: no training. It decides which method Pre-registration 12 builds.

## Why

pass@k at matched pass@1 depends only on how success is spread across questions:
pass@k = mean over questions of 1 - (1 - p_x)^k, concave in p_x, so a mean-preserving spread of
the p_x lowers it. RL makes the model sharper (probe entropy falls from about 0.6-0.9 to about
0.22-0.25 by step 100 in every run), and global sharpening spreads p_x. If the shared
accuracy-versus-diversity curve of Pre-registrations 5-10 is mostly a sharpening curve, then
undoing sharpening at inference (a higher sampling temperature on a late checkpoint) should
trace the same curve as stopping training earlier. If late checkpoints at a higher
temperature lie above the training trajectory, competence and sharpening are separable, and a
method that limits sharpening during training has room to move the curve. If they lie below,
training loses coverage that temperature cannot restore (forgetting), and a method must add
information (memory).

## Models and evaluation

Qwen2.5-Math-1.5B; the PSR runs of Pre-registration 10 (seeds 5, 6, 7; checkpoints saved at
steps 50, 100, 150; no new training).

- Base model at temperatures 0.5, 0.7, 0.85, 1.0.
- PSR steps 50 and 100 at temperature 1.0.
- PSR step 150 at temperatures 1.0, 1.15, 1.3, 1.45, 1.6.

Each evaluation: the fixed, subject-stratified set of 256 held-out MATH test problems
(`data/eval_frontier_256.json`; none in training or MATH-500), 32 samples per problem, top-k and
top-p off, answers up to 512 tokens, the same sampling seed; per-problem counts, answer texts,
mean answer length and boxed-answer rate saved. **Prefix entropy:** mean per-token entropy of
softmax(logits / T) on 256 fixed base-model answers (2 per problem on 128 MATH-500 problems,
from Pre-registration 8's step-0 samples, `data/frozen_prefixes_qwen.json`), so models are
compared on the same prefixes rather than on their own samples.

## Quantities

The **T = 1 trajectory** of a seed is the curve through (pass@1, pass@16) of the base model and
PSR steps 50, 100, 150, all at T = 1, ordered by pass@1 (ties by step), with linear
interpolation. For each temperature T > 1 of step 150 whose pass@1 lies inside the
trajectory's pass@1 range, D(T) = pass@16(step 150, T) minus the trajectory's interpolated
pass@16 at that pass@1. A seed's D is the mean over its qualifying temperatures; the pooled D is
the mean over seeds. Uncertainty: 2,000 bootstrap draws of the 256 problems (the same draw for
every model, temperature and seed), recomputing everything with the qualifying temperatures
fixed from the full data.

## Decision rule (primary)

- **Separable** (a late checkpoint at a higher temperature beats the training trajectory):
  pooled D at least +2 points and its 95% interval above 0.
- **Same curve:** the 90% interval of the pooled D inside [-2, +2].
- **Below** (raising the temperature costs more coverage than training less): pooled D at most
  -2 points and its 95% interval below 0.
- **Inconclusive** otherwise; **insufficient overlap** if any seed has no qualifying
  temperature; **incomplete evaluation** if any listed evaluation is missing.

Per-seed values and a simultaneous 90% band over all qualifying (seed, temperature) points are
reported.

## Reported without a rule

- The entropy-matched temperature T* of step 150 (prefix entropy equal to the base model's at
  T = 1, by linear interpolation in T) and D at T* where it is inside the range.
- The base model's temperature curve (T = 0.5 to 1.0) against step 150's (T = 1.0 to 1.6) in the
  pass@1 range where both exist: does training add coverage at matched accuracy beyond what the
  base model reaches by temperature alone?
- Mean answer length and boxed-answer rate at every point (high temperature can break
  formatting; a comparison point whose boxed-answer rate differs from the trajectory's by more
  than 10 points at the same pass@1 is flagged).

## What each outcome implies (Pre-registration 12)

- **Separable:** build a sharpening-limiting method (Dr. GRPO with an entropy floor on frozen
  prefixes, enforced on the actual optimizer step) and test it against Dr. GRPO plus
  temperature calibration and against adaptive entropy control.
- **Same curve** or **below:** limiting sharpening cannot help along this path; build an
  information-adding method (memory of verified successes) and test it against the same
  baselines.
- **Inconclusive** or **insufficient overlap:** enlarge the evaluation before choosing.

## Cost

25 evaluations of 8,192 answers, about 27 GPU-hours on single L40S GPUs (about $60-80), in 7 jobs.
