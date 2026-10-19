# Pre-registration 11: is the shared curve a sharpening curve? (temperature diagnostic)

Declared October 19, 2026, before any evaluation of this design (see this file's commit time).
Revised once before launch after an independent review (Codex): narrower verdict names, a
crossing rule, the interpolation convention, D at the entropy-matched temperature, validated
inputs, token lengths and truncation, and a temperature of 1.07. Evaluation only: no training.
It sets the priority for the method that Pre-registration 12 builds; it does not prove that
the chosen method will work.

## Why

pass@k at matched pass@1 depends only on how success is spread across questions:
pass@k = mean over questions of 1 - (1 - p_x)^k, concave in p_x, so a mean-preserving spread of
the p_x lowers it. RL makes the model sharper (probe entropy falls from about 0.6-0.9 to about
0.22-0.25 by step 100 in every run), and global sharpening spreads p_x. If the shared
accuracy-versus-diversity curve of Pre-registrations 5-10 is mostly a sharpening curve, then
undoing sharpening at inference (a higher sampling temperature on a late checkpoint) should
trace the same curve as stopping training earlier. If late checkpoints at a higher
temperature lie above the training trajectory, competence and sharpening are separable at
inference, which makes a sharpening-limiting training method worth testing (it does not show
that such a method works). If they lie below, one hypothesis is that training loses coverage
that temperature cannot restore, which makes an information-adding method (memory) the
higher priority; temperature comparisons cannot establish either conclusion about training
interventions.

## Models and evaluation

Qwen2.5-Math-1.5B; the PSR runs of Pre-registration 10 (seeds 5, 6, 7; checkpoints saved at
steps 50, 100, 150; no new training).

- Base model at temperatures 0.5, 0.7, 0.85, 1.0.
- PSR steps 50 and 100 at temperature 1.0.
- PSR step 150 at temperatures 1.0, 1.07, 1.15, 1.3, 1.45, 1.6.

Each evaluation: the fixed, subject-stratified set of 256 held-out MATH test problems
(`data/eval_frontier_256.json`; none in training or MATH-500), 32 samples per problem, top-k and
top-p off, answers up to 512 tokens, the same sampling seed; per-problem counts, answer texts,
mean answer length in tokens, the share of answers cut at 512 tokens, and the parseable
final-box rate saved. Results are validated on loading (32 samples, 256 problems, the stated
temperature, prefix entropy present). **Prefix entropy:** mean per-token entropy of
softmax(logits / T) on 256 fixed base-model answers (2 per problem on 128 MATH-500 problems,
from Pre-registration 8's step-0 samples, `data/frozen_prefixes_qwen.json`), so models are
compared on the same prefixes rather than on their own samples (token-weighted, on retokenized
text without the stop token, cut at 512 tokens; it differs from the training-time probe).

## Quantities

The **T = 1 trajectory** of a seed is the curve through (pass@1, pass@16) of the base model and
PSR steps 50, 100, 150, all at T = 1, ordered by pass@1 (ties by step), with linear
interpolation; in the bootstrap, a point outside the resampled range takes the end value, and
how often that happens is reported. For each temperature T > 1 of step 150 whose pass@1 lies inside the
trajectory's pass@1 range, D(T) = pass@16(step 150, T) minus the trajectory's interpolated
pass@16 at that pass@1. A seed's D is the mean over its qualifying temperatures; the pooled D is
the mean over seeds. Uncertainty: 2,000 bootstrap draws of the 256 problems (the same draw for
every model, temperature and seed), recomputing everything with the qualifying temperatures
fixed from the full data.

## Decision rule (primary)

- **Separable** (a late checkpoint at a higher temperature beats the training trajectory):
  pooled D at least +2 points and its 95% interval above 0.
- **Average agreement:** the 90% interval of the pooled D inside [-2, +2] (no average
  separation; not pointwise equivalence).
- **Below** (raising the temperature costs more coverage than training less): pooled D at most
  -2 points and its 95% interval below 0.
- **Crossing:** added to any verdict if some (seed, temperature) points' simultaneous band lies
  above +2 and others' below -2; the method choice then waits for more evaluation.
- **Inconclusive** otherwise; **insufficient overlap** if any seed has no qualifying
  temperature; **incomplete evaluation** if any listed evaluation is missing.

Per-seed values and a simultaneous 90% band over all qualifying (seed, temperature) points are
reported.

## Reported without a rule

- The entropy-matched temperature T* of step 150 (prefix entropy equal to the base model's at
  T = 1, by linear interpolation in T) and D at T* (pass@1 and pass@16 interpolated in T) where
  it is inside the range.
- The base model's temperature curve (T = 0.5 to 1.0) against step 150's (T = 1.0 to 1.6) in the
  pass@1 range where both exist: does training add coverage at matched accuracy beyond what the
  base model reaches by temperature alone?
- Mean answer length, truncation and parseable-box rates at every point (high temperature can
  break formatting; a comparison point whose box or truncation rate differs from the
  trajectory's, interpolated at the same pass@1 including the base model, by more than 10
  points is flagged).

## Research priority for Pre-registration 12 (not conclusions)

- **Separable:** build a sharpening-limiting method (Dr. GRPO with an entropy floor on frozen
  prefixes, enforced on the actual optimizer step) and test it against Dr. GRPO plus
  temperature calibration and against adaptive entropy control.
- **Average agreement** or **below:** inference-time de-sharpening does not recover coverage
  here, so a sharpening-limiting method becomes the lower priority (a hypothesis, not a
  conclusion about training); build an
  information-adding method (memory of verified successes) and test it against the same
  baselines.
- **Inconclusive** or **insufficient overlap:** enlarge the evaluation before choosing.

## Cost

28 evaluations of 8,192 answers, about 30 GPU-hours on single L40S GPUs (about $60-80), in 7 jobs.

## Result and enlargement (October 19)

Primary verdict on the 256 problems: **inconclusive** (pooled D -1.54, 90% [-2.63, -0.47]; T 1.07
and 1.15 near 0 to +2 per seed, T 1.3 about -6 to -7.5 and format-flagged). Output:
`RESULTS_11_analysis_output.txt`. As pre-registered, the evaluation is enlarged before Pre-
registration 12 chooses a branch: the evaluations that enter D (base at T = 1; PSR steps 50, 100,
150 at T = 1; step 150 at T = 1.07, 1.15, 1.3; seeds 5-7) are repeated on the 256 further problems
of `data/eval_frontier_256b.json` (the evaluation set of Pre-registration 12 minus this set, labels
corrected), and the same rule is applied to the combined 511 problems (the audit-excluded item
dropped). **Declared before the enlarged results exist:** if the enlarged verdict is still
inconclusive (or average agreement, or below), Pre-registration 12 runs branch M; only
"separable" selects branch S.
