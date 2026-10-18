# Pre-registration 9: does the one-curve result hold for a second model family?

Declared October 18, 2026, before any run of this design (see this file's commit time).
Revised twice before launch after independent reviews (Codex), which asked for
uncertainty-based decision rules, exact windows, explicit inconclusive and insufficient-
overlap outcomes, correct pooling across seeds, a model-identity check, lower generation
memory, an answer-budget gate, and enough training questions. The earlier drafts were never
run.

## Why

Pre-registrations 4 to 8 (Qwen2.5-Math-1.5B, 150 steps) found that PSR, NSR, baselines in
between, the NSR paper's implementation, and a pass@4 objective trace one pass@1-versus-
pass@32 curve and differ in speed, with speed ordered by the theorem's predicted noise. Those
were point comparisons on 128 problems, and equivalence within 2 points was not
statistically established. Qwen2.5-Math also reacts to random rewards. This test uses a
non-Qwen family, longer training, and a larger evaluation with intervals.

## Model and training

OLMo-2-0425-1B-SFT (AllenAI, Apache-2.0, 1.48B parameters, never RL-trained). Launch
asserts the loaded architecture is Olmo2ForCausalLM. Training questions: all seven MATH
training subjects with numeric answers (5,685; the earlier runs used four subjects, 3,920,
which 300 steps of 16 questions would exceed), none in any test set. Verifier, AdamW 1e-6,
16 questions by 8 answers per step, strictly on-policy, sequence-sum loss, **300 steps**,
**answers up to 1,024 tokens** (a GPU smoke test found 37 to 62% of OLMo's answers truncated
at 512, which would let length and format learning pass for accuracy), generation batch 64 (halved automatically on out-of-memory). Calibration on 64 questions
(512 answers) with per-answer norms and their questions saved.

Arms, seeds 1 and 2: Dr. GRPO; PSR; NSR; pass@4 objective (exactly unbiased, verified over
all 256 reward patterns); optimal answer-collection rule at 128 answers per step on average.
The optimal rule is a **transferred heuristic**: its signal shape and noise ratio R = 130 were
measured on Qwen; only the difficulty prior and F0 / F1 are measured on OLMo.

## Measurements

- Training jobs: greedy accuracy on the 364 numeric MATH-500 problems at steps 0, 50, ...,
  300; per step: first-block training accuracy (an unbiased estimate on fresh questions for
  every arm), boxed-answer rate, truncation rate, entropy probe every 5 steps; split-half
  noise diagnostics in PSR and NSR; bf16 checkpoints at steps 50, 100, ..., 300.
- Evaluation jobs (which fail if any expected checkpoint is missing): on a fixed, seeded, subject-stratified set of **256 held-out MATH test problems**
  (none in MATH-500 or training; `data/eval_frontier_256.json`, drawn proportionally by subject from
  `data/eval_frontier_1024.json` with Python's random.Random(256); 256 rather than 1,024 because the smoke test measured
  about 2 answers per second at 1,024 tokens, and a bootstrap on earlier data gives a 90%
  half-width near 1 point at this size), 32 samples per problem at temperature 1 for
  the base model and every checkpoint; answer texts saved. Greedy accuracy on all 3,382
  held-out problems at step 300.
- Accuracy is pass@1 (mean share correct); **diversity is pass@16** (unbiased from 32
  samples, less noisy than pass@32, which is reported as secondary).

## Predictions and decision rules

1. **Asymmetry.** F0 / F1 from calibration: pass if the 95% bootstrap interval (resampling
   the 64 calibration questions) lies above 1.2 in both seeds.
2. **Noise and speed.** (a) PSR's and NSR's measured per-question second moment over steps
   0 to 49: pass if NSR / PSR exceeds 1 with a 95% interval (bootstrap over steps) above 1 in
   both seeds. (b) Speed: T is the first step at which the 10-step mean first-block training
   accuracy (full windows only, ending at step 9 or later) reaches the midpoint between that
   run's mean over steps 0 to 9 and PSR's mean over steps 250 to 299 (same seed); runs that
   never reach it get T = 300. Pass if NSR's T exceeds PSR's in both seeds.
3. **One curve (primary).** For a run R against a reference P of the same seed: at each of
   R's checkpoints after step 0 whose pass@1 lies inside P's pass@1 range, interpolate P's
   pass@16 linearly at R's pass@1 along P's trajectory (base model, then steps 50 to 300);
   take the mean difference over those checkpoints. Every expected evaluation (steps 0 and
   50 to 300 for both runs and both seeds) must be present, otherwise the verdict is
   **incomplete evaluation**. The qualifying checkpoints are fixed
   from the full data. If fewer than two qualify in either seed, the comparison's verdict is
   **insufficient overlap** (no seed is dropped). The pooled estimate is the mean over the
   two seeds. Uncertainty: bootstrap the 256 problems with replacement (the same draw for
   all runs and checkpoints of a comparison), recomputing everything, 2,000 draws. Applied
   to NSR against PSR and pass@4 against Dr. GRPO:
   - **one curve** if the 90% interval lies inside [-2, +2] points (two one-sided tests at
     5%);
   - **different** if the 95% interval excludes 0 and the estimate is at least 2 points away
     from 0;
   - **inconclusive** otherwise.
   The root-mean-square of the pointwise differences and the overlap range are reported.
   The bootstrap covers problem sampling, not run-to-run variability; with two seeds the
   seed-to-seed spread is reported as a pilot estimate.
4. **Speed at equal cost.** Optimal rule minus Dr. GRPO, greedy, 3,382 problems, step 300:
   per problem, average the two seeds' paired differences, then take the mean and its 95%
   interval over problems. **Faster** if the estimate is at least +1 point and the interval
   excludes 0; **slower** if the interval lies below 0; otherwise inconclusive. If either
   seed's total generated answers is not within 2% of 38,400, the verdict is **invalid
   answer budget**; if either seed's complete training log is missing, it is **missing data**. Tokens and GPU time are reported.

Formatting: boxed-answer and truncation rates are reported per arm; a comparison where either
rate differs by more than 10 points is flagged, since format learning can look like accuracy.

## Reporting addendum (October 18, before any evaluation result)

After a review of Pre-registration 10's design pointed out that a signed average over
checkpoints can hide crossings, two reporting changes apply here too; the decision rules are
unchanged. The verdict "one curve" is reported as **average matched-accuracy equivalence**,
and a simultaneous 90% band over all qualifying checkpoints (bootstrap of the maximum
standardized deviation) is reported for each comparison. Reference points with equal pass@1
are ordered by step.
