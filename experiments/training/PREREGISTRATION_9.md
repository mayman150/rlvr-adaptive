# Pre-registration 9: does the one-curve result hold for a second model family?

Declared October 18, 2026, before any run of this design (see this file's commit time).
Revised before launch after an independent review (Codex), which asked for uncertainty-based
decision rules, exact windows, explicit inconclusive outcomes, a model-identity check, and
lower generation memory. The first draft was never run.

## Why

Pre-registrations 4 to 8 (Qwen2.5-Math-1.5B, 150 steps) found that PSR, NSR, baselines in
between, the NSR paper's implementation, and a pass@4 objective trace one pass@1-versus-
pass@32 curve and differ in speed, with speed ordered by the theorem's predicted noise. Those
were point comparisons on 128 problems, and equivalence within 2 points was not
statistically established. Qwen2.5-Math also reacts to random rewards. This test uses a
non-Qwen family, longer training, and a larger evaluation with intervals.

## Model and training

OLMo-2-0425-1B-SFT (AllenAI, Apache-2.0, 1.48B parameters, never RL-trained). Launch
asserts the loaded architecture is Olmo2ForCausalLM. Same data, verifier, AdamW 1e-6,
16 questions by 8 answers per step, strictly on-policy, sequence-sum loss, **300 steps**,
generation batch 64. Calibration on 64 questions (512 answers) with per-answer norms saved.

Arms, seeds 1 and 2: Dr. GRPO; PSR; NSR; pass@4 objective (exactly unbiased, verified over
all 256 reward patterns); optimal answer-collection rule at 128 answers per step on average.
The optimal rule is a **transferred heuristic**: its signal shape and noise ratio R = 130 were
measured on Qwen; only the difficulty prior and F0 / F1 are measured on OLMo.

## Measurements

- Training jobs: greedy accuracy on the 364 numeric MATH-500 problems at steps 0, 50, ...,
  300; per step: first-block training accuracy (an unbiased estimate on fresh questions for
  every arm), boxed-answer rate, truncation rate, entropy probe every 5 steps; split-half
  noise diagnostics in PSR and NSR; bf16 checkpoints at steps 50, 100, ..., 300.
- Evaluation jobs: on a fixed set of **1,024 held-out MATH test problems** (none in MATH-500
  or training; `data/eval_frontier_1024.json`), 32 samples per problem at temperature 1 for
  the base model and every checkpoint; answer texts saved. Greedy accuracy on all 3,382
  held-out problems at step 300.
- Accuracy is pass@1 (mean share correct); **diversity is pass@16** (unbiased from 32
  samples, less noisy than pass@32, which is reported as secondary).

## Predictions and decision rules

1. **Asymmetry.** F0 / F1 from calibration: pass if the 95% bootstrap interval (over
   answers) lies above 1.2 in both seeds.
2. **Noise and speed.** (a) PSR's and NSR's measured per-question second moment over steps
   0 to 49: pass if NSR / PSR exceeds 1 with a 95% interval (bootstrap over steps) above 1 in
   both seeds. (b) Speed: T is the first step at which the 10-step mean first-block training
   accuracy reaches the midpoint between that run's step-0 value and PSR's mean over steps
   250 to 299 (same seed); runs that never reach it get T = 300. Pass if NSR's T exceeds
   PSR's in both seeds.
3. **One curve (primary).** For a run R against a reference P of the same seed: at each of
   R's checkpoints after step 0 whose pass@1 lies inside P's pass@1 range, interpolate P's
   pass@16 linearly at R's pass@1 along P's trajectory (base model, then steps 50 to 300);
   take the mean difference over those checkpoints. If fewer than two of R's checkpoints
   qualify in a seed, that seed is "insufficient overlap". The pooled estimate is the mean
   over seeds. Uncertainty: bootstrap the 1,024 problems with replacement (the same draw for
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
4. **Speed at equal cost.** Optimal rule minus Dr. GRPO, greedy, 3,382 problems, step 300,
   paired by problem and pooled over seeds: **faster** if the estimate is at least +1 point
   and its 95% interval excludes 0; **slower** if the interval lies below 0; otherwise
   inconclusive. Total generated answers must be within 2% of 38,400; tokens and GPU time are
   reported.

Formatting: boxed-answer and truncation rates are reported per arm; a comparison where those
rates differ by more than 10 points is flagged, since format learning can look like accuracy.
