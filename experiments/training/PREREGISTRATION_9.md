# Pre-registration 9: does the one-curve result hold for a second model family?

Declared October 18, 2026, before any run of this design (see this file's commit time).

## Why

Pre-registrations 4 to 8, all on Qwen2.5-Math-1.5B for 150 steps, found that every rule we
tried (PSR, NSR, baselines in between, the NSR paper's implementation, a pass@4 objective)
traces one accuracy-versus-diversity curve and differs only in speed, which follows the
theorem's predicted noise. Qwen2.5-Math reacts even to random rewards, so the result needs a
second, non-Qwen family and longer training.

## Model and setting

OLMo-2-0425-1B-SFT (AllenAI, Apache-2.0, 1.48B parameters, not RL-trained; local check:
greedy accuracy 1 of 8 MATH-500 problems, boxes 50 to 75% of answers). Same data, verifier,
optimizer (AdamW 1e-6), 16 questions by 8 answers per step, strictly on-policy,
sequence-sum loss, but **300 steps**.

## Arms (seeds 1 and 2)

Dr. GRPO; PSR; NSR; pass@4 objective (verified unbiased); optimal answer-collection rule at
128 answers per step on average. Split-half noise diagnostics in PSR and NSR.

## Measurements

Greedy accuracy on the 364 numeric MATH-500 problems at steps 0, 50, ..., 300; pass@32 on the
first 256 of them at steps 0, 100, 200, 300 (answer texts saved for CoT-Pass@k); entropy
probe every 5 steps; greedy accuracy on the 3,382-problem held-out MATH set at step 300;
final models saved. Calibration measures F0 and F1 before training.

## Predictions

1. **Asymmetry:** F0 / F1 > 1.5 (wrong answers carry larger squared gradients).
2. **Noise and speed:** NSR's per-prompt noise exceeds PSR's (measured, steps 0 to 49), and
   NSR takes more steps than PSR to first reach a 10-step mean training accuracy halfway
   between its start and Dr. GRPO's level at step 300, in both seeds.
3. **One curve:** at matched pass@1 (interpolation rule of Pre-registration 5, points after
   step 0), |NSR minus PSR| and |pass@4 minus Dr. GRPO| mean pass@32 differences are below 2
   points in both seeds.
4. **Speed at equal cost:** the optimal rule minus Dr. GRPO on the 3,382-problem set
   (paired) is positive on average over the two seeds.

Each prediction is reported as passed or failed; a failure of 3 would mean the one-curve
result is specific to Qwen2.5-Math or to short training.
