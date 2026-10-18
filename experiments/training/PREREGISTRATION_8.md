# Pre-registration 8: does changing the direction move the accuracy-diversity frontier?

Declared October 17, 2026, before any run of this design (see this file's commit time).

## Why

Pre-registrations 4 to 7 found that rules sharing the same expected update (PSR, NSR, and
baselines in between, also under the NSR paper's implementation) differ only in noise, and
that noise moves training along a fixed accuracy-diversity frontier (pass@32 at a given
pass@1) without moving the frontier. The principle that follows: to change *what* the model
learns, change the expected update; to change *how fast*, change the noise. This tests the
first half.

## Arms

- **Dr. GRPO:** expected update 7 grad p per question (group of 8), the pass@1 direction.
- **pass@4 objective:** per-answer weight from the unbiased pass@k estimator (k = 4, group
  of 8) minus a leave-one-out baseline that does not depend on the answer. Verified exactly
  over all 256 reward patterns: expected update 2 * 4 (1-p)^3 grad p, the gradient of pass@4,
  so it pushes hard questions and leaves questions the model already solves alone.

Seeds 3 and 4, otherwise the protocol of Pre-registration 5 (strictly on-policy, AdamW 1e-6,
16 questions by 8 answers, 150 steps; pass@32 on the first 128 numeric MATH-500 problems at
steps 0, 25, 50, 75, 100, 150; greedy accuracy at the same steps; entropy probe every 5
steps). Answer texts of every pass@k evaluation and the final models are saved, for
CoT-Pass@k grading.

## Prediction and decision

For each pass@4 checkpoint, Dr. GRPO's pass@32 at the same pass@1 is interpolated along
Dr. GRPO's trajectory (rule of Pre-registration 5); the mean difference is taken over the
pass@4 points inside Dr. GRPO's pass@1 range.

- **Frontier moved** if the mean difference is at least +2 points in both seeds.
- **Frontier fixed** if its absolute value is below 2 points in both seeds.
- Otherwise inconclusive.

Secondary (reported, not decisive): pass@1, pass@32, and pass@k curves at step 150;
CoT-Pass@k from the saved texts, graded by Claude Haiku 5.5 (3 judgments per correct answer).
