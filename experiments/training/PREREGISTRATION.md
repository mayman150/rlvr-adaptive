# Pre-registration: scaled failure advantages versus Dr. GRPO

Declared October 16, 2026, before any training result existed (see this file's commit
time). Requested by the author ("start the experiment").

## Prediction being tested

The two-moment theorem and the measured gradient asymmetry (wrong answers carry 2.8 to
3.5 times larger squared score norms than correct ones in the last-layer blocks) predict
that scaling failure advantages by w = F1 / F0 raises the training signal per generated
answer. In the exact model with F0 = 4 F1, this alone raises a fixed group of 8 from 0.38
to 0.63 of the optimal rule's efficiency.

## Design

Qwen2.5-Math-1.5B (base), numeric-answer MATH training problems (algebra, prealgebra,
number theory, counting and probability; 3920 problems), on-policy updates with one
gradient step per batch, 16 prompts by 8 answers per step, 150 steps, 512-token cap,
AdamW at learning rate 1e-6, gradient-norm clipping at 1, sequence-sum loss with a constant
normalizer. Arms differ only in advantages:

- **baseline:** A = r - mean(r) within each group (Dr. GRPO);
- **scaled:** the same, with failure advantages multiplied by w.

w = F1 / F0 is measured once per run on the starting model over all parameters, from 16
held-out training prompts with 8 answers each, and clipped to [0.1, 1]. Both arms see the
same prompt order and generate the same number of answers. Two seeds, each running both
arms on the same instance type (one L40S).

## Endpoints

- **Primary:** greedy accuracy on the 364 numeric-answer MATH-500 problems after 150
  steps, scaled minus baseline, averaged over seeds.
- **Secondary:** on-policy training reward averaged over the last 50 steps; accuracy at
  steps 0, 50, and 100.

## Decision rule

This is a pilot with two seeds. It counts as **promising** if the scaled arm's primary
endpoint exceeds the baseline's by at least 2 points in both seeds, or its secondary
endpoint is higher in both seeds with a primary difference that is not negative. It
counts as **no evidence** otherwise. A promising result justifies a larger, powered study;
it is not by itself a claim of improvement.

## Known limits

Greedy decoding on the base model often loops before training, so early accuracy
partly measures format learning; both arms share this. The efficiency prediction is for
one step of progress on mean accuracy; Adam's preconditioning and exploration effects are
outside the theory.
