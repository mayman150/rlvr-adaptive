# Pre-registration 2: the optimal collection rule

Declared October 16, 2026, before any run of these arms existed (see this file's commit
time), after the shape measurement and before the pilot's final verdict. Requested by the
author ("okay do so").

## Prediction

With the measured inputs (gradient signal fading as p^2 (1-p)^2, F0 / F1 from calibration,
noise ratio R = 130 from the MLP block), the optimal reward-adapted rule restricted to
decisions every 4 answers with a cap of 16 has 1.69 to 2.27 times the training signal per
generated answer of Dr. GRPO with groups of 8, and 1.33 to 1.42 times that of SARA-style
probe-then-fill, across difficulty distributions.

## Arms

All arms share the model (Qwen2.5-Math-1.5B base), data, optimizer (AdamW, 1e-6), clipping,
loss normalizer (128 answers by 512 tokens), 150 steps, evaluation, and seeds 1 and 2.

- **baseline:** Dr. GRPO, 16 prompts by 8 answers per step; reused from the pilot runs
  `worth-train-s1-20261016-213109` and `worth-train-s2-20261016-230523`, same seeds and
  same baseline code path.
- **optimal:** answers in blocks of 4 per prompt; after each block, stop or continue by the
  optimal lookup table; on stopping, every success gets advantage a and every failure b from
  the table (scaled so the largest magnitude is 1). The table is re-solved every 10 steps
  from a beta-binomial fit to the last 200 prompts (calibration groups at the start), with
  F0 / F1 from calibration.
- **probe:** a block of 4; abandon the prompt if all agree, otherwise fill to 16 and use
  mean-centered advantages.

Variable-size arms choose the number of prompts each step so that cumulative answers track
128 per step; total answers over 150 steps must be within 5% of 19,200, and tokens are
reported.

## Endpoints and decision

- **Primary:** greedy accuracy on the 364 numeric MATH-500 problems after 150 steps,
  optimal minus baseline, both seeds.
- **Secondary:** optimal minus probe on the same endpoint; on-policy training reward over
  the last 50 steps; accuracy at steps 50 and 100; generated tokens.

**Promising** if optimal exceeds baseline by at least 2 points in both seeds, or its
late training reward is higher in both seeds with a primary difference that is not
negative. **No evidence** otherwise. Two seeds justify a larger study, not a claim.

## Known limits

The rule is optimal for one-step progress on mean accuracy under the declared model;
exploration, Adam's preconditioning, and the change of difficulty during training are
outside the theory, and the table is re-solved to track the last of these. Training
reward is not directly comparable across arms, because they spend their answers on
different prompts.
