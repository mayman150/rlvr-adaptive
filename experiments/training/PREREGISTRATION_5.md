# Pre-registration 5: is negative reinforcement a slower positive reinforcement?

Declared October 17, 2026, after the seed-1 result of Pre-registration 4 and before any
run of this design (see this file's commit time).

## Why

On-policy, the NSR update equals the PSR update minus sum_i grad log pi(y_i), a term with
expected value exactly zero (the advantage shift A -> A - 1). In seed 1 of
Pre-registration 4, NSR's per-prompt gradient noise was 6.4 times PSR's (the theorem
predicts about 7), NSR learned more slowly (first-block training reward 33.6% against
51.3% over steps 25 to 49), and at step 150 it had lower pass@1 (37.7 against 42.4) and
the same pass@64 (83.0 against 83.2). This suggests NSR is PSR slowed down by extra noise,
so its diversity at a given step reflects less progress rather than a different mechanism.

## Prediction

At matched pass@1, NSR and PSR have the same pass@32. Concretely, for each NSR evaluation
point, PSR's pass@32 at the same pass@1 is obtained by linear interpolation along PSR's
trajectory; the mean difference NSR minus PSR over NSR points inside PSR's pass@1 range is
below 2 points in absolute value in both seeds. The alternative (negative samples
preserve diversity beyond what slower progress explains) is a difference of at least 2
points in favor of NSR in both seeds.

## Runs

PSR and NSR, seeds 3 and 4, otherwise as Pre-registration 4 (strictly on-policy,
sequence sums, AdamW 1e-6, 16 prompts by 8 answers, 150 steps). pass@k with 32 samples on
the first 128 numeric MATH-500 problems at steps 25, 50, 75, 100, 150 (and step 0 in the
PSR runs); greedy accuracy at the same steps; on-policy entropy probe every 5 steps.
