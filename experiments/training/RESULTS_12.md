# Pre-registration 12: results (main runs, seeds 8, 9, 10)

Analysis: `python analyze_v12.py --branch M --candidate CM --audits M,U` (output in
`RESULTS_12_main_output.txt`), run October 21, 2026 once every registered evaluation was complete.
Evaluation: 511 held-out problems (answer keys audited), 32 samples, steps 50, 100, 150; Dr. GRPO
(B) also at T = 1.15 and 1.3 at step 150; greedy accuracy on 3,382 test problems at step 150.

## Registered verdicts

| Test | Result | Verdict |
|---|---|---|
| **Primary: CM against B's trajectory-and-temperature envelope** | +1.19, 90% [+0.63, +1.69], 95% [+0.54, +1.79]; per seed +1.52, +1.44, +0.59 | **average agreement** (below the +2 bar; 90% interval inside ±2) |
| Untouched-half confirmation (256 problems) | +1.62, 95% [+0.61, +2.52]; per seed +2.31, +1.50, +1.05 | positive, but moot: the primary did not reach "moves the curve" |
| Equal cost: step-150 pass@1, CM minus B | -2.05, 90% [-2.54, -1.55] | not shown (CM is about 2 points less accurate when sampled) |
| Audit: uniform memory (M) against B's trajectory | +0.61, 95% [-0.04, +1.27] | average agreement |
| Audit: unlikeliness reward (U) against B's trajectory | +0.14, 95% [-0.32, +0.62] | average agreement |
| Coverage weighting: CM against M's trajectory | +0.66, 95% [+0.18, +1.12]; per seed +1.13, +0.98, -0.13 | not shown (needs at least +1) |

## Accuracy at step 150 (mean of seeds 8, 9, 10)

| Arm | pass@1 (511) | pass@16 (511) | Greedy test accuracy (3,382) |
|---|---|---|---|
| Dr. GRPO (B) | 44.1 | 74.5 | 47.9 |
| Coverage memory (CM) | 42.1 | 75.2 | 47.7 (CM minus B: -0.18, 90% [-0.83, +0.43]) |
| Uniform memory (M) | 42.9 | 74.0 | 47.3 |
| Unlikeliness (U) | 43.4 | 74.0 | 47.6 |

## Reading

Coverage-weighted memory gives a small, consistent lift at matched accuracy (+1.2 points of pass@16,
positive in every seed, 95% interval above zero), larger than uniform replay (+0.6) and the published
unlikeliness reward (+0.1), and keeps greedy test accuracy. It does not reach the registered +2 bar,
so the registered verdict is **average agreement**, not "moves the curve". The pilot's +2.1 was
optimistic. Sampled pass@1 at step 150 is about 2 points lower than Dr. GRPO's, i.e. CM moves more
slowly along the curve at equal steps; greedy accuracy is unchanged.

The diagnosis predicts why the lift is small: a policy's own groups of 8 rarely find a success on
rarely solved questions, so the archive cannot reach the questions whose pass@k can still rise.
Pre-registration 14 (seeding the archive from base-model discovery) was declared before this result
was known; its references are these runs.
