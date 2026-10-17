# Pre-registration 7: same mean, graded noise (baseline dose-response)

Declared October 17, 2026, before any run of this design (see this file's commit time).

## Idea being tested

Advantages A = r - b with a constant b have, on-policy, exactly the same expected update
n grad p for every b (b = 0 is PSR, b = 1 is NSR), and per-prompt noise
n [(1-b)^2 p F1 + b^2 (1-p) F0]. With the measured F0 / F1 of about 2.5 and p near 0.3, the
predicted noise relative to b = 0 is 0.93, 1.7, 3.4 and 6.0 for b = 0.25, 0.5, 0.75, 1, with
a minimum near b = 0.15. If what separates PSR and NSR is only noise, every outcome should
follow this ordering.

Two readings of "only noise" are distinguished:
- **slower clock:** extra noise slows progress (Adam divides by a noise estimate), and runs
  trace the same accuracy-diversity frontier at different speeds;
- **exploration:** extra noise adds diversity beyond what slower progress explains.

## Runs

Seeds 3 and 4 (paired with PSR and NSR of Pre-registration 5, same protocol): shift arm with
b = 0.25, 0.5, 0.75; and PSR with learning rate 4e-7 (1e-6 divided by 2.5, about the square
root of NSR's measured noise ratio), the slower-clock control. Split-half diagnostics on.
pass@32 on the first 128 numeric MATH-500 problems at steps 25, 50, 75, 100, 150; greedy
accuracy at 0, 25, 50, 75, 100, 150; entropy probe every 5 steps.

## Predictions

1. **Noise.** Measured per-prompt second moment / theorem within 30% for every shift run
   (steps 0 to 49).
2. **Dose-response.** Steps until the first-block training reward (10-step mean) first
   reaches 0.45 increase with the theorem's predicted noise: Spearman correlation at least
   0.8 across b in {0, 0.25, 0.5, 0.75, 1}, pooled over seeds 3 and 4.
3. **Mechanism.** For each run, pass@32 at matched pass@1 is compared with PSR's (b = 0)
   trajectory by linear interpolation (Pre-registration 5 rule).
   - **Slower clock** if every |difference| is below 2 points (mean over the run's points
     inside PSR's pass@1 range, both seeds), and PSR at 4e-7 also lies within 2 points of
     NSR's frontier.
   - **Exploration** if the b = 0.75 and b = 1 runs lie at least 2 points above PSR's
     frontier in both seeds.
   - Otherwise inconclusive.
