# Pre-registration 26, interim: weight-direction diagnostic (Step 1a)

October 28, 2026. Job worth-p26-pack-s8-20261028-013859 (lane 4); raw output `results_p26/dirdiag_w.json`. The
gradient diagnostic (Step 1b), the par/perp/other evaluations and the four pilots are still running.

## Notation

- W_0: base weights (Qwen2.5-Math-1.5B, bf16); W_r: run r's step-150 checkpoint (saved in bf16);
  Delta_r = W_r - W_0, restricted to the 196 two-dimensional non-embedding weight matrices.
- Rank-1 share: sum over matrices of sigma_1^2 divided by sum over matrices of ||Delta||_F^2.
- Update cosine: <Delta_a, Delta_b> / (||Delta_a|| ||Delta_b||) over all 196 matrices (flattened).
- Top-direction alignment: energy-weighted mean over matrices of |<u_1^a, u_1^b>| |<v_1^a, v_1^b>|.
- Trajectory Gram: Gram matrix of B1k8's flattened updates at steps 25, 50, 100, 150; shares are its eigenvalues over
  their sum (RELEX's trajectory-rank measure).
- bf16 retention: cosine and norm ratio between the realized update of a constructed checkpoint (after rounding
  W_0 + intended part to bf16) and the intended part, aggregated over matrices.

## Results

| Quantity | Value |
|---|---|
| Rank-1 share, all 11 runs | 0.047-0.056 (B1k8 0.055) |
| Update cosine, Dr. GRPO seed to seed (B1k8, B1k9, B1k10) | 0.341-0.343 |
| Update cosine, B1k8 vs methods with a different advantage rule (PSR1k8, NSR1k8, AC1k8) | 0.335-0.364 |
| Update cosine, B1k8 vs seed-8 Dr. GRPO variants (HG1k8, HGm1k8, PG1k8, MH8, AVH8) | 0.430-0.466 |
| Update cosine, all method pairs excluding B1k8 and the seed controls | 0.266-0.456 (median 0.358) |
| Top-direction alignment with B1k8 | 0.821-0.908 (seeds 0.854-0.857) |
| Trajectory cosine of B1k8, steps 25 vs 150 / 100 vs 150 | 0.392 / 0.762 |
| Trajectory Gram eigenvalue shares | 0.785, 0.119, 0.067, 0.030 |
| bf16 retention of "par" (rank-1 part) | cosine 0.352, norm ratio 0.457 |
| bf16 retention of "perp" (remainder) | cosine 0.975, norm ratio 1.016 |

## Readings (as registered, with the amendment's qualifications)

- **Shared direction, weight space: not supported.** The registered threshold was pairwise update cosine >= 0.5; the
  observed cosines are 0.27-0.47. A method's cumulative update overlaps with Dr. GRPO's about as much as a second Dr.
  GRPO seed does (0.34); seed-8 variants that share B1k8's data order and rollout seeds overlap somewhat more
  (0.43-0.47).
- **Per-matrix updates are not rank-1** in this regime (top singular direction carries about 5% of the energy),
  although the top singular directions are well aligned across runs (0.82-0.91).
- **Trajectory:** the step-25 update is weakly aligned with the step-150 update (0.39); the top trajectory eigenvalue
  share is 0.79, so the trajectory is dominated by one direction but is not one-dimensional.
- **The par/perp test is not interpretable:** bf16 rounding preserves only part of the rank-1 component (cosine 0.35,
  norm ratio 0.46), so "par" does not represent W_0 plus the rank-1 part. Its evaluation is reported but not read.

## Caveat (not registered; found here)

All checkpoints are stored in bf16. With learning rate 1e-6 and 150 steps, many per-element cumulative updates are
comparable to the bf16 spacing of the weights, so each Delta contains rounding error that is roughly independent
across runs. This likely attenuates the update cosines toward 0 and the rank-1 share toward that of an unstructured
matrix; the size of the attenuation was not measured. The gradient diagnostic (fp32, in memory) does not have this
problem and is the cleaner test of a shared direction.
