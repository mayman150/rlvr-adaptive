# Pre-registration 26, interim: weight-direction diagnostic (Step 1a)

October 28, 2026. Job worth-p26-pack-s8-20261028-013859 (lane 4); raw output `results_p26/dirdiag_w.json`. The
gradient diagnostic (Step 1b), the par/perp/other evaluations and the four pilots are still running.

## Notation

- W_0: base weights (Qwen2.5-Math-1.5B, bf16); W_r: run r's step-150 checkpoint (saved in bf16);
  Delta_r = W_r - W_0, restricted to the 196 two-dimensional weight matrices other than the embeddings and lm_head.
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
| Rank-1 share (energy captured jointly by the per-matrix rank-1 approximations), all 11 runs | 0.047-0.056 (B1k8 0.055) |
| Update cosine, Dr. GRPO seed to seed (B1k8, B1k9, B1k10) | 0.341-0.343 |
| Update cosine, B1k8 vs methods with a different advantage rule (PSR1k8, NSR1k8, AC1k8) | 0.335-0.364 |
| Update cosine, B1k8 vs seed-8 Dr. GRPO variants (HG1k8, HGm1k8, PG1k8, MH8, AVH8) | 0.430-0.466 |
| Update cosine, all method pairs excluding B1k8 and the seed controls | 0.266-0.456 (median 0.357) |
| Top-direction alignment with B1k8 (all pairs: 0.760-0.908) | 0.821-0.908 (seeds 0.854-0.857) |
| Trajectory cosine of B1k8, steps 25 vs 150 / 100 vs 150 | 0.392 / 0.762 |
| Trajectory Gram eigenvalue shares | 0.785, 0.119, 0.067, 0.030 |
| bf16 retention of "par" (rank-1 part) | cosine 0.352, norm ratio 0.457 (construction error energy 0.89 of intended) |
| bf16 retention of "perp" (remainder) | cosine 0.975, norm ratio 1.016 (construction error energy 0.05 of intended) |

## Readings (as registered, with the amendment's qualifications)

- **Shared direction, weight space: the saved-weight cosine criterion was not met.** The registered support rule
  needed pairwise update cosine >= 0.5 and "par" recovering at least 80% of B1k8's pass@1 gain over "other"; the
  observed cosines are 0.27-0.47, so the rule fails on its first condition. A method's cumulative update overlaps with Dr. GRPO's about as much as a second Dr.
  GRPO seed does (0.34); seed-8 variants that share B1k8's data order and rollout seeds overlap somewhat more
  (0.43-0.47).
- **Rank-1 approximations jointly capture 4.7-5.6% of the total saved matrix-update energy**, so the updates are not
  rank-1 in aggregate (individual matrices were not examined), although the top singular directions are well aligned
  with B1k8's (0.82-0.91).
- **Trajectory:** the step-25 update is weakly aligned with the step-150 update (0.39); the top trajectory eigenvalue
  share is 0.79, so the trajectory is dominated by one direction but is not one-dimensional.
- **The intended "par" component is severely distorted by rounding** (cosine 0.35, norm ratio 0.46); its evaluation is
  reported with this caution (the registration set no fidelity cutoff). "perp" is preserved (cosine 0.975) and is
  assessed separately against B1k8's trajectory and temperature envelope at matched pass@1.

## Caveat (not registered; found here)

All checkpoints are stored in bf16 (training keeps fp32 weights; the saved copy is rounded). With learning rate 1e-6
and 150 steps, per-element cumulative updates may be comparable to the bf16 spacing of the weights (not yet measured).
If the rounding error of each saved Delta is independent across runs, mean-zero and diffuse, it attenuates the update
cosines and dilutes the rank-1 share; deterministic rounding need not satisfy these assumptions, so either measure
could move in either direction. Planned check (no training): per matrix and run, the fraction of elements with
Delta = 0, the fraction where W_r is an immediate bf16 neighbour of W_0 and the update energy they carry, and the
uniform-cell estimate of rounding-noise energy (sum of q_i^2 / 12, asymmetric cells at exponent boundaries) relative to
||Delta||^2. The gradient diagnostic avoids checkpoint differencing, but it measures a different object: the
conditional (mixed-group) problem-gradient structure within B1k8, in fp32 parameters with bf16 autocast.

# Step 1b: gradient diagnostic

Lane 5 of the same job; raw output `results_p26/dirdiag_g.json`. 32 problems mixed in both groups at each checkpoint
(58 and 75 problems tried at steps 50 and 150).

## Notation

- S_A, S_B: CountSketch (2^20) of each problem's Dr. GRPO surrogate gradient from group A or group B (8 answers each,
  T = 1, 1,024 tokens), over the same matrices as the weight diagnostic; parameters fp32, forward in bf16 autocast.
- One-group shares: eigenvalues of S_A S_A^T (or of the cosine matrix) over their sum. Cross-group shares: signed
  eigenvalues of (S_A S_B^T + S_B S_A^T) / 2 divided by the sum of their absolute values (not explained-variance
  fractions); negative share: the negative eigenvalues' part of that sum.
- Batch-mean direction share: the fraction of a group's total sketch energy projected onto that group's own empirical
  mean direction.
- Same-problem cross-group cosine: cosine between a problem's S_A row and its S_B row, averaged over problems without
  energy weighting.

## Results

| Quantity | Step 50 | Step 150 |
|---|---|---|
| One-group Gram top share (A / B) | 0.075 / 0.077 | 0.057 / 0.106 |
| One-group cosine-matrix top share (A / B) | 0.046 / 0.048 | 0.039 / 0.038 |
| Mean pairwise cosine between problems (A / B) | -0.0002 / 0.0006 | 0.0014 / 0.0001 |
| Batch-mean direction share (A / B) | 0.040 / 0.044 | 0.039 / 0.047 |
| Cross-group Gram top share; negative share | 0.169; 0.390 | 0.114; 0.373 |
| Same-problem cross-group cosine, mean (quartiles) | 0.011 (-0.022, 0.014, 0.035) | 0.014 (-0.013, 0.010, 0.035) |

## Readings

- **Whitening reading: inconclusive.** The amended rule uses the cross-group top share at step 150 and declares it not
  interpretable when the negative share exceeds 0.2; the negative share is 0.373. The amended diagnostic is therefore
  inconclusive about the directional concentration of expected gradients conditional on both groups being mixed. (The
  one-group shares are reported for description only; the amendment superseded the one-group threshold.)
- **Description:** the sampled sketches show low cross-group directional agreement (mean same-problem cosine 0.011 and
  0.014) and near-zero mean inter-problem cosine. These are not energy fractions: per-problem normalization and
  unweighted averaging prevent reading them as the reproducible share of gradient energy without further assumptions.
- **Joint reading with Step 1a:** Step 1a fails the saved-update support criterion; Step 1b leaves expected-gradient
  structure unresolved. The two measure different objects (cumulative optimizer updates from bf16 checkpoints vs.
  conditional stochastic surrogate-gradient sketches); their joint pattern is compatible with a weak persistent shared
  component obscured by sampling noise, without identifying its mechanism.
- **Implication for the whitening pilot (PW):** its reweighting may be sensitive to gradient-estimation noise; its
  effect remains to be tested by the registered pilot.
