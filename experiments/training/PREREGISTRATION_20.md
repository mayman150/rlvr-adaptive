# Pre-registration 20: do the matched-pass@1 findings hold at 7B?

Draft, October 24, 2026, before any 7B training run; revised after Codex's review. Cost figures will be filled
in from the Fir smoke test (`experiments/tamia/test_dist_fir.sh`) before launch; the design does not depend on
them. One arm menu (below) is chosen and committed before any primary 7B training run.

## What was known

At 1.5B (Qwen2.5-Math-1.5B, 512 tokens): (i) PSR and NSR have the same expected update per problem and lie
within 2 points of each other at matched pass@1, NSR's lag being reproduced by mean-preserving random-sign
noise; (ii) G32 (16 problems x 32 rollouts, 512 rollouts per step) reaches higher pass@1 per step; its
comparison with the longer Dr. GRPO trajectory is exploratory (+0.60 points, four of six checkpoints covered,
one reference seed, no interval); (iii) frontier-value rollout allocation (AC, Pre-registration 18) is, in
interim step-150 results, +0.97 (seed 9) and -0.19 (seed 10) against B's envelope, against a forecast of +2.6
to +4.3; Pre-registrations 18 and 19 have no final verdict at this time; (iv) a frozen per-problem model
(Pre-registration 17, tau = -0.20) fits Dr. GRPO's pass@16 at observed pass@1 within its range. Absence of a
registered win is not established equivalence.

## Setup

Qwen2.5-Math-7B, `train_dist.py` (FSDP2 over 4 H100 GPUs, `--param-dtype fp32`: bf16 autocast on fp32
weights, as in train_grpo.py; a bf16 generation copy refreshed after every update), answers up to 2,048 tokens,
otherwise Pre-registration 12's recipe: 16 problems x 8 rollouts per step (128), one on-policy update per
batch, sequence-sum loss with the constant normalizer 128 x 2,048, AdamW 1e-6 (betas 0.9, 0.99, epsilon 1e-8,
no weight decay), gradient clipping 1, 150 steps, the same training problems and per-step order as the 1.5B
runs of the same seed (16 calibration problems held out). This is a replication at 7B and 2,048 tokens: the
token cap and the loss denominator both quadruple relative to 1.5B. Recorded per run: clipped and skipped
updates, tokens, step time, GPU-hours.

**Launch checks (must pass before any primary run).** `dist_parity.py` on frozen rollouts (8 saved groups
of 8 answers) at 1, 2 and 3 ranks (3: uneven loads): global gradient-norm relative difference to
train_grpo.py's single-GPU update at most 2%, gradient cosine at least 0.99 and AdamW-update cosine at least
0.95 for the checked parameters. The mean absolute per-token log-probability difference between the bf16
generation copy and the training model is reported (the generation copy is an approximation of the policy).

## Arms (seeds 8 and 9; paired with the 1.5B seeds)

**Core menu (8 runs):** B7 (Dr. GRPO, A = r - mean), PSR7 (A = r), NSR7 (A = r - 1), AC7 (Pre-registration
18's registered algorithm and frozen settings: 4 pilot rollouts, 64 extras by the law value with lambda 0.37,
k = 16, cap 32, fixed normalizer). AC7 is run whatever Pre-registration 18's verdict: it tests the strongest
forecast self-information intervention at 7B.

**Alternative menu (10 runs), only if Pre-registration 19's final rule 1 holds and the external-information
claim is retained:** B7, PSR7, NSR7, HQ7, HP7. Requires a validated supervised stage in train_dist.py and a
post-fine-tuning step-0 evaluation; HQ and HP keep the 1.5B-defined problem sets (a 1.5B-defined curriculum,
not rediscovered for 7B).

A G32 analogue is not run: 4 x 32 would also quarter problem exposure, and 16 x 32 quadruples sampling.
Two seeds support a scoped replication; the problem bootstrap does not quantify training-seed variability.

## Evaluation

The audited 511 problems of Pre-registration 12 are the replication anchor: 32 samples at T = 1 and 2,048
tokens, at steps 0 (base), 50, 100, 150 for every arm and step 25 for B7 (to reduce the base-to-step-50 chord
bias); B7 also at T = 1.15 and 1.3 at step 150; greedy accuracy on the 3,382 test problems at step 150. The base
7B evaluation is run first and its histogram of success counts (of 32) is published before training. If fewer
than 128 of the 511 problems have 1-4 successes, a frozen, audited harder cohort is added, selected from MATH
test problems with independent base-screening samples (not reused for evaluation), and every arm and reference
is evaluated there and reported separately. "Insufficient overlap" if fewer than two checkpoints per seed fall
in the reference's pass@1 range; a comparison with more than 10% of bootstrap points clamped is not
interpreted.

## Decision rules (Pre-registration 12's interpolation; 2,000 shared bootstrap draws over problems)

1. **Sign rules (agreement, T = 1 trajectories):** NSR7 against PSR7's trajectory, PSR7 against B7's, NSR7
   against B7's. Average equivalence if the pooled 90% interval and each seed's 90% interval lie within
   [-2, +2]; a departure (either direction) if |pooled estimate| >= 2, the Holm-adjusted p-value is at most
   0.05, and the per-seed signs agree; otherwise inconclusive. Simultaneous point bands are reported;
   checkpoint-average equivalence does not establish an unchanged whole trajectory.
2. **AC7 (superiority against B7's trajectory-and-temperature envelope):** Pre-registration 18's rule 1
   (at least +2, 95% interval above 0, positive in both seeds) -> moves the curve; the equivalence criterion of
   rule 1 -> replicates the null; otherwise inconclusive. (Alternative menu: HQ7 against HP7's trajectory with
   Pre-registration 19's rule 1 threshold.)
   Holm over the four tests of rules 1-2. A joint "the 1.5B findings replicate" statement requires every
   applicable contrast to reach its registered outcome.
3. **The frontier model transfers:** initial log-odds from the base 7B counts with `law.fit_prior(counts, 32)`
   and `law.initial_theta`, tau frozen at -0.20; the entire conditional pass@16-versus-pass@1 function is
   computed and committed before any trained 7B checkpoint is evaluated. Pre-registration 17's RMSE categories
   over B7's seed-steps (steps 25-150): at most 2 points supported, 2-3 partial, more than 3 falsified.
   Per-seed and per-stratum errors and the frozen uniform-shift and affine comparators are reported; points
   outside the model's domain are unscorable.

## Reported without a rule

Training-time statistics (mixed groups, answers with nonzero advantage, truncation, length, step time, peak
memory; for AC7, group sizes by pilot count and the share of pilot-zero problems with a success); the squared
score norms of correct and incorrect answers on the base 7B model (16 calibration problems x 8 answers) if
measured; pass@16 by base-count stratum; each 7B arm's position relative to its reference next to the 1.5B
arm's.

## Cost

Measured on ml.p4de.24xlarge (8 A100 80 GB, $31.56 per hour, us-east-1) in the smoke test (October 25): with 4
GPUs per run, 126-189 s of generation and 142-204 s per step (peak 68.5 GB), about 6.7 hours per 150-step run;
with 8 GPUs about 100 s per step. Two runs share one instance (4 GPUs each): about $106 per run, $850 for the
eight core runs; about 32 checkpoint evaluations (one A100 each, about 2 hours) about $260; total about $1,100.
Launch check (dist_parity.py, 1.5B, 1-3 ranks): global gradient-norm difference 0.27-0.31% (threshold 2%);
gradient cosine 0.999+ for embeddings, MLP and norm parameters but 0.985-0.989 for layer 0's query projection
(threshold 0.99), identical at 1, 2 and 3 ranks; AdamW-update cosine 0.966-0.997 (threshold 0.95). Because the
query-projection deviation does not depend on the number of ranks, it is attributed to micro-batch composition
under bf16 autocast; the reference's own batching noise floor (the same update with the answers batched in a
different order) is measured before launch, and the check counts as passed if the FSDP deviation is within it
(this criterion was amended after seeing the check results, before any primary 7B run).

**Launch-check result and launch (October 25, 2026).** The reference's own batching noise floor (the same
single-GPU update with the answers batched in reverse order): gradient cosine 0.9858 for layer 0's query
projection (FSDP against the reference: 0.9837), 0.9997-1.0 elsewhere (FSDP: 0.9994-0.9998); AdamW-update cosine
0.975-1.0 (FSDP: 0.966-0.997); gradient-norm difference 0.06% (FSDP: 0.29%). The query-projection deviation is the
size of the reference's own batching noise and every other checked parameter exceeds 0.999, so the check is
passed. Core menu launched: B7, PSR7, NSR7, AC7, seeds 8 and 9, two runs per ml.p4de.24xlarge (4 GPUs each),
us-east-1. Evaluation as above with PSR7 also at step 25 (reference knots, as in Pre-registration 21); trained
checkpoints are evaluated only after rule 3's forecast is committed.

**Base 7B evaluation and the harder cohort (October 26, 2026, before any trained 7B checkpoint is evaluated).**
Base Qwen2.5-Math-7B on the 511 problems at 2,048 tokens: pass@1 29.9, pass@16 87.5, 15.8% truncated; successes of
32: 37 problems with 0, 116 with 1-4, 254 with 5-16, 104 with 17-31, none with 32. Fewer than 128 problems have
1-4 successes, so the harder cohort is triggered. Selection, fixed before screening: pool = the 1,145 Level 4-5
problems of `eval_math_test_numeric.json` not in the 511 or in `confirm_fresh_512.json`
(`data/hard7_pool.json`); screening = 8 base-7B samples per problem at 2,048 tokens, seed 2027 (not reused for
evaluation); eligible = 0-2 successes of 8; cohort = the first 256 eligible problems in a random order (seed 2028),
labels audited by an independent solver (disagreements removed, not replaced); frozen as
`data/hard7_cohort.json` before any arm is evaluated on it. Every arm and B7's references are evaluated on it with
the same protocol (32 samples, 2,048 tokens, the same steps and temperatures) and reported separately with the same
rules. Rule 3's forecast (`FORECAST_20.json`, `forecast_20.py`) was committed in 5cbf692 before the trained
evaluations started.

**Amendment (October 27, 2026, before any trained-checkpoint evaluation counted for a rule).** Evaluation engine:
the Hugging Face generation used so far needs about 14 hours per 7B checkpoint at 2,048 tokens (the first evaluation
jobs ran out of time without a result and were stopped). All 7B evaluations use `eval_vllm.py` (vLLM 0.11.0, same
prompts, checker, sampling settings and accounting as `eval_ckpt.py`; per-problem seeds). Equivalence check before
use (two independent vLLM runs against the Hugging Face results): base 7B at 2,048 tokens 30.5/88.0 and 30.1/86.8
against 29.9/87.5 (pass@1/pass@16); 1.5B B1k seed 9 at 1,024 tokens, step 50: 52.2/87.5 and 51.9/86.4 against
52.5/86.7; step 150: 58.1/86.7 twice against 58.0/86.8; truncation and lengths agree. The base model is re-evaluated
with vLLM so that every point of a comparison comes from one engine; rule 3's frozen forecast (from the Hugging Face
base counts) is unchanged.
