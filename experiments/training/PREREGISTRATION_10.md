# Pre-registration 10: does the size of NSR's noise account for its slower learning?

Declared October 18, 2026, before any run of this design (see this file's commit time).
Revised twice before launch after independent reviews (Codex), which asked for narrower
causal verdicts, a same-policy manipulation check, pooled interval-based rules with
step pairing, a fair learning-rate comparison, a third fresh seed, fresh-only primary
comparisons, correct pairing of re-measured gradient sizes, labels that say what the Adam
statistics measure, and an average-equivalence label with simultaneous bands; the second
asked for narrower wording of the positive verdict, gates on every speed verdict,
per-step interpolation of the re-measured F's, interval-based asymmetry verdicts, and a
correct label for the Adam alignment statistic. The earlier drafts were never run.

## Why

An ICML-style review (Codex, October 18) accepted that noise *predicts* learning speed
(Pre-registration 7: five constant baselines with the same expected update, speed ordered by
predicted noise, rank correlation 0.9 to 0.97), but not that noise *accounts for* NSR's
slower learning. In the baseline family, raising the baseline raises the noise and pushes
the sampled wrong answers down harder at the same time.

## The intervention

**PSR+N.** Each answer gets advantage r_i + c_t s_i, with s_i an independent random sign
(+1 or -1 with probability 1/2, from a generator seeded by run seed and step). Given the
sampled answers, the added term averages to zero over the signs whatever c_t is, so the
expected raw (unclipped) gradient is exactly PSR's and NSR's, n grad p. c_t is set at each
step so that the theorem's diagonal second moment equals NSR's on that step's reward counts
c_q (out of G = 8 per question):

c_t^2 = sum_q [(G - c_q) F0 - c_q F1] / sum_q [c_q F1 + (G - c_q) F0]  (0 if negative),

with F0 and F1 from the run's calibration before training. PSR+N has NSR's predicted noise
size at its own policy, but no systematic push on wrong answers.

What it matches and what it does not: the trace of the per-answer second moment (diagonal
terms), using global, initial F's. It does not match reward-conditional cross terms, the
covariance geometry of the noise, or how Adam's per-coordinate scaling reacts to it. The
manipulation check below measures how well the match holds at the same policy, and the
Adam statistics are reported.

**What the outcomes can and cannot say.** If PSR+N learns as slowly as NSR, trace-matched
sign noise reproduces NSR's early accuracy lag: the noise size suffices to produce it here,
though geometry and Adam's response remain possible routes. If PSR+N learns faster, the
size alone is not sufficient; that would not identify the cause (the systematic push on
wrong answers, noise geometry, and Adam's adaptation remain candidates).

## Setup

Qwen2.5-Math-1.5B base, as Pre-registrations 4 to 8: 3,920 numeric MATH training problems in
four subjects, AdamW 1e-6 (2e-6 in the learning-rate arms), betas (0.9, 0.99), no weight
decay, 16 questions by 8 answers per step, strictly on-policy, sequence-sum loss, gradient
clipping at norm 1 (never active in the earlier on-policy runs; maximum norm 0.48), answers
up to 512 tokens, 150 steps, calibration on 64 questions (512 answers). **Fresh seeds 5, 6
and 7.** All arms of a seed see the same questions at every step (the question order depends
only on the seed), so comparisons are paired by step.

Arms:
- seeds 5, 6, 7: PSR, NSR, PSR+N, NSR×2 (NSR at learning rate 2e-6);
- seeds 5, 6: PSR×2 (PSR at 2e-6).
Each seed's four main arms run as four parallel processes on one 4-GPU instance; PSR×2 runs
alone.

## Measurements

- Every step: first-block training accuracy (the first 4 answers of each of the 16 questions;
  unbiased for every arm), split-half gradient statistics, gradient norm (clipping
  frequency), boxed-answer and truncation rates; entropy probe every 5 steps; c_t in PSR+N.
- **Same-policy shadow (PSR+N, steps 0 to 49):** split-half statistics of the NSR estimator
  computed on PSR+N's own answers and halves at the same policy (not used for the update).
- **Adam statistics (reported, no rule):** (i) the split-half statistics of the gradient
  divided by Adam's per-coordinate scale from before the step (D_{t-1} g_t; this is not
  Adam's update); (ii) lagged Adam direction alignment: the previous step's actual update direction
  u_{t-1} = m_hat / (sqrt(v_hat) + eps) projected on this step's population-gradient
  reference (fresh questions given the history), <u_{t-1}, ref_t>, also times the learning
  rate, and |u_{t-1}|^2.
- F0 and F1 re-measured at steps 50, 100 and 150 on the same 64 calibration questions with the
  same sampling seed as before training (paired).
- Greedy accuracy on the 364 numeric MATH-500 problems at steps 0, 25, 50, 75, 100, 150.
- pass@k from 32 samples on the first 128 of those problems (the same 128 as Pre-registrations
  5 to 8) at steps 25, 50, 75, 100, 150, and step 0 in the PSR run of each seed.
- bf16 checkpoints at steps 50, 100, 150 (for later evaluations; not used below).

## Predictions and decision rules

**Speed measure.** A_X is run X's mean first-block training accuracy over steps 25 to 49.
For an arm X, the pooled position between NSR and PSR over the fresh seeds S = {5, 6, 7} is
phi(X) = sum_s (A_X,s - A_NSR,s) / sum_s (A_PSR,s - A_NSR,s): 0 is NSR's speed, 1 is PSR's.
**Uncertainty:** 2,000 bootstrap draws; within each seed, steps are resampled in blocks of 5
consecutive steps, with the same step indices for every arm of that seed (keeping the
pairing by shared questions). This covers question sampling within runs, not seed-to-seed
variation; per-seed phi with intervals and the spread across seeds are reported. If the
pooled gap sum_s (A_PSR,s - A_NSR,s) / 3 is below 5 points, every speed verdict is
**inconclusive** (no seed is dropped). A power check with this procedure on the seed-3 and seed-4 runs of
Pre-registrations 5 and 7 (PSR, NSR and the b = 0.25, 0.5, 0.75 baselines in place of PSR+N)
gave pooled 90% half-widths of 0.09 to 0.12 for two seeds; step pairing matters (training
accuracies of different arms at the same step correlate at 0.6 to 0.8).

**Margin.** 1/3 of the PSR-NSR gap. In Pre-registration 7, the b = 0.75 baseline, with a
predicted noise 2.4 times smaller than NSR's, sat at phi = 0.34 (pooled); a margin of 1/3
asks PSR+N to be no further from NSR than that.

1. **Manipulation check.** Ratio of PSR+N's measured per-question second moment to the
   same-policy NSR shadow's, steps 0 to 49, pooled over seeds (bootstrap over steps as
   above): the 90% interval must lie within [1/1.3, 1.3]. Otherwise the primary verdict is
   **manipulation failed** (results still reported). Also reported: the ratio against the
   separate NSR run, centered variances (second moment minus signal), the same ratios after
   Adam scaling, and the share of steps with c_t = 0 or with clipping.
2. **Primary.** phi(PSR+N):
   - 90% interval inside [-1/3, 1/3]: **trace-matched sign noise reproduces NSR's early
     accuracy lag**;
   - 95% interval above 1/3: **it does not** (PSR+N is clearly faster than NSR);
   - 95% interval below -1/3: **PSR+N slower than NSR** (NSR is faster than its noise size
     predicts);
   - otherwise **inconclusive**. Heterogeneity is flagged if the per-seed estimates fall in
     different classes (below -1/3, within ±1/3, above 1/3).
3. **Learning rate (tested rates only; the same gates: all planned seeds and the 5-point
   gap).** (a) phi(NSR×2), with the same procedure: 95%
   interval above 0.5: **doubling the learning rate recovers at least half of NSR's lag**;
   95% interval below 0.5: **not recovered by doubling**; otherwise inconclusive. (b) At
   matched learning rate 2e-6 (seeds 5 and 6): the gap A_PSR×2 - A_NSR×2 is reported with its
   interval, and as a fraction of the 1e-6 gap in the same seeds; **NSR still lags at 2e-6**
   if the gap's 95% interval is above 0.
4. **Average matched-accuracy equivalence.** pass@16 (from 32 samples) at matched pass@1,
   Pre-registration 9's procedure: at each checkpoint after step 0 of a run R whose pass@1 lies
   inside the PSR reference's pass@1 range, PSR's pass@16 is interpolated linearly at R's pass@1
   along PSR's trajectory (base model, then steps 25 to 150, ordered by pass@1, ties broken by
   step; a resampled point outside the range takes the end value); qualifying checkpoints are
   fixed from the full data; the run's value is the mean over them; seeds are averaged;
   2,000 bootstrap draws over the 128 problems, the same draw for every run and seed of a
   comparison. **Average equivalence** if the 90% interval lies inside [-2, +2]; **different**
   if the 95% interval excludes 0 and the estimate is at least 2 points from 0; otherwise
   inconclusive; incomplete evaluation and insufficient overlap as in Pre-registration 9. A
   signed average can hide crossings, so a simultaneous 90% band over all qualifying
   checkpoints (bootstrap of the maximum standardized deviation) is reported, and the
   comparison is labeled **no checkpoint outside ±3** if every band lies inside [-3, +3].
   Comparisons, all against PSR of the same seeds:
   - **primary:** PSR+N and NSR, fresh seeds 5, 6, 7;
   - NSR×2, seeds 5, 6, 7 (PSR×2 is its matched-rate reference, reported);
   - **secondary, historical plus fresh:** NSR, seeds 3 to 7, with seeds 3 and 4 from
     Pre-registration 5 (pinned runs: worth-v5-PSR-s3-20261017-164938,
     worth-v5-NSR-s3-20261017-164942, worth-v5-PSR-s4-20261018-034043,
     worth-v5-NSR-s4-20261017-164950; same model, data, 128 problems and settings, except
     calibration on 16 questions, which changes which questions train and their order, and no
     diagnostics); also reported separately by protocol (3 and 4; 5 to 7). The interval is
     conditional on the trained runs and does not include seed variation.
5. **Asymmetry over training.** F0 / F1 at every measurement (0, 50, 100, 150) in every run,
   with 95% intervals from bootstrapping the 64 questions: **asymmetry persists** if every
   interval lies above 1.2; **fails** if some interval lies below 1.2; a measurement with no
   correct or no wrong answers makes the verdict **incomplete**; otherwise inconclusive.

Reported without a rule: the noise formula's prediction against the measured second moment
in steps 0-49, 50-99 and 100-149, with the calibration F's and with F0 and F1 interpolated
linearly at each step between the measurements at the window's ends (0 and 50, 50 and 100,
100 and 150) before weighting by that step's counts;
entropy; formatting (flagged if boxed-answer or truncation rates over steps 100 to 149
differ from PSR's by more than 10 points).

## Scope

This tests whether the size of the noise, as the theorem measures it, accounts for NSR's
speed in one model at the tested learning rates. It does not test noise geometry beyond the
second-moment trace, other optimizers, or larger models. Fourteen runs, about 7 to 8 hours
each, on three 4-GPU instances and two single-GPU instances.
