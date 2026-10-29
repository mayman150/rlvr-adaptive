# Pre-registration 33: sensitivity of Pre-registration 28's 7B readings (dense early reference, hard cohort)

Declared October 29, 2026, before any run of this design; reviewed by Codex before launch. Codex's review of the
draft paper (paper/icml/codex_v5_out.md) asked for two sensitivity checks where the 7B conclusion is weakest.

## Why

1. **Sparse early reference.** Our Dr. GRPO 7B reference (B7 seeds 8, 9; 2,048 tokens) has knots at the base and steps
   25, 50, 100, 150, with pass@1 rising from 30.5 at the base to 61.6-63.0 at step 25. Six of Pre-registration 28's
   interpretable 7B public checkpoints lie at pass@1 46-63, where the reference is a straight chord between two
   distant knots. If pass@16 along the trajectory is concave in pass@1 there (likely, not guaranteed), the chord
   understates the reference, which would bias those checkpoints' gains upward or hide a departure downward.
2. **Saturation.** pass@16 on the 511 test problems is 94-95 for B7's later checkpoints, which compresses differences. Pre-registration 20's
   frozen hard cohort (253 Level 4-5 MATH test problems, data/hard7_cohort.json) is less saturated; the public
   checkpoints were never evaluated on it.

## Runs (one 8 x A100 job)

- **B7e (dense early reference):** Dr. GRPO 7B with B7's exact recipe and seeds (8 and 9; train_dist.py, 16 x 8,
  lr 1e-6, fp32 parameters, 2,048 tokens), 25 steps, checkpoints at steps 3, 6, 12, 25. GPU nondeterminism makes
  these runs replicates of B7, not copies; step 25 is the reproducibility check against B7's step 25.
- **Evaluations:** B7e's eight checkpoints on the 511 test problems and on the hard cohort; the 14 public 7B
  checkpoints of Pre-registration 28 on the hard cohort (same revisions, tokenizers and identities). All with 32
  samples, T = 1, 2,048 tokens, vLLM (Pre-registration 20's protocol). Existing evaluations are reused: B7 and the
  base on both problem sets (base, steps 25-150, step-150 temperatures).

## Analyses (Pre-registration 28's estimator, guards and verdict rules unless stated)

- **Reproducibility:** per seed, B7e step 25 minus B7 step 25 in pass@1 and pass@16 on both problem sets, with 90%
  problem-bootstrap intervals (reported; large differences qualify the pooled reference below).
- **Dense reference (primary, frozen now):** per seed, the composite trajectory base -> B7e steps 3, 6, 12, 25 ->
  B7 steps 50, 100, 150, sorted by pass@1 (ties by step), with B7's step-150 temperature envelope; the reference value
  is the mean over the seeds whose envelope covers the pass@1 (Pre-registration 28). It is a composite sensitivity
  reference built from two replicate runs per seed, not one realized trajectory. Sensitivity: base -> B7e 3, 6, 12 ->
  B7 25-150; also B7e's own early curve (base -> B7e 3-25) wherever it covers a checkpoint. A reproducibility
  difference larger than 3 points of pass@1 or 1 point of pass@16 at step 25 in either seed is flagged, and the
  primary and sensitivity readings are then given equal weight in the text.
- **Three primary families (Holm within each; all 14 public 7B checkpoints with Pre-registration 28's eligibility and
  protocol-failure guards):** (A1) 511 problems against the dense reference; (A2) the hard cohort against the dense
  reference built from the hard-cohort evaluations; (B2) the hard cohort against the frozen law (tau = -0.20) from the
  base's 32-sample hard-cohort counts (results/worth-v20v-B7-*/BASE7hard; prior and initial state recomputed in every
  draw; strict reachability). Bootstrap: 2,000 problem draws per cohort, shared by all comparisons on that cohort.
  Equivalence is pointwise and unadjusted, as in Pre-registration 28. Verdicts as in Pre-registration 28 (departure if |gain| >= 2
  and Holm-adjusted p <= 0.05; equivalence if the 90% interval lies within +/-2; else inconclusive; more than 10% of
  draws outside support: not interpreted; truncation above 90%: protocol failure).
- **Reported for every family:** gains under the original sparse and the dense references (on both cohorts), paired
  gain-change intervals, support/drop rates, Holm-adjusted p-values, truncation and boxed rates, verdict tallies by
  training-data label, and Pre-registration 28's paired RePO-versus-GRPO (DeepMath) report on the hard cohort (one
  checkpoint per model; the paired difference of the two gains with shared bootstrap draws).
- **Reading.** "No upward departure" is a scoped robustness statement: it can coexist with inconclusive, unsupported
  or failed comparisons and does not establish equivalence. Any upward departure is reported as such, with its
  training-data label. These analyses can qualify, not overturn by themselves, the 1.5B results.
- Diagnostic (not a verdict): pass@2 at matched pass@1 (spread of per-problem success) under both references.

## Cost (estimate)

Training two 7B runs for 25 steps on four GPUs each (about 70 minutes), then 30 evaluations spread over the eight
GPUs (about 90 minutes): about 2.7 hours of one machine, about $85.
