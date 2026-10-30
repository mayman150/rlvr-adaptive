# Pre-registration 36: a teacher-free, coverage-weighted warm start for RLVR

Declared October 30, 2026, after Pre-registration 35's phase 1 and before any run of this design; revised before
launch after Codex's review (/tmp/p30/codex_p36_out.md, nine points; all adopted). The analysis is frozen in
analyze_p36.py, written before any P36 run.

## Why

Pre-registration 35 (phase 1, RL seed 8): SFT on the base's own verified solutions to hard problems (SELF) gave an
early-RL matched pass@16 gap over Dr. GRPO of +2.41 [95%: +1.13, +3.63], as large as teacher solutions on the same
prompts (+2.23) and teacher solutions with corrupted final answers (+2.03), while moving pass@1 only from 21.4 to
23.8. The offset therefore survives self-solution SFT (no teacher; correctness still filtered by the verifier) and
corrupted terminal labels (teacher reasoning and earlier answers intact); the controls remove these ingredients
separately, not jointly. First-order matched-accuracy coverage says a gain needs success to move toward problems with
low p_x, weighted by the marginal pass@k value k(1 - p_x)^(k-1). This registration tests a method built on both:
sample the SFT stream by that value, using the base's own rare successes, then run standard RL.

## Arms (Qwen2.5-Math-1.5B; nominal dose 25K presented response tokens; sft_start.py defaults: AdamW 1e-5, batches of
8, clip 1; then Dr. GRPO with Pre-registration 21's recipe, 150 steps; all arms on 8 x A100 machines)

Self pool (p36_prepare.py; data/p36_prepare_report.json with hashes): 3,795 training problems with at least one
finished, verified base response among Stage 0's 64 samples (27,185 responses, at most 8 per problem, one terminal
format; finished = original sample at most 1,000 tokens). The 110 problems with no verified base response (c = 0 of
64) cannot enter a self pool; targeting claims are limited to c >= 1.
- **CWS (method):** problems drawn with replacement with weight 16 (1 - p_x)^15, p_x the Beta-prior posterior mean of
  the 64-sample counts (55% of draws on c = 1-4, 43% on c = 5-16; uniform: 9% and 37%).
- **UNI:** the same pool and draw mechanism with uniform weights.
- **HARD:** the same pool and draw mechanism, uniform over the 331 pool problems with c <= 4 and zero elsewhere
  (data/p36_w_hardself.json): hard-only selection with P36's pool, sampler and seeds, so CWS minus HARD isolates the
  weighting within the self pool.
- **TAIL:** UNI with TailSFT's token filter (arXiv 2608.25756, sft_start --tail). An adaptation: batch 8, floor
  rounding of the drop count, a schedule on presented-token progress and a 25K dose, versus the paper's batch-16
  selection, nearest rounding and two epochs; no claim about TailSFT in general. Its initial scoring (one forward
  pass over all 27,185 pool sequences, 3,399 batches per seed) is counted in its preparation cost.
- **TCWS (teacher, secondary):** CWS's weights over the 3,550 problems with archived teacher solutions
  (Qwen2.5-Math-7B-Instruct, data/p30_archive.json, data/p36_weights_pmv16.json; 9.1% of draws on c = 0).
- **HSELF:** Pre-registration 35's SELF start (163 teacher-pair-selected hard problems, its own solution selection,
  shuffled passes, SFT seed 3030), RL seeds 8-10 from its phase 2. A different pool and sampler: CWS minus HSELF is a
  comparison with the strongest existing self start, not a test of weighting.
- **TIME (timing ablation):** HSELF's archive, selection and stream applied during RL instead of before it: one SFT
  batch (same optimizer settings, its own AdamW) every 5 RL steps from step 0 until 25K presented tokens
  (train_grpo --isft-*), from the base model; the dose completes within about the first 30 RL steps (the log records
  the step). HSELF's start passes through a bf16 save and reload before RL; TIME keeps fp32 parameters throughout.
  This difference is disclosed, not removed.
- **References:** Dr. GRPO from the base, B1k seeds 8-10 (one L40S each; Pre-registration 15's runs), and the long
  replicates B1kL seeds 8-10 (Pre-registration 30, Amendment 5) as a separate secondary reference.
Seeds: SFT seeds 3636, 3737, 3838 paired with RL seeds 8, 9, 10 for CWS, UNI, HARD, TAIL and TCWS (SFT and RL
variability both enter the seed-level estimates). CWS and UNI also get a fourth pair, SFT seed 3939 with RL seed 11;
it enters only the CWS-versus-UNI contrast (no B1k11 exists). HSELF and TIME use SFT seed 3030 with RL seeds 8-10.

## Evaluation

Checkpoints 1, 2, 5, 10, 25, 50, 100, 150 on the 511 test problems (32 samples, T = 1, 1,024 tokens); checkpoints
50, 100, 150 also at 2,048 tokens (references at 2,048 tokens: B1k8 from job worth-p30-check2k-s0-20261030-061010,
B1k9 and B1k10 from job worth-p30-ref2k-s0-20261030-201539, base with 64 samples in P36's job 4). Every start on the
test problems with 64 samples at 1,024 and 2,048 tokens.

## Estimator (fixed now; analyze_p36.py)

For a run X_s and a reference trajectory R_s (a list of evaluated models with the same RL seed), each checkpoint with
pass@1 u contributes pass@16(X_s) minus R_s's pass@16 linearly interpolated at u (R_s sorted by pass@1, ties by
step). Eligible checkpoints are fixed on the full data: u inside R_s's pass@1 range; a seed needs at least two
eligible checkpoints, otherwise the contrast is "insufficient overlap". The seed estimate is the mean over its
eligible checkpoints; the pooled estimate weights seeds equally. Bootstrap: 2,000 index resamples of the 511 test
problems (seed 36), shared by every quantity and every arm, references recomputed in every draw. **Support
(amended from Pre-registration 30):** the checkpoint set stays fixed; if any eligible checkpoint leaves R_s's range in
a draw, that seed's draw is unavailable (and so is the pooled draw), rather than averaging the surviving checkpoints.
A seed with more than 10% unavailable draws makes the contrast "not interpreted (support)". Two-sided p = 2 min(share
of available draws <= 0, share >= 0), capped at 1. The intervals condition on the trained runs: they measure test-set
sampling, not training-seed variability; seed-level estimates are always reported.

## Readings (fixed now)

- **A. Each arm against Dr. GRPO (R_s = B1k_s: base and steps 25-150; seeds 8-10).** Arms CWS, UNI, HARD, TAIL,
  TCWS, TIME; Holm over these six. "Curve raised": pooled >= +2, 95% interval above 0, every seed positive, Holm
  p <= 0.05; "lowered" is the mirror; otherwise inconclusive. Secondary (no verdicts beyond the same rule, separate
  Holm family): the same against B1kL_s (base and steps 25-700), once B1kL is available.
- **B. Method against baselines (primary).** At each CWS checkpoint, pass@16 minus the same-RL-seed baseline
  trajectory (the baseline's start and its RL checkpoints 1-150) at the checkpoint's pass@1. Baselines UNI (seeds
  8-11), HARD, TAIL, HSELF (seeds 8-10); Holm over these four. **CWS beats a baseline** if the pooled difference is at
  least +0.5 points (a minimum point estimate) with a 95% interval above 0, every seed estimate positive, and Holm
  p <= 0.05 for the test of zero; this establishes a difference from zero with an estimate of at least +0.5, not that
  the true difference exceeds +0.5. **The baseline beats CWS** by the mirror rule. These two Holm-adjusted
  verdicts take priority; otherwise the result is inconclusive, and "practical average equivalence" is noted if the
  pooled 90% interval lies within +/-1 point (about half of H25's +1.92), a descriptive, multiplicity-unadjusted
  reading that supports no decision. A contrast needs every registered seed pair and checkpoint set; otherwise it
  is unavailable and enters its Holm family with p = 1.
- **C. Teacher recipe versus self recipe (secondary, its own family of one).** TCWS's checkpoints against CWS's
  trajectory, rule B. The two recipes differ in pool (3,550 versus 3,795 problems; TCWS includes c = 0 problems),
  solution source and solution processing, so C compares recipes; it does not isolate a teacher-source or a
  teacher-weighting effect (no matched teacher-uniform arm; R25 and H25 are not sampler-matched).
- **D. Timing (its own family of one).** One orientation: HSELF's checkpoints 1-150 against TIME's trajectory (the
  base, then TIME's checkpoints 1-150), rule B; "before RL is better" if HSELF beats TIME's trajectory, "during RL
  is better" by the mirror rule. Post-dose reading (descriptive): HSELF's checkpoints against TIME's checkpoints
  saved strictly after the logged completion step (a checkpoint at step t is saved before step t's SFT update), at
  least two eligible checkpoints per seed; with the current stream the dose completes at logged step 30 (28,041
  presented tokens), so checkpoints 50, 100 and 150.
- **E. Pre-registration 35, phase 2 (frozen here; seeds 8-10, checkpoints 1-150).** SELF (= HSELF), TEACHF, CORR,
  H25b against B1k_s: reading A's rule, Holm over these four. Teacher content: TEACHF's checkpoints against SELF's
  trajectory, rule B, its own family of one. The phase-1 readings (CORR and H25b within 0.5 points of H25) are
  re-reported with three seeds, as descriptive comparisons against H25's Stage 2 estimate.
- **F. OPD (descriptive).** CWS's checkpoints against OPD25's trajectory (Pre-registration 34, seeds 8-10), rule B's
  estimator, no verdict.
- **Mechanism diagnostics (not verdicts; diversity_decomposition.py gets P36's run manifest and the shared
  seed-36, 2,000-draw schedule before any P36 result is read):** matched pass@2 and wrong-answer collision for every arm
  (diversity_decomposition.py); pass@16 by base-count stratum; start offsets; 2,048-token gaps at checkpoints 50-150;
  SFT presented and retained tokens, updates, problems and solutions visited; preparation cost (base samples drawn,
  verifier calls, TAIL's scoring passes).

## Decision (week 4)

Every decision needs interpretable results (status ok) for all four B contrasts and for CWS's and UNI's A readings;
otherwise the decision is "unavailable" until they are.

- **Method paper ("coverage-weighted self-SFT before RL"):** CWS beats UNI and HARD, and no baseline (UNI, HARD, TAIL,
  HSELF) beats CWS.
- **"Hard-problem self-SFT before RL" (weighting not credited):** CWS beats UNI but not HARD, and no baseline beats
  CWS (HARD against UNI is reported descriptively).
- **"Self-SFT before RL" without a demonstrated targeting benefit:** CWS does not beat UNI, and CWS and UNI both have pooled
  estimates of at least +1 against Dr. GRPO with 95% intervals above 0.
- Otherwise the paper remains a measurement paper on coverage at matched accuracy.
Claims about training-seed reproducibility rest on the per-seed estimates (four seeds for CWS versus UNI, three
otherwise), not on the problem bootstrap.

## Cost (estimate)

Seventeen SFT starts (minutes each), twenty P36 RL runs and twelve P35 phase-2 runs with evaluations (about 6.5
hours each on one A100), packed on four 8 x A100 machines (about 7 hours each): about $900 in total.
