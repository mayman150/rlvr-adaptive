# Pre-registration 39 (Stage 0 of the November 1 schema): harvest yield, large-k re-evaluation, retention probes

Declared November 1, 2026, before any run of this design; revised before launch after Codex's review (verifier v2, rebuilt hard sets, problem-level gates, retention against each run's own start); the schema is paper/debate_2026_11_01/SCHEMA.md (agreed
by Claude, Codex and Fable). Stage 0 is generation and evaluation only (no training). One 8 x A100 job,
plans/p39_stage0.json.

## Frozen data (data/hard_sets_report.json, hashes there)
- hard_val.json (200: 100 OlympiadBench, 100 AIME) for screening; hard_test.json (400: 200 + 200) untouched until
  Stage 2; built by build_hard_sets.py (filters, 8-gram overlap definition against all of MATH train
  (train_all_subjects.json, train.json) and the 511 test problems, within-pool duplicate clustering, relay and
  missing-picture removal; split seed 39). Local deduplication cannot establish absence from pretraining data.
- Verifier v2 (from this registration on): verifier.py also accepts \frac12, a sign before a fraction, and a leading
  "+". All Stage 0 measurements use v2; cohorts below were re-scored with v2 (9 training counts changed; one problem
  left c = 0).
- p39_math511_hard.json: the 208 test problems with at most 8 of 64 base successes (Pre-registration 30, Stage 1,
  re-scored with v2).
- p39_train_c0.json: the 109 training problems with 0 of 64 base successes (Stage 0 census at 1,024 tokens, v2);
  p39_train_c12.json: the 143 with 1-2 of 64.

## Measurements (T = 1; prompts as in training; seed 3939)
- (a) HARVEST pilot: the base, 1,024 samples per c = 0 training problem at 2,048 tokens (8 chunks of at most 14 problems); every attempt is charged (no early stopping in the pilot); per-sample token counts and finish flags are saved.
- (b) SmolLM3-3B-Base census: 64 samples per training problem at 2,048 tokens (plain prompt, as in its viability
  evaluation); its verified solutions on the c = 0 problems form the PEER pool.
- (c) Large-k re-evaluation: 256 samples at 2,048 tokens on p39_math511_hard (208) and hard_val (200) for the base,
  B1k8 and B1k9 (step 150), LH1 (step 1,000), SELF8, TEACHF8, CWS8 (P36), OU8 (UCB-Con, P32), PSR1k8 and S32 (step 150; trained at 512 tokens, P14).
- (d) Retention probes: 256 samples at 1,024 tokens (these runs' training budget) on p39_train_c12 for the base,
  the SELF start, SELF8 (steps 50, 150) and B1k8 (steps 50, 150).

## Readings and gates (fixed now)
- Harvest yield: the number of the 109 problems with at least one verified, finished solution (and with at least
  two), by depth (first 64, 256 and 1,024 samples); solutions per problem; lengths; finish rate. A blind audit
  (Codex) by problem: for every problem with a verified finished solution, up to two solutions (the shortest first)
  are scored as valid, flawed but answer-correct, or a guess. **Gate H:** at least 44 distinct problems (40% of 109)
  with an audited-valid solution; otherwise HARVEST is dropped and PEER can still proceed. Note: the c = 0 census used
  1,024 tokens, the harvest 2,048; the depth-64 yield at 2,048 tokens is reported to separate budget from depth.
- PEER pool: the c = 0 problems with at least one SmolLM3 verified solution (count and lengths), audited the same way.
- Headroom: **Gate V:** the base's pass@128 on the 200 hard_val rows (RE_BASE counts[208:]) between 10% and 80%, also
  reported by source (OlympiadBench, AIME); otherwise the hard set is revised before Stage 1 (hard_test untouched).
- Large-k question ("did a k = 16 null hide an effect?"): for each model, pass@1, pass@16, pass@64 and pass@128 on
  the 208 and the 200 (unbiased estimates from 256 samples), with 95% problem-bootstrap intervals (2,000 draws, seed
  39), and differences from B1k8 at step 150 (paired by problem). Descriptive: a difference is flagged if its 95%
  interval excludes 0 at pass@128; matched-pass@1 readings are not formed here (single checkpoints).
- Retention (suppression versus catch-up): for each of the 143 probed problems, the success rate (256 samples) at
  each checkpoint is compared with the run's own start: SELF8 (steps 50, 150) with the SELF start, B1k8 (steps 50,
  150) with the base. A problem declines if the Newcombe (hybrid Wilson score) 95% interval for the difference of the
  two rates lies below 0. Reported: the share of declining problems, the mean change, and the shares by base count
  (1 or 2). Codex's rare-success constraint is considered for Stage 1 only if at least 20% of problems decline by
  step 150 in either run.
- Throughput: tokens per second per A100 for each model and budget, for Stage 1 cost planning.

## Cost (estimate)
About 1.53 million completions (at most 2.9 billion new tokens); planned 45-55 A100-hours packed on one 8 x A100
machine (about 7 hours, cap 10 hours): about $230. Measured throughput is reported for Stage 1 planning.

## Amendment 1 (November 1, after Stage 0's results; before any run of this add-on): Stage 0b
One 8 x A100 job (plans/p39b_addon.json; channel p39-stage0-ckpts, now also holding the TEACHF start and TEACHF8 step
50), generation, evaluation and smoke tests only. Readings, all descriptive:
- Teacher coverage: Qwen2.5-Math-7B-Instruct (pinned revision; T = 0.7, top-p 0.8; 2,048 tokens), 64 samples on each
  of the 191 training problems with at most 4 of 64 base successes (verifier v2) and no archived teacher solution
  (data/p39b_teacher_missing.json); reported: problems with at least one verified finished teacher solution, by base
  count (c = 0 and c = 1-4).
- Headroom, validation only: SmolLM3-3B-Base on hard_val at 2,048 and 4,096 tokens (256 samples, plain prompt) and the
  Qwen base at 3,072 tokens; pass@1/16/64/128 and truncation by source. The pilot uses a budget at which the base's
  hard_val pass@128 lies within 10-80% and truncation is below 10% where available.
- Retention: TEACHF8 (steps 50, 150) against the TEACHF start on the 143 probe problems, with the Newcombe rule of the
  Readings.
- Smoke tests (pass/fail, no science): the hint curriculum (10 steps, hard schedule with 4 of 16 problems from the 453
  hard problems, teacher hints withdrawn at step 6) runs, logs non-trivial hint levels and a non-increasing frontier;
  a 2,048-token SFT start trains and RL runs from it; SmolLM3 completes 3 RL steps at 2,048 and at 4,096 tokens, with
  peak memory and seconds per step recorded.
Cost: about 3.5 hours on one machine, about $110.
