# Pre-registration 39 (Stage 0 of the November 1 schema): harvest yield, large-k re-evaluation, retention probes

Declared November 1, 2026, before any run of this design; the schema is paper/debate_2026_11_01/SCHEMA.md (agreed
by Claude, Codex and Fable). Stage 0 is generation and evaluation only (no training). One 8 x A100 job,
plans/p39_stage0.json.

## Frozen data (data/hard_sets_report.json, hashes there)
- hard_val.json (200: 100 OlympiadBench, 100 AIME) for screening; hard_test.json (400: 200 + 200) untouched until
  Stage 2. Source filters: OlympiadBench OE_TO_maths_en_COMP with a single rational answer and no unit; AIME
  1983-2024; problems with >= 30% 8-gram overlap with MATH train or the 511 test problems removed; figure-dependent
  questions without [asy] removed; split seed 39, stratified by source.
- p39_math511_hard.json: the 208 test problems with at most 8 of 64 base successes (Pre-registration 30, Stage 1).
- p39_train_c0.json: the 110 training problems with 0 of 64 base successes (Stage 0 census); p39_train_c12.json: the
  145 with 1-2 of 64.

## Measurements (T = 1; prompts as in training; seed 3939)
- (a) HARVEST pilot: the base, 1,024 samples per c = 0 training problem at 2,048 tokens.
- (b) SmolLM3-3B-Base census: 64 samples per training problem at 2,048 tokens (plain prompt, as in its viability
  evaluation); its verified solutions on the c = 0 problems form the PEER pool.
- (c) Large-k re-evaluation: 256 samples at 2,048 tokens on p39_math511_hard (208) and hard_val (200) for the base,
  B1k8 and B1k9 (step 150), LH1 (step 1,000), SELF8, TEACHF8, CWS8 (P36), OU8 (UCB-Con, P32) and PSR1k8 (step 150).
- (d) Retention probes: 256 samples at 1,024 tokens (these runs' training budget) on p39_train_c12 for the base,
  the SELF start, SELF8 (steps 50, 150) and B1k8 (steps 50, 150).

## Readings and gates (fixed now)
- Harvest yield: the share of the 110 problems with at least one verified solution, and with at least two;
  verified solutions per problem; length distribution; finish rate. A blind audit (Codex) of 50 random verified
  harvested solutions (seed 39) scores each derivation as valid, flawed but answer-correct, or a guess; the usable
  yield counts only problems with at least one audited-valid solution, estimated as the yield times the audited-valid
  share. **Gate H:** usable yield >= 40% of the 110; otherwise HARVEST is dropped and PEER can still proceed.
- PEER pool: the c = 0 problems with at least one SmolLM3 verified solution (count and lengths), audited the same way.
- Headroom: **Gate V:** base pass@128 on hard_val between 10% and 80%; otherwise the hard set is revised before
  Stage 1 (still without touching hard_test).
- Large-k question ("did a k = 16 null hide an effect?"): for each model, pass@1, pass@16, pass@64 and pass@128 on
  the 208 and the 200 (unbiased estimates from 256 samples), with 95% problem-bootstrap intervals (2,000 draws, seed
  39), and differences from B1k8 at step 150 (paired by problem). Descriptive: a difference is flagged if its 95%
  interval excludes 0 at pass@128; matched-pass@1 readings are not formed here (single checkpoints).
- Retention (suppression versus catch-up): for each probed problem, the success rate at each checkpoint (256
  samples). Reported: the share of the 145 problems whose rate falls below the base's by more than a binomial 95%
  margin, for SELF8 and B1k8 at steps 50 and 150, and the mean change by base count. Codex's rare-success constraint
  is considered for Stage 1 only if at least 20% of problems show such a decline at step 150 in either run.
- Throughput: tokens per second per A100 for each model and budget, for Stage 1 cost planning.

## Cost (estimate)
About 36 A100-hours packed on one 8 x A100 machine (about 5 hours): about $170.
