# Pre-registration 14: does seeding the replay archive with base-model discoveries move the curve?

Declared October 20, 2026, before any training run of this design (see this file's commit time).
The discovery stage (base-model sampling, below) started before this file was committed; it is
fixed data preparation with no outcome measurement. Designed with an independent reviewer
(Codex, three rounds).

## What was known when this was chosen

Pre-registration 12's primary verdict was pending (seed 8's evaluation unfinished). Interim
results visible at this time: coverage-weighted memory (CM, lambda 1) about +1.4 (seed 9) and +0.6
(seed 10) points of pass@16 above Dr. GRPO's trajectory-and-temperature envelope at matched pass@1;
uniform memory about +0.5 and +0.7; unlikeliness about 0; CM's final pass@1 about 1.8 points
below Dr. GRPO in seeds 9 and 10. Pre-registration 12's table and verdict will be reported in full
and unchanged; this is a separate, registered follow-up, and its evaluation set is the same
(reused) 511 problems, so its halves cannot serve as untouched confirmation.

## Why

In 150 steps of 16 questions, each training question is visited about once; a group of 8 finds a
success on a question with p = 1/16 only 40% of the time (p = 1/32: 22%), so the policy's own archive
misses most of the questions whose pass@16 can still rise. Thirty-two base-model samples raise
discovery to 87% and 64%. Seeding the archive with these discoveries tests whether replay can
reach the tail; uniform replay and supervised warm-starting on the same data test whether the
coverage-weighted, continual use of it matters.

## Discovery (shared by all arms and seeds)

Base Qwen2.5-Math-1.5B, all 3,920 training problems, 32 samples each at temperature 1, answers up to
512 tokens, fixed seed (1414 + chunk offset). Per question: the success count c of 32, and the first
answer that is verifier-correct, ends with a stop token, and has a parseable final box (identical
filters for every arm; not the shortest). Merged into `data/discovery_qwen_train32.json`. Its cost
(125,440 completions, GPU-hours measured) is reported in full; inference is conditional on this
one archive.

## Arms (seeds 8, 9, 10; paired with Pre-registration 12's runs)

All are Dr. GRPO with Pre-registration 12's settings (16 x 8, AdamW 1e-6, 150 steps, 512 tokens):
- **D32 (candidate): seeded coverage-weighted memory.** The archive starts with one base success per
  solved question; priorities start from (c, 32); a question's base trace is retired when the
  policy's first own success for it is stored; at most 4 own successes per question; priorities
  update from the latest online group; 8 replayed questions per step drawn by the posterior mean of
  k p (1 - p)^(k-1), k = 16; **lambda = 0.5**.
- **U32: seeded uniform memory.** As D32, uniform question selection.
- **S32: warm start.** Supervised fine-tuning on the same initial successes (mean token
  log-likelihood, AdamW 1e-5, batches of 8, 2.46 million response tokens, gradient clipping 1),
  optimizer reset, then plain Dr. GRPO.
- **C05: online coverage memory at lambda 0.5** (no seeding), isolating seeding at equal strength.
- References from Pre-registration 12 (same seeds, same questions per step): Dr. GRPO (B) and its
  envelope; online CM at lambda 1 (context only).

## Evaluation

Pre-registration 12's set (511 problems, labels audited), 32 samples, steps 50, 100, 150 for every
arm; S32 also right after fine-tuning (step 0) and at T = 1.15 and 1.3 at step 150; greedy accuracy
on 3,382 test problems at step 150.

## Decision rules

Pre-registration 12's procedure (matched-pass@1 interpolation, the envelope recomputed per
bootstrap draw, 2,000 draws over problems, overlap and completeness rules).
1. **Joint win (D32 against B's envelope):** pooled gain at least +2 points, 95% interval above 0,
   positive in all three seeds, **and** step-150 pass@1 minus B's with its 90% interval inside
   [-1, +1]. Otherwise reported by Pre-registration 12's categories (moves the curve without equal
   accuracy; average agreement; worse; inconclusive).
2. **Mechanism contrasts** (D32 against U32's trajectory, D32 against S32's trajectory, D32 against
   C05's trajectory): a contrast supports its claim if the pooled gain is at least +1 point with a
   95% interval above 0. Holm's correction over the directional tests of rule 1 (D32), Pre-
   registration 12's CM primary test, and these three contrasts.
3. Claims by outcome: joint win plus the U32 and S32 contrasts: coverage-weighted continual replay
   improves the use of a shared discovered archive. Joint win, but comparable to S32 or U32:
   superiority over warm-starting or uniform replay is not shown. Coverage gain without accuracy
   equivalence: an accuracy trade-off, reported as such. Agreement or inconclusive: scoped verdict;
   no general discovery bottleneck is claimed.

Reported without a rule: archive growth and retirement of base traces, replay tokens, GPU-hours
including discovery and SFT, gains by base-difficulty strata (c of 32), formatting, and the
reasoning validity of correct answers at step 150 if rule 1 passes.

## Cost

Discovery about $40, training and SFT about $135, evaluation about $185, reserve $40.
