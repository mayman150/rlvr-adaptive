# Pre-registration 35: what does the H25 start carry? Finishing, format, answer key and teacher content

Declared October 30, 2026, before any run of this design; the plan agreed in the three-way review
(paper/debate_2026_10_30/); reviewed by Codex before launch.

## Why

Pre-registration 30, Stage 2: RL from H25 (SFT on 25K nominal presented tokens of teacher solutions to 262 hard
training problems) keeps a positive pass@16 gap over Dr. GRPO-from-base at matched pass@1 through step 50 (+1.92,
95% [+0.68, +3.12]), descriptively close to the start's own pre-RL offset (+1.7 to +1.9). The 25K starts truncate
45-47% of answers at 1,024 tokens, so the offset could reflect finishing or answer formatting rather than reasoning
content, and an earlier self-SFT warm start also shifted the curve. Each control below changes one ingredient.

## Starts (sft_start.py; nominal dose 25K presented response tokens; AdamW 1e-5, batches of 8, clip 1; SFT seed 3030)

Archives and selections from p35_prepare.py (data/p35_prepare_report.json; hashes there):
- **SELF (self-SFT):** the base's own verified responses (Stage 0's 64 samples) on F = the 230 H problems with at
  least one (554 responses; mean 532 tokens). 32 H problems are excluded (no verified base response).
- **TEACHF (teacher on F):** teacher responses on the same 230 problems, as many per problem as SELF has (at most the
  teacher's count), chosen by nearest token length (494 responses; mean 482 tokens). SELF minus TEACHF isolates the
  source of the solutions on identical prompts.
- **FMT (format-only):** up to three boxed base responses per H problem regardless of correctness (786 responses; 14%
  correct), with the loss on the final box's opening "\boxed{", its closing brace and the stop token only
  (--verify boxed --loss-mask format); reasoning and answer tokens are masked. Manipulation check: the start's
  boxed rate must rise and its truncation fall relative to the base at 1,024 tokens; otherwise a null from FMT is not
  read as evidence against a finishing explanation.
- **CORR (corrupted answer key):** the teacher responses on H with the last box's content replaced by a wrong answer
  (n -> n + 1 for integers; "+1" appended otherwise), kept only if the verifier rejects it (--verify boxed). If CORR's
  offset matches H25's, the start does not learn the answer key.
- **H25b:** H25 with a second SFT seed (3131), same archive and selection.

## Evaluation and RL (phase 1)

Each start: the 511 test problems, 64 samples, T = 1, at 1,024 and at 2,048 tokens. RL: Dr. GRPO, Pre-registration
21's recipe, seed 8, 50 steps, checkpoints 1, 2, 5, 10, 25, 50 (32 samples, T = 1, 1,024 tokens).

## Readings (Pre-registration 30's estimator; 2,000 shared problem-bootstrap draws; seed 35)

- **Start offset:** the start's pass@16 minus B1k_s's trajectory at the start's pass@1, averaged over seeds 8-10 (as
  in Stage 2's step-0 report); at 2,048 tokens, against B1k8's 2,048-token trajectory (the registered 2,048-token
  check, job worth-p30-check2k-s0-20261030-061010) only.
- **Early-RL gap (seed 8):** mean over eligible checkpoints 1-50 of pass@16 minus B1k8 at matched pass@1, 90% and 95%
  intervals, the 10% support rule, at least two eligible checkpoints.
- Every reading also for H25 seed 8 (from Stage 2) so that controls and H25 are compared on the same seed.
- **Phase-2 rule (fixed now):** a control gets RL seeds 9 and 10 and steps to 150 (a separate job) if its seed-8
  early-RL gap is at least +0.96 (half of H25's +1.92), or if it enters a confirmatory contrast. SELF and TEACHF
  enter one together: the teacher-content contrast TEACHF minus SELF is read only with three seeds each.
- **Readings fixed in advance (descriptive in phase 1; no Holm family until phase 2):**
  finishing/format explains at least half of H25's effect if FMT's early-RL gap is >= +0.96 and its manipulation
  check passes; the answer key is not what transfers if CORR's gap is within 0.5 points of H25 seed 8's; the offset is
  reproducible across SFT seeds if H25b's gap is within 0.5 points of H25 seed 8's; at 2,048 tokens, the start
  offsets are compared with the 1,024-token offsets (a large drop indicates budget sensitivity, attributed to
  finishing only with truncation and extraction evidence).
- Reported: presented and supervised tokens, optimizer updates, problems and solutions per start; boxed, truncated
  and mean length for every start and checkpoint; extraction audit of 50 random FMT-start answers.

## Cost (estimate)

Five SFT starts (minutes each), ten 64-sample start evaluations, five 50-step RL runs (about 1.5 hours each) and
thirty checkpoint evaluations on one 8 x A100 machine: about 3 hours, about $95. Phase 2 (if triggered) is costed
separately.
