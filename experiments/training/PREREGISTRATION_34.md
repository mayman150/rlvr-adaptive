# Pre-registration 34: an on-policy-distillation start on the matched-pass@1 curve

Declared October 29, 2026, before any run of this design; reviewed by Codex before launch. Runs with Pre-registration
30's Stage 2 and shares its dose, problems, reference and readings.

## Why

Dong et al. (arXiv 2609.28145, "RL Starts before RL") report that a student prepared by on-policy distillation (OPD)
from an RL-trained teacher reaches much higher accuracy after the same RL than a base start (Qwen3-8B, 8 text
benchmarks: 59.18 vs 51.90), although the two starts have nearly identical pre-RL pass@k (pass@1 19.3 vs 18.4,
pass@64 80.0 vs 80.4). They report endpoint accuracy only, with one seed per stage. Our question: does RL from an
OPD start lie above Dr. GRPO's pass@16-vs-pass@1 curve at matched pass@1, i.e., does OPD change the curve or only
how far along it RL travels? Pre-registration 30 asks the same of SFT starts; OPD differs in using the teacher's
full distributions at the student's own prefixes, so the teacher never generates text (avoiding the decoding
problems recorded in Pre-registration 30, Amendments 2-3).

## Start

`opd_start.py`: student Qwen2.5-Math-1.5B (base); teacher Qwen2.5-Math-7B-Instruct (revision
ef9926d75ab1d54532f6a30dd5e760355eb9aa4d; same vocabulary; tokenizer files replaced by ours as for every hub model);
problems: Pre-registration 30's R selection (262 problems, data/p30_sel_R.json); one student sample per problem per
visit (T = 1, 1,024 tokens), problems visited in shuffled passes; loss: per-token reverse KL between the student's and
the teacher's full next-token distributions at the student's prefixes, averaged over response tokens; AdamW 1e-5,
betas 0.9/0.99, batches of 8, clip 1 (sft_start.py's settings); OPD seed 3434. The start is saved when the presented
(generated) response tokens reach the dose chosen by Pre-registration 30, Amendment 4 (D tokens). Reverse KL is the
standard OPD objective and the one of the paper's main pipeline; its forward-KL variant (better after RL in the
paper's vision-language ablation) is not run. Before OPD, a 48-problem teacher gate (T = 0.7, top-p 0.8, pass@1 at
least 0.6) checks the downloaded teacher; a failed gate stops the lane.

## Runs and evaluation

The OPD-R start is evaluated like Pre-registration 30's starts (255 validation and 511 test problems, 64 samples,
T = 1, 1,024 tokens) and its validation human-solution NLL is scored (p30_nll.py). RL: Dr. GRPO, Pre-registration
21's recipe, seeds 8, 9, 10 from the OPD-R start; checkpoints 1, 2, 5, 10, 25, 50, 100, 150 on the 511 test problems
(32 samples, T = 1), step 150 also at T = 1.15 and 1.3.

## Readings (Pre-registration 30's Stage 2 rules, Amendment 4)

- **Primary:** OPD-R's RL checkpoints against B1k seeds 8-10 (same-seed T = 1 trajectory; envelope reported) at
  matched pass@1: curve raised / lowered / average equivalence / inconclusive by Pre-registration 30's rules (+2
  threshold, 95% interval, all seeds, Holm over this registration's two contrasts), with P27's LH1 as a single-seed
  extension beyond B1k's support and per-checkpoint gaps shown (early-RL versus persistence).
- **Secondary (Holm with the primary):** OPD-R minus SFT-R at the same dose D, both after RL, at matched pass@1 (each
  arm's same-seed trajectories; the paired gain difference with shared bootstrap draws).
- **Reported:** the start's pass@1, pass@16, pass@32 and pass@64 against SFT-R at dose D and the base (the paper's
  "same pre-RL pass@k" observation in our setting); step-150 pass@1 against SFT-R and B1k; OPD's reverse KL and
  batch reward by step; frozen law forecasts from the OPD start's counts written before RL, as in Pre-registration 30.
- Scope: one teacher, one selection (R), one objective (reverse KL), one dose, 1.5B at 1,024 tokens.

## Cost (estimate)

OPD start (student generation plus teacher scoring, about 20-60 minutes on one GPU), its two 64-sample evaluations,
and three RL runs with evaluations (about 5 hours each on one GPU), packed with Pre-registration 30's Stage 2: about
16-18 GPU-hours, about $65-70 at the packed rate.
