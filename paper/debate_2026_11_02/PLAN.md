# Agreed research plan (Claude, Codex, Fable; November 1, 2026). Target: NeurIPS 2027, quality first

## Paper
A science paper on how RLVR acquires and retains reasoning; a method claim only if the mechanism supports one.

**Question.** Under fixed acquisition and training budgets, when does teacher reasoning give persistent,
reasoning-valid coverage gains over deep self-sampling? Does delivery (SFT before RL versus in-context guidance
withdrawn during RL) change how long those gains last?

**Possible outcomes:**
- deep self matches the teacher, meaning the teacher's advantage is acquisition efficiency;
- the teacher stays better on identical problems, meaning it carries information the student's search does not
  recover;
- only guidance during RL keeps the gain, meaning delivery governs consolidation;
- every gain decays, meaning RL erodes it or the reference catches up (these two are distinguished directly);
- the teacher wins only through covering more problems, meaning acquisition coverage rather than content.

Every outcome is informative, and a negative result tests the strongest counterexample family on two models. The
effect is described as displacement from the Dr. GRPO reference curve, not as a global frontier, and the paper
engages the assumptions of No Free Lunch (2610.39295) rather than claiming to refute it.

## Design
- **Core, Qwen2.5-Math-1.5B:** a 2 x 2 of source {teacher, deep self} x delivery {SFT then RL, MFC-style curriculum
  port}, plus plain Dr. GRPO. The curriculum port uses per-problem hint frontiers on the Dr. GRPO core, with complete
  withdrawal by step 200. That cut is a registered deviation from MFC, and the frontier state at the cut is reported.
- **SmolLM3-3B-Base:** plain Dr. GRPO plus both sources under the delivery selected on validation (a narrower
  cross-family claim; whether to run SmolLM3's full factorial is still open).
- **Controls:**
  - source comparisons on problems with both self and teacher traces, and separately on whole acquisition pipelines;
  - matched trace banks, multiplicity, lengths and exposure, with exposure logged;
  - answer-free teacher banks (every target-answer occurrence removed) and an answer-only control;
  - SFT on the visible-prefix bank, only if an effect is attributed to scheduling;
  - interleaving is not rerun (already answered by TIME).
- **One bounded RL-side audit arm** in the pilot (zero-success-only updates or Pass@k Training). DATPO is not
  reimplemented, and the paper says why.
- **Retention terms** (KL to the SFT start; 2605.29190's novelty reward, correctly named) only after diagnosing what
  declines.
- **Measurement:**
  - three paired seeds; persistence to 700 steps (500 in the minimal version); references extended to 1,200;
  - registered late sealed-test readings, pass@1 to pass@256, with n >= k;
  - a seed-level plus problem-level bootstrap;
  - matched accuracy for displacement, equal acquisition-plus-training compute for usefulness;
  - reasoning-valid coverage on a frozen hard subset with calibrated judges (blinded calibration set, two judges,
    human audit, unknowns explicit).
- **Setting:**
  - the frozen hard sets (OlympiadBench + AIME) and a common hard training pool, with rarity re-measured at the chosen
    budget;
  - Qwen responses of at most 3,072 tokens (4,096 total context);
  - fill the teacher-coverage hole: teacher traces exist for only 60 of the 109 zero-observed-success problems.

## Stages and costs
Shared model: $4 per A100-hour; 4 A100-minutes per Qwen step and 8 per SmolLM3 step at 2,048 tokens (the SmolLM3
figure is a placeholder until calibrated).

| Stage | Content | Full | Minimal |
|---|---|---|---|
| 0 | Amended acquisition census, harvest, headroom gates for both families, TEACHF retention probes, Gate-H fallback, separate training calibration | $400-700 | $400-700 |
| 1 | Qwen five or six arms x 150 steps, exploration audit, SmolLM3 smoke runs | $400-500 | $400-500 |
| 2 | Qwen confirmation (2 x 2 + Dr. GRPO, 3 seeds) | about $4,500-5,000 | about $3,300 |
| 3 | SmolLM3 selected-delivery confirmation | about $5,300-6,800 | about $3,700 |
| 4 | Content and exposure controls, mechanism | $1,500-3,200 | only if needed |
| **Total** | | **$12-16K** | **about $8.5K (Fable) to $12K (Codex)** |

Minimal-version cuts (Fable): 500 steps with references to 700, 256 samples at two late points with k <= 128, and
Stage 4 only if a positive effect needs it. Not cuttable: two families, three seeds, the full 2 x 2 on Qwen, and
reference support built before the arms.

Timeline: Stages 0-1 in November; Qwen confirmation in December and January; SmolLM3 in February and March; controls,
mechanism and writing in April and May; decide on an early arXiv release at the end of Stage 2.

## Remaining disagreements
- Whether delivery effects must replicate across SmolLM3's full factorial (Codex) or only the selected delivery.
- Forced withdrawal by step 200 versus MFC's success-driven frontier (Fable: register it as a deviation).
