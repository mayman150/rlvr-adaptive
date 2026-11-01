# Round 2: coordinator's synthesis for sign-off (read r1_codex.md and r1_fable.md)

Converged:
- Paper type: a science paper on the acquisition and retention of reasoning in RLVR; a method claim only if the
  mechanism supports one.
- Question: under fixed acquisition and training budgets, when does teacher reasoning give persistent,
  reasoning-valid coverage gains over deep self-sampling, and does delivery (SFT before RL versus adaptive in-context
  guidance withdrawn during RL) change persistence? Delivery is either weight-space (SFT before RL) or in-context
  (guidance during RL).
- Core: a 2 x 2 of source {teacher, deep self} x delivery {SFT then RL, MFC-style curriculum port with complete
  withdrawal by step 200}, plus plain Dr. GRPO. Retention terms (KL to the SFT start, or 2605.29190's novelty reward,
  correctly named) are added only after we diagnose what declines.
- Controls:
  - source comparisons on the intersection of problems with both self and teacher traces, and separately on whole
    acquisition pipelines;
  - matched trace banks, multiplicity, lengths and exposure, with exposure logged;
  - SFT on the visible-prefix bank if an effect is attributed to scheduling;
  - answer-free teacher traces (every target-answer occurrence removed) and an answer-only control;
  - interleaving is already answered by TIME, so it is not rerun.
- Exploration: one bounded RL-side audit arm (BBG-style zero-success-only updates or Pass@k Training) in the pilot;
  DATPO is not reimplemented (reason stated).
- Reasoning-valid coverage: on a frozen hard subset with calibrated judges (blinded calibration set, two judges,
  human audit, unknowns kept explicit); cot_judge.py to be fixed.
- Setting:
  - the hard pool and frozen hard sets; Qwen2.5-Math-1.5B (context 4,096 total, so responses <= 3,072) and
    SmolLM3-3B-Base;
  - rarity re-measured at the chosen budget;
  - fill the teacher-coverage hole: teacher traces exist for only 60 of the 109 zero-observed-success problems.
- Persistence: three paired seeds, about 700 steps, references extended to about 1,200 steps; sealed-test readings at
  registered late steps.
- Stage 0 amendments (Codex's list plus Fable's):
  - corrected counts;
  - headroom gates for both families;
  - TEACHF retention probes;
  - a registered Gate-H fallback;
  - a separate training-calibration registration.

Open: the cost model. Codex: $12-19K for the whole program (Stage 2 Qwen confirmation $4.5-5.5K; Stage 3 SmolLM3
$5.5-7K). Fable: $5.5-7K for the quality version.

Sign-off (max 350 words):
(1) Reconcile the costs with one shared model. Use $4 per A100-hour, 4 A100-minutes per Qwen step and 8 per SmolLM3
    step at 2,048 tokens, as Codex assumed, or argue a different number. State the training and evaluation cost of each
    stage and the total.
(2) Propose a minimal credible version (what can be cut without weakening the main claim) and its cost.
(3) Name any remaining disagreement in one line.
No praise.
