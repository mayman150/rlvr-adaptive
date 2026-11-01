# Agreed plan (Claude, Codex, Fable; November 1, 2026): test the question where it can fail

## Question and claim
In settings with headroom, on two model families: does some RL-side intervention that uses only the policy's own
verified rollouts (Pass@k Training, wide groups at equal total rollouts, or a Clip-Cov entropy-preserving update)
raise pass@128 on a frozen hard set above Dr. GRPO, persistently? Or do all RL-side arms stay on one curve, with
only warm starts rising above it, and only transiently?

Two readings are kept separate:
- equal-compute gains (primary for practice);
- matched-pass@1 frontier displacement (the diversity claim), judged at pre-registered late checkpoints with support
  guards.

Thresholds, fixed before any run:
- superiority Delta >= 2 points on each family, fixed by scientific relevance; the minimum detectable effect is
  reported separately;
- equivalence margin +/- 2 points;
- a middle cell: gains only where base coverage is nonzero, which supports "RL redistributes within support" and is
  a win for neither side.

## Setting
- A (primary), hard math: training pool of about 4,000 single-rational-answer problems from Big-Math-RL-Verified
  (Apache-2.0; per-problem solve rates for strata), with DeepScaleR-Preview (MIT) as backup. The pool is
  8-gram-deduplicated against hard_val and hard_test (blocking). Tests: hard_val for screening, hard_test (OlympiadBench
  + AIME) sealed until confirmation; MATH-511 for continuity. AIME rows are used for evaluation only (licence).
- B (mechanism/fallback): procedurally generated Countdown instances (Reasoning Gym, Apache-2.0); valid solutions are
  counted exactly as canonical expression trees.
- Models: Qwen2.5-Math-1.5B (continuity) and SmolLM3-3B-Base. Nothing counts unless it holds on SmolLM3.
- Budget: train at the smaller of 2,048 and 4,096 tokens at which base truncation is below 10%. 256 samples; report
  pass@1/16/64/128/256. Diversity is measured separately: exactly in B; in A by distinct final answers plus a blinded
  strategy taxonomy at matched successful-sample counts.

## Gates (week boundaries are gates, not dates)
1. Stage 0, amended before launch:
   - headroom screens for both models on hard_val at 2,048 and 4,096 tokens (validation only);
   - Gate V: base pass@128 between 10% and 70%, and truncation gates, on both models;
   - corrected completion count;
   - a throughput and memory calibration that includes a full SmolLM3 4,096-token generation, backward and optimizer
     cycle on one A100 (fp32 Adam needs about 48 GB before KV caches).
2. Before pilots:
   - the trainer rejects unknown flags;
   - Pass@k Training and Clip-Cov match the official losses and gradients, and each passes a 25-step smoke run with
     its published signature; an arm that fails is dropped.
3. Week 2 pilot: one seed, 150 steps, equal measured compute, on both models.
   - Arms: Dr. GRPO, Pass@k Training, wide groups (total rollouts and update counts specified), Clip-Cov, and a 25K
     teacher warm start.
   - The RL candidate is frozen on validation.
4. Week 3 confirmation: candidate vs Dr. GRPO, three seeds.
   - Steps: 450, with references to 600. If calibration shows the cost is too high, this may be cut to 300, with
     references to 450.
   - SmolLM3 runs the confirmation only (no third arm).
   - The warm-start arm is continued, to test its transience.

## Cost
Codex's estimate of the nominal 4,096-token schedule is $3,800-4,200 at perfect utilization, before screening. The
plan is rebudgeted after Week 1 calibration (2,048 tokens if truncation allows; fewer pilots; 3B confirmation only);
the target is about $3,000.

## Sources
r1_codex.md, r1_fable.md, r2_codex.md, r2_fable.md, r2_synthesis.md.
