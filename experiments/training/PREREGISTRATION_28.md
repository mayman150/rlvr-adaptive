# Pre-registration 28 (evaluation only): published checkpoints on the curve, and a forced-answer test

Declared October 28, 2026, before any of these evaluations; reviewed by Codex before launch.

## Why

(1) Our curve comes from our own runs. If the listed published RLVR checkpoints, trained from the same base models
with other recipes, horizons and prompts, also lie on it under our fixed protocol, the result extends beyond our
runs, and the frozen frontier law predicts these checkpoints' pass@16 from their pass@1. (2) Investigation 27's
decomposition attributes about a third of Dr. GRPO's pass@1 gain at 1,024 tokens to finishing answers; a forced answer
measures what that decoding intervention (an answer cue plus extra greedy tokens) recovers on the base model.

## Checkpoints (Hugging Face, verified October 28: base model, architecture, safetensors)

- **1.5B (Qwen2.5-Math-1.5B):** sail/Qwen2.5-Math-1.5B-Oat-Zero; Elliott/LUFFY-Qwen-Math-1.5B-Zero (off-policy
  guidance from stronger-model traces); rzzhan/ExGRPO-Qwen2.5-Math-1.5B-Zero (experience replay);
  RoadQAQ/ReLIFT-Qwen2.5-Math-1.5B-Zero (interleaved supervised fine-tuning); relex-rlvr/RLVR-Qwen2.5-Math-1.5B and
  relex-rlvr/RELEX-Qwen2.5-Math-1.5B (rank-1 extrapolation); ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1 and
  -7.5k-MATH; RLHFlow/Qwen2.5-Math-1.5B-GRPO-n8-easy and -DAPO-easy; yangzhch6/Zero-Solver-Qwen2.5-Math-1.5B-L;
  zbeeb/Qwen2.5-Math-1.5B-Base-GRPO.
- **7B (Qwen2.5-Math-7B):** sail/Qwen2.5-Math-7B-Oat-Zero; hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo and -SimpleRL-Zero;
  Elliott/LUFFY-Qwen-Math-7B-Zero and -Zero-On-Policy; RLHFlow/Qwen2.5-Math-7B-Zero-RAFTpp;
  ypwang61/One-Shot-RLVR-Qwen2.5-Math-7B-pi1; stellalisy/rethink_rlvr_reproduce-ground_truth- and -random-
  qwen2.5_math_7b-lr5e-7-kl0.00-step150 (correct vs random rewards).
Labels: external solutions or supervision (LUFFY off-policy variants, ReLIFT); random reward (the rethink_rlvr
random-reward run, not correctness-only); LUFFY-Zero-On-Policy is LUFFY's on-policy baseline; the rest are
correctness-only RL. Each checkpoint is pinned to the Hub revision recorded in the plan; a checkpoint whose
vocabulary or added tokens differ from ours is not evaluated (reported as such).

## Protocol (identical for every model)

Our prompt (the base model's tokenizer and chat template replace each checkpoint's), T = 1, 32 samples, the 511 test
problems, vLLM; 1.5B at 1,024 and 3,072 tokens, 7B at 2,048 tokens. Checkpoints were trained with their own prompts and
budgets, so this measures them under one fixed protocol; truncation, length and boxed rates are reported.
Also run here for Pre-registration 27 and Investigation 27: the 1.5B base with 256 samples at 1,024 tokens, and at
T = 0.4, 0.6, 0.8 (32 samples).

## References and readings (problem bootstrap, 2,000 draws with indices shared by all evaluations of a size; the
## reference is recomputed in every draw)

- **Reference A (where the checkpoint's pass@1 lies in the support):** Dr. GRPO's trajectory and temperature envelope
  at the same budget (B1k seeds 8-10 for 1.5B at 1,024; B7 seeds for 7B at 2,048), Pre-registration 21's estimator
  for a single point.
- **Reference B (everywhere it is defined):** the frozen frontier law (law.py, tau = -0.20) from the base model's
  32-sample counts at the same budget (strict: no prediction outside the law's reachable range). At 3,072 tokens only
  reference B exists.
- **Reference A details:** the trajectory (base and steps 25-150) and the step-150 temperature trajectory (T = 1,
  1.15, 1.3) are separate curves; the envelope takes the larger defined value; a point outside both supports is
  "insufficient overlap" for reference A; clamped bootstrap points are reported, and more than 10% clamping makes the
  point "not interpreted".
- **Per checkpoint:** gain = pass@16 minus the reference at the checkpoint's pass@1. Departure: |gain| >= 2 with a
  Holm-adjusted 95% interval excluding 0. Equivalence (pointwise, unadjusted): 90% interval within +/-2. Otherwise
  inconclusive. **Holm families:** one family per (model size, budget, reference): 1.5B at 1,024 with A; 1.5B at
  1,024 with B; 1.5B at 3,072 with B; 7B at 2,048 with A; 7B at 2,048 with B. Supervision labels do not split
  families.
- Reference B at a budget other than the law's fitting budget (1,024 tokens for 1.5B, 2,048 for 7B) is an
  extrapolation across budgets, reported separately from extrapolation across pass@1. Base counts: BASE15 (1.5B, 32
  samples, 1,024 tokens), BASE15_3k (this job, 3,072), and Pre-registration 20's 7B base evaluation (2,048).
- **Group reading:** the numbers of correctness-only checkpoints that are equivalent, inconclusive, and departing up or
  down, under each reference. **Registered expectation:** no correctness-only checkpoint departs upward under
  reference A; departures, if any, occur under reference B beyond the law's validated range, for long-context
  checkpoints at 1,024 tokens (truncation), or for the external-supervision checkpoints.

## Forced answer (reported)

For every sample without a final box, the evaluator appends "\n\nFinal answer: \boxed{" (7 inserted tokens) and
generates up to 32 new greedy tokens, then scores the full text with the same verifier ("forced" counts). Control:
the same samples continued greedily for the same 39 tokens without the cue ("extended" counts). The cue's effect is
forced minus extended; forced and extended points use extra computation beyond the budget, which is stated when they
are compared with reference A. Forced and extended text tails are saved for audit. Models: the 1.5B base and B1k8 at steps
25, 50, 100, 150 (1,024 tokens); the 7B base at 2,048. Reported: forced pass@1 and pass@16; the share of the base's
step-0-to-B1k8-step-150 pass@1 gain closed by forcing the base, (forced base - base) / (B1k8 step 150 - base); whether
the forced points lie on Dr. GRPO's curve (reference A). This measures the intervention's effect, not the share of RL's
improvement caused by learning to finish.

## Cost

Estimate: one 8 x A100 machine for about 2.5-3 hours (39 evaluation specifications, 44 evaluation points: B1k8
contributes four checkpoints and the low-temperature specification three temperatures; longest-first over 8 lanes;
downloads included): about $80-100.
