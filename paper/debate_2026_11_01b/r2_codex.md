Edits required for sign-off:

1. Separate equal-compute gains from matched-pass@1 frontier displacement. Define persistence at preregistered late checkpoints with interpolation support guards. Limit negative conclusions to the confirmed candidate; continue the warm-start arm to test transience.

2. Fix Δ by scientific relevance; report minimum detectable effect separately. Power the decision requiring Δ on **each family**, using paired problem/seed uncertainty. Pooled power can hide SmolLM3 failure. Label 0/256 as finite-sample zeros.

3. Rebudget before fixing steps. At R1’s 4,096-token estimate, the nominal schedule implies approximately **960–1,040 GPU-hours**, depending on pilot reuse: **$3,840–4,160** with perfect eight-GPU utilization, before screening/evaluation or SmolLM3 slowdown. Reduce pilots or increase budget; reserve capacity. Specify wide-group rollout totals and update counts.

4. Gate SmolLM3 on a complete 4,096-token generation/backward/optimizer cycle on the booked A100. The current FP32 trainer needs roughly **48GB** for 3B weights, gradients and Adam moments alone. Measure peak memory including KV caches and optimizer temporaries; bound generation batches and training microbatches.

5. Require loss/gradient parity with official Pass@k Training and Clip-Cov before pilots. Existing `passk` and entropy-floor options require separate validation. Make the [trainer](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/train_grpo.py:2145) reject unknown flags; it currently ignores them. Exclude unvalidated methods from confirmatory claims.

6. Recount Stage 0 after amendments: four 200-problem, 256-sample headroom screens add **204,800 completions and 0.629 billion maximum tokens**. Apply truncation gates to hard_val too. Distinct final answers measure output dispersion; correct-strategy diversity needs blinded trace taxonomy at matched successful-sample counts.