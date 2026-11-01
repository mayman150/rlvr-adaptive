# Round 2: coordinator's synthesis for sign-off (read r1_codex.md and r1_fable.md)

Agreed by both: hard math (A) is the primary setting, a procedurally generated Countdown task (B, Reasoning Gym,
canonical expression trees) the mechanism/fallback setting; code is out of scope this cycle. Models: Qwen2.5-Math-1.5B
(continuity) and SmolLM3-3B-Base (second family); nothing counts unless it holds on SmolLM3. The training pool must be
8-gram-deduplicated against hard_val and hard_test (blocking). Primary reading at equal measured compute; matched-pass@1
frontier as diagnostic; strategy diversity measured separately (exact in B; blinded taxonomy plus distinct final
answers in A). Pre-register a middle outcome (gains only where base coverage is nonzero).

Coordinator's resolution of the differences:
1. Claim (merging both): "In settings with headroom, on two model families, some RL-side intervention using only the
   policy's own verified rollouts (Pass@k Training, a wide-group BroRL-style run at equal rollouts, or an
   entropy-preserving update) raises pass@128 on a frozen hard set above Dr. GRPO's matched-pass@1 frontier by at least
   Delta, persistently; versus: all RL-side arms stay on one curve and only warm starts rise above it, transiently."
2. Data: Big-Math-RL-Verified (Apache-2.0, per-problem llama8b solve rates for difficulty strata) filtered to single
   rational answers, about 4,000 problems stratified by solve rate; DeepScaleR-Preview (MIT) as the backup pool.
3. Token budget: Week 1 measures truncation at 2,048 and 4,096; train at the smaller budget with base truncation
   below 10% on the training pool; evaluate at the same budget.
4. Delta: fixed in Week 1 by a power calculation (the smallest effect with >= 80% power for the pooled contrast,
   given the 400 test problems, 256 samples and three seeds), not below 2 points.
5. Week 2 arms (one seed, 150 steps, equal measured compute, both models): Dr. GRPO, Pass@k Training (official
   implementation), wide groups at equal total rollouts, Clip-Cov entropy-preserving update, 25K teacher warm start.
   The RL candidate is frozen on validation. Week 3: candidate vs Dr. GRPO, three seeds, about 450 steps, references
   to about 600 steps.
6. Stage 0 amendments before launch (it is still queued): add SmolLM3 and Qwen headroom screens on hard_val at 2,048
   and 4,096 tokens (validation only, 256 samples), save distinct final answers (already in saved texts), correct the
   registered completion count (1.63 million scheduled, at most 3.1 billion tokens). Gate V must pass on both models.

Sign-off (max 300 words): accept, or give concrete edits; name anything that would make the plan fail in practice
(compute, capacity, implementation risk of Pass@k Training / Clip-Cov in our trainer, SmolLM3 at 3B on one A100).
