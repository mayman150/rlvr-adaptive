# R2 — Fable sign-off

Accept, with five edits.

1. Cost realism (main failure mode): at $32 per 8xA100 hour ($4/A100-h), the binding item is SmolLM3-3B in Week 3. The 2-min/step figure is for 1.5B at 1,024 tokens; a 3B step at 2,048-4,096 tokens is unmeasured and plausibly 6-8 A100-min. Three seeds x 450 steps plus references to 600 would then cost $1,200-1,500 for the 3B alone, crowding the training cap. Edit: make Week 1 throughput calibration a formal gate with pre-agreed authority to cut Week 3 to 300 steps (references 450), and run 3B confirmation-only (candidate vs Dr. GRPO, no third arm).

2. Implementation risk: Pass@k Training's official repo (RUCAIBox) and Clip-Cov are not in our trainer. Require, before Week 2, a 25-step smoke run per ported arm reproducing the published qualitative signature (entropy trajectory for Clip-Cov; advantage-weight check for Pass@k). An arm that fails the smoke test is dropped, not debugged into Week 3.

3. Pre-register both thresholds now: the superiority Delta (power-fixed, >= 2 points) and an equivalence margin (+/- 2 points). Without the margin, "inconclusive" is unbounded and the frontier side can never win.

4. Reconcile Gate V: PREREGISTRATION_39 says pass@128 in 10-80%; Codex's plan says 10-70%. Fix one number (10-70) in a single pre-launch amendment, both models, 2,048 and 4,096 tokens, validation only — alongside the completion-count correction (1,626,624 / 3.106B tokens).

5. AIME rows in hard_val/hard_test are benchmark-only or unclear licence: fine for evaluation, bars redistributing the sets with the paper; state this.

Capacity: p4de queues of hours mean week boundaries must be gates, not calendar dates; submit the amended Stage 0 immediately. Dedup blocker, two-family requirement, middle-outcome cell: as written.
