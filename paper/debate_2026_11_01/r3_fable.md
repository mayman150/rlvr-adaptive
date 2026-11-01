# R3 (Fable), November 1, 2026

**(1) Accept, with four edits.**
- Stage 1's gate must not depend on frontier support: teacher-like starts exceed B1k's pass@1 range at steps 100–150, so "at matched pass@1 where supported" will often be empty. Make equal-compute hard-set pass@128 the sole binding Stage-1 gate; the frontier reading is advisory.
- Drop BroRL-G512 from Stage 2's required controls (≥$1,000 alone; it trades prompt breadth for depth — cite 2510.01180 instead). Extended Dr. GRPO at equal compute is the binding compute control; add G512 only if Stage 1 passes early and budget remains.
- Restrict PEER's pool to the frozen HARVEST/TEACHF prompt set where both sources have verified solutions, length-bin matched; otherwise it is a content-difficulty comparison, not a source comparison.
- Kill wording: a failed harvest or faded gain establishes the channel result only, not a model-support ceiling (Codex's disagreement line is correct; the fallback must not claim a ceiling).

**(2) Stage 1 gate.** Settle on ≥+2 pp hard-set pass@128 vs extended Dr. GRPO at equal compute AND ≥+1 pp vs SELF/uniform-RFT, both seeds positive. Reason: the largest verified content effect in-house is +1.5–2.1 pp; a ≥3 pp screen on two seeds has near-zero power against a true +2 effect, so it selects noise, not methods. Stage 2 keeps the registered confirmatory rule (pooled ≥+2, 95% CI above zero, all seeds positive, Holm).

**(3) PEER belongs** — it is the cheapest discriminator of "different model" vs "stronger model," and the census is already queued. Confounds: (i) difficulty profile — SmolLM3 (pass@1 44.3 vs 21.6) solves a different problem set, so an unrestricted pool changes content difficulty, not just source; (ii) length/finishing — SmolLM3 solutions are short (mean 467 vs 676 tokens; truncation 18% vs 36%), and unmatched lengths re-import the finishing confound; (iii) format (boxed 82% vs 57%) and possible pretraining contamination of MATH solutions — residual, shared with TEACHF; disclose. PEER is teacher-free in strength, not external-free; word the claim accordingly.

**(4)** P(ICML method paper) ≈ 0.30, P(strong fallback mechanism paper) ≈ 0.55, remainder inconclusive — acceptable odds because both branches consume the same runs.
