# R1 (Fable), 2026-10-31

The brief's numbers check out against the named files; one understatement is flagged in (d).

## (a) C1, with the teacher as ceiling. Ranking: C1 > measurement paper > C3 > C2.

- **C3 is contradicted locally and saturated externally.** The human-prefix RL arm was null, >10 RL-time changes never met the criterion, and timing is irrelevant once the dose completes (HSELF beats TIME +1.02, Holm p=0.006, entirely pre-dose; post-dose gap −0.03 [−0.62,+0.57]). Zero-signal-triggered injection is also the most crowded 2026 design space (below).
- **C2 is an ablation, not a method.** CORR (+2.14, "curve raised") already establishes label-robustness; "unverified teacher SFT" is bounded by the teacher, cannot beat TEACHF by construction, and collapses into distillation prior art (2505.14216).
- **C1 fits the only discriminating evidence.** The sole baseline-beating registered contrast is content: TEACHF vs SELF +1.53 [95% +0.34,+2.76], all seeds. Self starts fade regardless of weighting (CWS vs UNI +0.21 [−0.30,+0.69]; all self arms end at Dr. GRPO's endpoint), and self gains vanish at 2,048 tokens (CWS +0.27, UNI −0.08, SELF −0.03) while TEACHF keeps +1.72 [+0.61,+2.83]. H25's gain migrates into c=0 (+0.86 early → +1.52, though "late" is step 50 only). So durable coverage appears to require solutions outside the base's sampled support; C1 is the only teacher-free source. Caveat: nothing yet shows the base can rationalize c=0 problems; that is the experiment, not an assumption.

## (b) Novelty (2026 items verified at abstract level only; full texts not checked)

Already covered — hints/off-policy injection during RL on hard or all-fail problems: QuestA 2507.13266 (partial solutions in prompts during RL, 1.5B, claims pass@1 and pass@k gains); BREAD 2506.17211 (expert prefixes when all rollouts fail); LUFFY 2504.14945 (off-policy teacher traces in GRPO); ReLIFT 2506.07527 (interleaved SFT on hardest questions); Hint-GRPO 2503.23905 (difficulty-adaptive hints); GHPO 2507.10628 (adaptive guidance injection); 2609.13997 (graded teacher hints on 128 unsolved problems, claims large-k boundary expansion); LSPO 2607.27787 (LoRA fitted on ground-truth solutions for zero-reward prompts, pass@k gains); DAHS+BHA 2604.07747 (hint synthesis against sharpening); MENTOR 2610.13594 (hints after failures); self-generated hints during RL: HDPO 2606.03021, SAGE 2602.03143. Cold-start SFT before RL is standard (DeepSeek-R1 2501.12948); OPD-before-RL warm starts: 2609.04108 ("sequential beats joint" — in tension with your OPD25 null; address it). Distillation expands pass@k where RLVR does not: 2505.14216. Diagnosis/frontier: invisible leash 2507.14843 (support shrinkage); RL squeezes, SFT expands 2509.21128; 2610.39295 (pass@1 gains carry a pass@k price). STaR 2203.14465 invented answer-conditioned rationalization. Closest match: SPLICE 2610.13996 — reference-solution-guided hint splicing into failed traces, then SFT; no pass@k/coverage claim, and it needs full reference solutions, not answer-only hints. I could not verify TailSFT 2608.25756 or BOPTR beyond your notes; I found no paper combining answer-only self-rationalization + pre-RL SFT + coverage at matched pass@1, but absence is not proven.

Defensible claim: "A ~25K-token answer-conditioned self-rationalization warm start — no teacher model, no human solutions — durably raises RLVR pass@16 at matched pass@1, concentrated on problems with zero base coverage, where equal-dose SFT on the base's own sampled solutions fades to zero; preregistered, with corrupted-answer, format, and prompt/length-matched self and teacher controls." Every qualifier is load-bearing: drop "durably," "matched pass@1," or "teacher-free" and QuestA/2609.13997/LSPO cover you.

## (c) Minimal decisive experiment (~$550, two weeks)

Arms (25K dose, P35 machinery, SFT seed 3030; reuse B1k, SELF, TEACHF trajectories):
1. **HINT** (3 RL seeds, 150 steps, ~$300): base conditioned on the reference final answer writes solutions for the P35 prompts plus c≤1 problems; keep finished ones; strip the hint before SFT. The "verified" filter is circular (the box echoes the hint); disclose it.
2. **HINT-CORR** (1 seed, ~$95): HINT data with corrupted final boxes. If corruption kills the gain, the carrier was the answer label (leakage), not reasoning.
3. **HINT-C0** screen (1 seed, ~$95): hinted solutions from c≤1 problems only — isolates reach into zero coverage.
Evaluate at 1,024 and 2,048 tokens, with stratum decomposition.

Primary contrast: HINT vs SELF trajectory (rule B, as E'): pooled ≥+1, 95% CI >0, all seeds positive, Holm. Secondary: HINT vs B1k (rule A); HINT vs TEACHF non-inferiority (within −0.5); descriptive c=0 contribution.

Abandon if: hinted generation yields usable solutions for <~half of c≤1 problems; HINT–SELF 90% CI inside ±1; the gain sits in c=1–8 early and fades by step 100–150 (reweighting signature); it vanishes at 2,048 tokens; or HINT-CORR ≈ 0 while HINT > 0. Also abandon the durability framing if tomorrow's 700-step references show TEACHF's gap itself decays.

## (d) Risks

- **Instrument support (serious).** Teacher-like starts end at pass@1 63–65, beyond B1k's range; steps 100–150 were already unsupported in P30. Register a fixed-step pass@16 secondary (Codex P38 point 4) and depend on the 700-step references.
- **Finishing/budget confound.** The brief understates this: self-start gains are ≈0 at 2,048 tokens, so the +1.8–2.5 "early self gain" is partly a 1,024-token truncation artifact. Any HINT claim must replicate at 2,048.
- **Leakage.** Hints are test-independent, but MATH train/test near-duplicates would convert hinted training text into memorized test answers — run n-gram dedup against the 511 before training; HINT-CORR covers label-style leakage.
- **Power/headroom.** pass@16 ceiling ~87–89; the movable strata are 51 (c=0) and 157 (c=1–8) test problems; CIs are ~±1, so a true +0.5 is undetectable at this budget.
- **Hardware/ops.** L40S references vs A100 arms (Codex flag); SageMaker duplicate-job billing if the watcher dies.
- **Novelty clock.** Several 2026 papers already claim hint-driven boundary expansion; only the matched-pass@1, teacher-free, durability-tested version remains — write it that narrowly.
