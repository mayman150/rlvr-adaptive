# R2 (Fable), 2026-10-31

**Correction accepted.** My round-1 claim that the early self gain is "partly a 1,024-token truncation artifact" was unsupported: the 2,048-token checkpoint evaluations cover steps 50–150 only, where self gains were already ~0 at 1,024. One caveat to the coordinator's note: the cited positive 2,048 start offsets (H25/R25, +1.5 to +3.3; also TEACHF +3.16, CORR +2.64) are all teacher-sourced starts; SELF's own 2,048 start offset is "n/a (outside support)" (p35_analysis.txt). The early self window at 2,048 is unmeasured, not refuted. Cheap fix: eval-only passes of existing SELF/CWS checkpoints 1–25 at 2,048, support permitting.

## 1. Novelty

ExPO 2507.02834 is verified and decisive: it generates positive samples by conditioning on the ground-truth answer, targets problems where the model produces no correct solutions, uses them during RL post-training (GRPO/DPO), and reports superiority over expert-CoT guidance. With STaR 2203.14465 (same data recipe, hint stripped before fine-tuning), SPLICE 2610.13996, and LSPO 2607.27787, C1's mechanism is established. Worse, your own TIME result (before-RL vs interleaved equal once the dose completes) removes "warm start vs during-RL" as a claimable delta against ExPO. **A C1 method-novelty paper is not defensible at ICML.** What remains defensible is the empirical claim Codex stated and I co-sign: whether answer-only rationalization yields coverage gains at matched pass@1 that survive RL and track the teacher at equal dose — with C1 as the tested instantiation, not the contribution. No published work I could verify measures hint-derived supervision on the matched-pass@1 frontier with durability criteria; that measurement is the paper.

## 2. Codex's C4 (retention-based supervision selection)

Feasible but weak as a core. Cost per candidate batch: SFT is negligible; a 25–50-step Dr. GRPO continuation is ~$17–35/seed; probe coverage eval (32–64 samples on a few hundred disjoint problems) ~$10. So ~$30–50 per candidate-seed; a 6-candidate, 2-seed screen fits in ~$500. Three problems: (i) the selection signal (differences in retained coverage) must exceed ~±1 pp noise at 1–2 seeds, while the largest known content effect is +1.53 over three seeds; (ii) the evidence shows near-zero variance within the self family (CWS–UNI +0.21; CWS–HSELF −0.31), so the selector has nothing to rank unless candidates span content sources — where the answer (teacher > self) is already known; (iii) proving the probe ranking predicts 150-step outcomes is itself a validation study. Verdict: a protocol, not a method; register it as a secondary diagnostic inside the C1 screen, not the paper's core.

## 3. Agreed experiment (~$600, two weeks)

- **RAT** (3 RL seeds, 150 steps, ~$300): answer-conditioned self-rationalizations, 25K dose, on a frozen rare-success subset restricted to prompts with archived teacher solutions; hints stripped; verified-box only (circularity disclosed).
- **TEACH-M** (3 seeds, ~$300): teacher traces on exactly those prompts, matched on tokens, updates, visits, lengths, terminal format.
- Reuse SELF, TEACHF, B1k, and the 700-step references; eval-only 2,048 passes.
- **Primary family (Holm):** G1 RAT vs B1k, late checkpoints 50/100/150 with required step-150 confirmation and per-seed support (rule A, ≥+2); G2 RAT vs SELF trajectory ≥+0.5, 95% CI >0, all seeds.
- **Abandon if:** generation yields <2 usable rationalizations per problem over most of the c≤1 subset; RAT–SELF late window inside ±1 or negative; the gain sits in c=1–8 early and fades; or step-150 support is unavailable even with long references.

Residual disagreement, one line each. Fable: G3 should be non-inferiority to TEACH-M (95% lower bound > −1); teacher-free parity is the claim. Codex (r1): RAT must exceed the teacher by ≥0.5.

## 4. Fallback paper

Strongest option with existing evidence plus ~$600 (P38 Part B, fixed per Codex's review, alongside the running 700-step references): a preregistered measurement paper. One-sentence claim: "Small-dose supervision moves RLVR's pass@k-at-matched-pass@1 frontier durably only when its solutions lie outside the base model's sampled support — over ten RL-time interventions and all self-derived warm starts, regardless of targeting or timing, return to the Dr. GRPO frontier, while teacher-sourced starts persist and concentrate their gain on zero-coverage problems."
