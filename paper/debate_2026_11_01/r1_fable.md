# R1 (Fable), November 1, 2026

## 0. Numbers

The coordinator's figures reproduce from the named files: TEACHF +2.10 (Holm p=0.002, curve raised), CORR +2.14; TEACHF−SELF +1.53, 95% [+0.34, +2.76]; CWS−UNI +0.03; teacher step-150 pass@1 63–65 vs B1k 57.5–58.1; pass@16 86–89, c≥9 stratum 99.3–99.9; at 2,048 tokens self arms ≈0 vs TEACHF +1.72 [+0.61, +2.83]. Also verified: deep sampling (256 extra draws on 232 c≤4 train problems) solves 53% of never-solved problems, mean success rate 0.93% (full_story.tex:658).

## 1. Diagnosis

The findings are sound; two inferences are not.
(a) The instrument critique is correct: the support rule discards exactly the teacher arms' steps 100–150 ("beyond B1k's support", p30 file), so durability of the only curve-raising effect is unmeasured by construction. Headroom and the 1,024-token confound are real.
(b) But the coordinator's cure is already published, four times. "Adaptive rollout allocation + pass@k credit" is DARS + Knapsack RL + Pass@k Training; since September 2026, DATPO combines difficulty-adaptive rollouts with a diversity advantage explicitly to expand pass@k. Betting 11 weeks and ~$2K on winning that race at 1.5B/MATH is bad risk allocation. Allocation also has a tested in-house null ("sample allocation to uncertain problems"), and DARS itself reports plain rollout scaling does not help.
(c) What the combined evidence isolates: the only replicated frontier move is content outside the policy's sampled support (teacher); every in-support self start fades regardless of weighting or timing. The untested cell is content outside the 64-sample support but inside the model's support. That cell is the user's question in testable form, because rollouts are precisely a way to extract such content.

## 2. Question and thesis

Q: Is RLVR's coverage leash the policy's *sampled* support (fixable with sampling compute) or the *model's* support (needs a teacher)?
Thesis: **Rollouts help RLVR diversity as data, not as gradient: verified solutions harvested by deep, difficulty-allocated sampling from the base on zero-coverage problems, distilled in a ~25K-token pre-RL SFT, durably raise pass@k at matched pass@1 as teacher content does — while rollout-side levers (bigger/adaptive groups, pass@k credit, entropy control) only move along the frontier.**

## 3. Method: HARVEST

Allocate up to 512–2,048 base samples per train problem ∝ 1/p̂ on c≤1-of-64 problems; keep verifier-passed solutions (dedup against test; audit for lucky-guess answers — the 0.93% success rate warns these are extreme-tail draws); 25K-token SFT; unchanged Dr. GRPO; optional mid-RL re-harvest. Why it should work: yield is proven (53% of never-solved problems); teacher evidence shows out-of-sampled-support content persists (curve raised, survives 2,048, concentrated on c=0: 27–33 vs 15–24) exactly where in-support self-SFT fades. Novelty: STaR/RFT do rejection-sampling SFT but not deep allocation onto zero-coverage problems, and never evaluate frontier movement or durability; BroRL spends deep rollouts as gradient; LSPO re-rolls with an adapter during RL. The RL-side levers (pass@k advantage + Clip-Cov or temperature) run as a preregistered falsification arm: I rate <25% that one moves the frontier, but if it does, the paper flips to it — validated on a matched-accuracy instrument none of those papers has.

## 4. Setting and metric

Keep Qwen2.5-Math-1.5B (references reusable); 2,048 tokens primary (base truncation 36%→18%). Redefine strata by 256 base samples. Primary reading: joint (pass@1, pass@64) at fixed step and fixed total-rollout compute, on hard strata plus a ≥500-problem harder set (OlympiadBench-style, verifiable answers) for headroom; the matched-pass@1 frontier remains the mechanism diagnostic where supported — fixing the conditioning blind spot without discarding the asset. Replicate on SmolLM3-3B (viable per smollm3_viability.txt).

## 5. Stages

- **Stage 0 (week 1, ~$75, eval-only, decisive).** (i) 256-sample re-eval of existing step-150 checkpoints (B1k, SELF, CWS, TEACHF, 1,000-step) on hard strata at 2,048 — did any RL null hide a large-k effect? (ii) Run the harvest; measure yield and audited quality. Gate: usable verified solutions on ≥40% of c=0 train problems.
- **Stage 1 (weeks 2–3, ~$400).** HARVEST ×2 seeds; pass@k-advantage ×2 seeds; 150 steps; reuse B1k/SELF/TEACHF. Gates: HARVEST−SELF ≥ +1, all seeds positive; pass@k arm vs B1k ≥ +1 hard-set pass@64 at fixed compute without pass@1 loss.
- **Stage 2 (weeks 4–6, ~$800).** Winner to 3 seeds, confirmatory preregistration; corrupted-label control; token/update-matched TEACH-M; long references for support.
- **Stage 3 (weeks 6–8, ~$400).** SmolLM3-3B; 500-step persistence, 1 seed. Writing weeks 8–11. Total ≤ $1,700.

## 6. Kill and fallback

Kill: harvested content fades like SELF by step 100–150, vanishes at 2,048, or fails the yield/quality gate → the leash is model support, not sampled support. Fallback (~70% already evidenced): the mechanism paper — "at matched accuracy, ten RL-side levers and all in-support self starts return to the Dr. GRPO frontier; only content beyond the model's sampled reach moves it, durably, concentrated on zero-coverage problems" — with Stage-0 large-k re-evals closing the "k=16 was too small" loophole.

## 7. Literature (abstracts verified today unless noted)

2510.01180 BroRL: hundreds of rollouts/example revive ProRL-saturated models; no matched-accuracy control. 2508.10751 Pass@k Training: pass@k as reward, analytical advantage. 2505.22617: entropy collapse law R=−a·e^H+b; Clip-Cov/KL-Cov restrict high-covariance-token updates. 2509.06941 outcome-based exploration: tested in-house, equivalent. 2509.25849 Knapsack RL: rollout budget as knapsack, up to 93 rollouts on hard tasks, +2–4 pass@1; no frontier reading. 2508.13755 DARS: extra rollouts on hard problems raise pass@K; plain rollout scaling does not. 2609.08650 DATPO (9/2026): difficulty-adaptive tree rollouts + sibling-diversity advantage to expand pass@k — closest prior to the coordinator's thesis. 2504.13837: base overtakes RL at large k. 2507.14843 invisible leash: observed support shrinkage outweighs expansion (hedged). 2509.21128 RL-squeezes/SFT-expands (ICLR 2026): trajectory clustering, no pass@k. Listing-level only: 2606.06080 max@k advantage estimators; 2605.07689 sign advantage = first-success credit; 2606.25451 pass@k-weighted single rollout. Could not verify: TailSFT 2608.25756, BOPTR 2609.01274 beyond prior notes.
