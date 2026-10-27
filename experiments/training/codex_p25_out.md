# Pre-registration 25: do different gradient sources share Dr. GRPO’s local tangent?

Declared October 27, 2026, before any run of this design.

**Question.** The matched-pass@1 null spans 1.5B at 512 and 1,024 tokens and 7B at 2,048 tokens. Rollout allocation supplied rare problems with 2.5× more successes without moving the curve; human-prefix guidance and the minimum-help pilot also failed to move it (Results 18, 20, 21, 23; P22). Check 17’s likelihood margins suggest transfer to rare problems, but do not measure causal changes in success. The missing comparison is between updates from different sources, starting from identical weights and taking equal-size steps.

**Hypothesis.** At a fixed checkpoint, donor updates produce
`d_perp = Δpass@16 − s × Δpass@1 ≈ 0`, where `s` is the local slope of B1k’s own trajectory. A second prediction is that rare-sourced updates produce no greater held-out rare-stratum success gain per unit parameter displacement than ordinary updates. These restrictions are tested separately: aggregate tangency can conceal offsetting stratum changes.

**Anchors and update construction.** Use B1k seeds 8–10 at steps 25 and 50, with 1,024-token responses. Replay the original B1k recipe through step 55, retaining FP32 anchors, trained optimizer states and neighboring checkpoints at 20/30 and 45/55; freeze these weights before donor generation. Each branch starts afresh from its anchor.

Construct the loss gradient `g` with `Runner` and `backward_weighted`, then apply exactly one controlled step,
`θ′ = θ − ρ g / ||g||₂`, over all trainable parameters. This tests gradient direction without introducing a fresh Adam state. Ordinary losses use sequence sums, advantages `r − mean(r)` within each group, and denominator `128 × 1,024`.

Choose each anchor’s radius using ordinary pilot gradients and the 255 validation problems only. From `{1, 2, 4, 8}` times the median nearby trained B1k step norm, choose the smallest radius producing 0.5–1 point of ordinary validation pass@1 improvement; otherwise declare calibration inconclusive. Freeze radii before evaluating donors on the 511 problems. Also evaluate every direction at `ρ/2`.

Retain FP32 weights through saving and evaluation; add an explicit FP32 option to `eval_vllm.py`, whose current BF16 conversion could obscure small updates. Verify displacement norm after reloading.

**Donors.** Four independent replicates per type and anchor, with fresh training rollouts and fixed generation seeds declared beforehand:

1. **Ordinary:** 16 uniformly sampled training problems × 8 on-policy answers.
2. **Rare:** the same recipe restricted to archived base counts `c ≤ 4/32`, including `c = 0`.
3. **Medium:** restricted to `5 ≤ c ≤ 16/32`.
4. **Human guidance:** rare groups using P22’s fixed human prefixes in 2 of 8 answers. Prefix fraction is `0.5(1 − t/100)`, cut before the first box; prefix plus continuation fits 1,024 tokens. Group rewards grade the combined answer; only continuation log-probabilities enter the loss.
5. **Minimum-help:** only the minimum-help loss on rare failures. Use P23’s up to four groups with at most one current success, four failed prefixes per group, prefix length at most 256 before the box, hints of 0/32/64 tokens and two continuations per level. Values are 1, 2/3, 1/3 or 0 according to the smallest successful hint level, centered within problem. Credit only the original failed prefix.
6. **Random:** an independent isotropic Gaussian parameter direction, normalized to the same radius.

Rare, guided and minimum-help replicates share prompts and initial unassisted answers, drawn from rare problems with retained, reverified human solutions. Report archive exclusions and stratum counts. Zero gradients, no guided successes, or no contrasting minimum-help values are failed interventions; report them and mark that type inconclusive, without redrawing until it works.

**Evaluation and estimands.** Evaluate unassisted on `data/eval_frontier_512.json`—511 problems—with `eval_vllm.py`, 32 samples, T = 1 and 1,024 tokens. Use common evaluation seeds and problem order across anchors, branches and reference knots, with distinct output directories. Compute pass@k as the mean of `1 − choose(32 − c, k)/choose(32, k)`. Freeze test strata from the independent base-model 32-sample evaluation at 1,024 tokens: 0, 1–4, 5–16 and >16 successes. Report both endpoint changes in every stratum, lengths and truncation.

Estimate each seed’s slope from its bracketing B1k checkpoints:
`s = (P16[t+5] − P16[t−5]) / (P1[t+5] − P1[t−5])`.
Report raw `d_perp` and standardized `D = d_perp/(m/1 point)`, where `m` is ordinary mean pass@1 movement at radius `ρ`. Thus thresholds express displacement per one point of ordinary progress; near-zero donor movement remains well-defined. Rare selectivity is the rare-source minus ordinary difference in `ΔP1[c≤4]/ρ`.

**Estimator and guards.** Use 2,000 shared problem bootstrap draws, also resampling paired donor replicates; recompute slopes and normalization in every draw. Average replicates within seed, weight seeds equally, and assess the two checkpoints separately. Intervals condition on these trained seeds.

Require positive, identifiable slope denominators; ordinary test movement at least 0.25 point with its 90% interval above zero; both checkpoints eligible in every seed; and post-update pass@1 inside the local reference range. Eligibility is fixed on full data. Following P21, more than 10% of bootstrap points outside that range in any seed makes the comparison uninterpretable. Missing evaluations are “incomplete evaluation.” The locality gate requires the 90% interval of `(d_perp[ρ] − 2d_perp[ρ/2])/(m/1 point)` within ±0.5. Truncation above 20% qualifies the verdict.

**Decisions.** Support the joint hypothesis only if every type at both checkpoints has pooled and per-seed 90% intervals for `D` inside [−1, +1], and every rare-source selectivity contrast has a 90% upper bound ≤0. Falsify tangency if any type has mean `D ≥ +1`, a 95% interval above zero and agreeing seed signs. Apply Holm across 12 transverse tests and six rare-selectivity tests. A significant positive selectivity contrast rejects the transfer prediction even if aggregate tangency holds. A significant negative transverse departure rejects strict tangency without establishing improvement. Remaining outcomes are inconclusive.

**Meaning for the paper.** Support supplies causal evidence for a local constraint on the tested gradient sources and weakens a starvation-only explanation. Positive transverse movement shows local opportunity missed by the long-run interventions; an ordinary-update departure implicates optimizer geometry. Selective rare improvement with zero aggregate displacement indicates compensating losses elsewhere. These findings concern the tested directions and checkpoints, rather than a universal parameter-space constraint.

This connects to cross-prompt interference: shared parameters permit hard-prompt reweighting to rotate updates (2602.21189). ([arxiv.org](https://arxiv.org/abs/2602.21189)) GradAlign selects problems by alignment with validation gradients (2602.21492); this experiment asks whether directional differences improve the matched-accuracy frontier. ([arxiv.org](https://arxiv.org/abs/2602.21492))

**Cost.** There are 144 donor directions and 288 branch evaluations, plus six anchors and 12 neighboring references: 306 full-test evaluations. At approximately 13 minutes per 1.5B evaluation, these cost 66 A100-hours. Allow 10–15 more for replay, validation calibration and donor construction: approximately 80 A100-hours at that rate, or 150 if FP32 doubles evaluation time.