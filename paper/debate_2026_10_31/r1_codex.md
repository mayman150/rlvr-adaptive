(a) **None currently supports a novel, durable method claim. I would pilot C1, with low confidence that it beats a matched teacher.** Answer conditioning can supply traces outside the ordinary self-solution pool. That is more plausible than further reweighting existing successes, given CWS–UNI’s near-zero interim difference.

The strongest evidence is the observed-window TEACHF–SELF difference: +1.53 pp [0.34, 2.76]. But teacher durability remains unmeasured: H25’s “late” stratum decomposition contains only step 50 across three seeds, and teacher checkpoints at 100–150 exceed the short reference’s accuracy support. [Results](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/results_p26/p36_analysis_partial_20261031.txt:57), [strata](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/results_p26/p36_interim_strata.txt:8).

C2 has a reasonable efficacy prior if teacher reasoning remains mostly correct. CORR does **not** justify arbitrary unverified traces: it corrupted the final box of previously verified solutions while retaining their reasoning. “Verification-free distillation” is also more accurate than “label-free.” C3 is lower priority: the prefix null and post-dose TIME result weaken its motivation, although uniform interleaving does not rule out adaptive failure targeting.

(b) The relevant literature substantially covers these recipes:

- **STaR** already generates answer-conditioned rationalizations for failed questions, then fine-tunes with the hint removed. **ExPO** incorporates answer-conditioned self-explanations into GRPO and reports superiority over expert-CoT guidance. C1’s core mechanism is established. ([arxiv.org](https://arxiv.org/html/2203.14465v2), [arxiv.org](https://arxiv.org/html/2507.02834v2))
- **QuestA** supplies partial solutions through prompt augmentation; **BREAD** branches rollouts from expert prefixes; **Hint-GRPO** adaptively supplies hints. Scaffolding difficult groups is established. ([arxiv.org](https://arxiv.org/html/2507.13266v1), [arxiv.org](https://arxiv.org/html/2506.17211), [arxiv.org](https://arxiv.org/html/2503.23905v2))
- **LUFFY** mixes off-policy demonstrations with student rollouts. **ReLIFT** identifies all-failure questions and interleaves targeted SFT on externally supplied solutions—essentially C3. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2025/file/a9d5c33e38442450f9781bd7bad3c81e-Paper-Conference.pdf), [arxiv.org](https://arxiv.org/html/2506.07527v2))
- **Learning from Failures** trains feedback-conditioned retries, which differ from independent rollouts. Cold-start SFT and OPD preparation already have direct comparisons; **IDA-OPD** explicitly targets pass@k while broadly maintaining pass@1. ([arxiv.org](https://arxiv.org/html/2503.04808v1), [arxiv.org](https://arxiv.org/abs/2609.28145), [arxiv.org](https://arxiv.org/abs/2608.29846v1))

A precise **empirical** claim worth testing is: *answer-only rationalization produces coverage gains that survive ordinary RL and exceed matched self and teacher starts at equal supervision dose*. That outcome alone would not establish method novelty.

A plausible C4 technical contribution is **selecting supervision by coverage retained after RL**: score candidate trace batches using short Dr. GRPO continuations on disjoint training probes, then freeze the selector before test evaluation. Fading self gains motivate this objective; they do not establish that the selector works. It would need an ablation against uniform rationalization and static hardness weighting.

(c) Spend approximately $600 on a two-week **C1 screen**, using four arms × three seeds × 150 steps:

1. Answer-conditioned student rationalizations on a frozen, rare-success training subset.
2. Verified teacher traces on exactly those prompts.
3. Ordinary SELF.
4. Fresh Dr. GRPO.

Use the existing teacher archive where possible. Remove hints from SFT inputs and all evaluation prompts; initially exclude unverified traces. Match C1 and teacher arms on actual supervised tokens, updates, prompt visits, response lengths and terminal formatting. Pair independent SFT seeds and RL seeds across arms. SELF cannot be fully prompt-matched where no ordinary self-solutions exist; that comparison tests recipes, while C1–teacher better isolates source.

**Primary contrast:** C1 minus the same-seed teacher trajectory at matched pass@1, averaged over fixed checkpoints 100 and 150, with required step-150 confirmation. Advancement additionally requires ≥2 pp versus Dr. GRPO and ≥0.5 pp versus both SELF and teacher, positive 95% intervals and every seed positive. Freeze one Holm family covering the late-window and terminal tests. These are minimum point estimates, not demonstrated lower bounds on those margins.

Require every terminal checkpoint to have support in every seed. Do not let eligible early checkpoints substitute for missing late comparisons. [Estimator](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_36.md).

Abandon the C1 paper direction if gains are transient, or it is credibly equivalent/inferior to matched teacher supervision. Wide intervals mean inconclusive. Passing this screen warrants longer runs and a technical-method ablation; it does not establish 700-step durability. **The stated budget cannot buy a fully controlled long-duration study.**

(d) The main risks are:

- **Accuracy support:** tomorrow’s longer Dr. GRPO references may still never reach the treatment’s pass@1. SELF also needs sufficient support. No extrapolation or raw endpoint pass@16 can rescue the coverage claim.
- **Hardware:** A100 treatments against historical L40S references weaken attribution. Keep fresh comparisons on identical hardware, software, precision and checkpoint handling; historical extensions remain provisional. [Review](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/codex_p38_review.md).
- **Finishing:** substantial truncation makes formatting consequential. Retain 1,024-token primary evaluation and add a common 2,048-token diagnostic.
- **Hint circularity/contamination:** answer agreement does not verify a hinted derivation. Blindly audit reasoning; remove exact and near train–test duplicates.
- **Interpretation:** 0/64 means sampling-zero, not impossible. Wrong-answer entropy is not useful coverage. The problem bootstrap conditions on the trained runs; three seeds do not measure pool-selection uncertainty.