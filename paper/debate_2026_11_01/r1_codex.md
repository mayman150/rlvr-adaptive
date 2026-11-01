1. **Diagnosis.** Yes: a scoped empirical null has become a restriction on method search. Matched accuracy remains useful: higher pass@1 and pass@k at fixed compute can reflect accuracy improvements alone. But pass@k measures the distribution of success across problems; it does not count distinct correct reasoning strategies. [Metric definitions](/Users/mmayman/Desktop/adaptive-rlvr/docs/brief/brief.tex:34).

   Verified corrections: CWS–UNI is **+0.03 pp [−0.37, +0.44]**, but fails support checks; equivalence is unestablished. TEACHF–Dr. GRPO is **+2.10 [0.86, 3.47]** over eligible steps 1–50; TEACHF–SELF is **+1.53 [0.34, 2.76]**. These establish supported-window effects, not late matched-accuracy persistence. Fading self-SFT offsets could mean the reference catches up rather than successes being forgotten. [P36](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/results_p26/p36_analysis.txt:46), [checkpoint details](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/results_p26/p36_phase2_detail.txt:6).

   Claude also overstates signal scarcity: at \(p=1/8\), eight rollouts find a success with probability \(1-(7/8)^8=65.6\%\). Larger groups trade prompt breadth for repeated attempts. Increasing evaluation k can worsen saturation; harder problems are essential.

2. **Question and thesis.** Can shared RL updates suppress rare successes, and can targeted constraints prevent that?

   **My bet:** Constraints on rare success probabilities improve unseen hard-problem pass@128 at equal total compute, with a supported matched-accuracy gain and no material pass@1 loss.

3. **Method and novelty.** Use **RL-side constraints**. Identify low-success training prompts from earlier observations. Collect independent, fixed-size probes; retain reward labels and behaviour log-probabilities. Estimate correctness probabilities using importance ratios. Project the proposed Adam update to limit estimated probability declines on a rotating rare-prompt subset, using independently refreshed probability floors. Refresh probes when importance weights lose effective support. Count probing, scoring and projection costs.

   This protects estimated total correctness probability. Fresh samples must validate retention; preserving stored-trace likelihood alone is insufficient. Transient self-SFT gains motivate the hypothesis but do not establish its mechanism. Compare against ordinary replay using identical samples and comparable scoring costs.

   Claude’s adaptive-groups/pass@k combination substantially overlaps **DARS** and **Pass@k Training**. ([arxiv.org](https://arxiv.org/abs/2508.13755), [arxiv.org](https://arxiv.org/abs/2508.10751)) **MCPO** constrains drift on mastered prompts; **UCPO** redistributes mass among correct solutions; **GEM** already projects updates to prevent interference. The candidate contribution is uncertainty-aware constraints on *rare-prompt total success probability*. That distinction needs technical justification and empirical ablation before a novelty claim. ([arxiv.org](https://arxiv.org/abs/2604.16972), [arxiv.org](https://arxiv.org/abs/2605.00365), [arxiv.org](https://arxiv.org/pdf/1706.08840))

4. **Setting and metrics.** Start with Qwen2.5-Math-1.5B and MATH training, using an identical fixed difficulty mixture across arms. Target at least 400 audited, deduplicated English, text-only, open-ended OlympiadBench math questions; retain the original 511 MATH questions for continuity. OlympiadBench provides these categories and automated scoring for open-ended questions. ([github.com](https://github.com/OpenBMB/OlympiadBench))

   Train initially at 2,048 tokens; evaluate at common 2,048- and 4,096-token caps. Require usable validation headroom at pass@128 before committing. Freeze choices without inspecting confirmatory test outcomes.

   Draw 256 samples per problem; report pass@1/16/64/128. Primary: terminal pass@128 at equal total training compute, with a one-point pass@1 noninferiority margin. Report active GPU time, generated tokens and updates, including preparation. Require supported matched-accuracy gains for the coverage-diversity claim. Audit correct-solution strategy variety separately. Replicate on SmolLM3-3B after the first model passes.

5. **Experiments and gates.** Costs below are planning caps, contingent on measured throughput and packed jobs.

   - **Week 1, ≤$100:** Independently resample rare training/probe problems at existing SELF and GRPO checkpoints. Establish actual probability loss and whether ordinary updates predict it. Stop this mechanism if fading offsets are only reference catch-up. Audit prior-work overlap.
   - **Week 2, ≤$400:** Dr. GRPO, DARS, constraints, DARS-plus-constraints; two seeds, equal compute corresponding to 50 baseline steps. Advance only with measured retention improvement, ≥3 pp validation pass@128 gain over the strongest control in both seeds, and ≤1 pp pass@1 loss. This screens candidates; it does not confirm the thesis.
   - **Weeks 3–5, ≤$1,500:** Three paired seeds at 150-step-equivalent budgets. Include fresh Dr. GRPO, the strongest published control selected on validation, matched replay and removal of the constraint. Require terminal gain ≥3 pp, Holm-adjusted 95% interval above zero, every seed positive, and pass@1 noninferiority. Unsupported terminal matching cannot establish diversity.
   - **Weeks 6–9, ≤$1,500:** SmolLM3 replication, 300-step persistence on the first model, and decoding-budget sensitivity. Conditional total cap: **$3,500**; reserve remaining time for analysis and writing.

6. **Kill and fallback.** Kill if constraints preserve their estimator without improving fresh-sample success, gains remain on training prompts, longer decoding removes them, or the method is credibly equivalent to the strongest control within ±1 pp. Wide intervals and missing support remain inconclusive.

   Fallback: a bounded empirical paper separating discovery, retention, across-problem coverage and correct-strategy variety, with teacher starts as positive controls. Claim neither an immutable frontier nor that external information is necessary.