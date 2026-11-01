I would pursue a **science paper for NeurIPS 2027**, centered on finite-budget acquisition and retention of reasoning. A method paper should follow only if the mechanism analysis supports an intervention beyond existing curricula or novelty bonuses.

I read the named files, checked the cited primary arXiv texts, inspected the trainer and Stage 0 plan, and independently recomputed the late matched-accuracy point estimates.

**1. Research question and defensible thesis**

The single question should be:

> Under a fixed acquisition and training budget, when does teacher reasoning produce persistent, reasoning-valid coverage gains over deep self-sampling, and how does delivery affect those gains?

This supports several informative outcomes:

- **Deep self-sampling matches teacher guidance:** the observed teacher advantage reflects acquisition efficiency within the tested budgets.
- **Teacher remains better on identical problems:** its traces contain useful structure that the student’s tested search procedure does not efficiently recover.
- **Either source succeeds only with guidance during RL:** delivery governs consolidation.
- **Early gains disappear after withdrawal:** acquisition improves initial coverage, but subsequent optimization erodes or catches up with that advantage. Distinguish those mechanisms directly.
- **Teacher gains come entirely from additional covered training problems:** the advantage concerns acquisition coverage rather than better content on shared problems.

The contribution would be a controlled decomposition of **acquisition coverage, trace content, delivery, and persistence**. It should not depend on finding a winning cell.

Scope the conclusions to sampling procedure, token budget, training duration, and model families. “Zero-observed-success” describes evidence under a protocol; it does not establish mathematical absence of support.

Also narrow the current paper’s language. At matched pass@1, pass@128 measures how success is distributed across problems. Strategy diversity requires additional measurements. Call the measured effect displacement from the Dr. GRPO reference curve; that curve has not been established as a globally optimal frontier.

The existing persistence evidence motivates this question but does not settle it. My recomputation gives step-150 gaps of **+0.474 points for TEACHF and +0.469 for SELF**. TEACHF’s individual-seed gaps are approximately +0.79, +1.03, and −0.40. The draft correctly labels the analysis descriptive and reports failed support for some registered extensions. [Current draft](/Users/mmayman/Desktop/adaptive-rlvr/paper/icml/paper_v6.tex:260)

**2. Priorities and critique of the 2 × 3 design**

My priorities would be:

| Direction | Decision |
|---|---|
| 1. Deep self-sampling versus teacher | Essential |
| 2. Persistence after guidance ends | Essential |
| 3. Reasoning-valid coverage | Essential |
| 4. Structured exploration | One bounded audit; broader comparisons optional |
| 5. Reasoning-distribution mechanism | Essential in a small form; exhaustive clustering optional |

Direction 4 becomes essential to any claim that external information is necessary. A narrower acquisition-and-retention paper can make conditional claims about its tested self-search procedure without reproducing every exploration method. Repeating the existing UCB-Con study adds little.

Start with a **2 × 2 core**, plus plain Dr. GRPO:

- Source: audited deep-sampled self traces or teacher traces.
- Delivery: SFT then RL, or adaptive guidance during RL followed by complete withdrawal.

The coordinator’s third column combines different mechanisms. KL to the SFT start constrains departure from that distribution; 2605.29190 rewards correct trajectories surprising under the SFT reference. Those should not be treated as interchangeable retention treatments. Add one after diagnosing what actually declines.

The factorial needs several corrections:

- **Separate content from acquisition coverage.** Compare sources on the intersection of problems with valid self and teacher solutions. Separately compare whole acquisition pipelines, including teacher-covered problems lacking harvested self solutions. Conditioning on successful harvest selects recoverable problems.
- **Match banks and account for exposure.** Hold problems, trace multiplicity, length distributions, and presentation budgets fixed. Log unique reasoning tokens and repeated exposures. Identical banks alone do not control how often guidance is consumed.
- **Control visible content.** Adaptive hints expose partial traces; full-trace SFT exposes more information. If attributing an effect specifically to curriculum scheduling, include SFT on the corresponding visible-prefix bank.
- **Strengthen answer controls.** CORR preserves intermediate answers, so it cannot establish that answer information is irrelevant. Use audited pre-answer traces with explicit target-answer occurrences removed, and an answer-only control. Report example-matched and token-matched readings where their lengths differ.
- **Separate diagnostic and practical comparisons.** Match accuracy for reference-curve displacement; match total acquisition-plus-training compute for usefulness.

Use the 64/256/1,024 sampling depths first as acquisition curves. Turning every depth into every delivery arm would expand the study before establishing usable yield.

**3. What would actually be new**

The novelty is the controlled scientific comparison, not teacher curricula, prefix conditioning, warm-start forgetting, or a generic diversity reward.

| Prior work, checked against primary text | Remaining contribution |
|---|---|
| **MFC, 2609.13997:** adaptive per-problem teacher hints, with a success-only monotone frontier | Source-controlled comparison, complete withdrawal, extended retention, and matched-accuracy displacement |
| **Curriculum RL, 2606.22317:** three rounds of boundary sampling, teacher-SFT bridges, and RL | Separate acquisition coverage from content and examine persistence after the final bridge |
| **AdaBack, 2506.18110:** reward-adaptive partial-rationale curriculum | Controlled source and exposure contrasts; merely withdrawing prefixes is already covered |
| **PrefixRL, 2601.18795:** self-rejection-sampled prefixes, SFT+RL baselines, acquisition compute accounting, held-out pass@k | Reasoning-valid, matched-accuracy, source-controlled persistence |
| **DATPO, 2609.08650:** adaptive trees, sentence-entropy forks, sibling-diversity advantages | Determine whether its coverage advantage survives accuracy matching and extended training |
| **2605.29190:** frozen-SFT-reference novelty reward preserving exploratory primitives | Test acquisition source and transfer/retention under controlled exposure; adopting its bonus is a baseline |
| **Quagmires, 2510.01624:** extensive multi-family SFT–RL study and pre-RL predictors | Identical-problem source contrasts and supported longitudinal measurements |
| **No Free Lunch, 2610.39295:** shared-strategy tilting theory with explicit specialization assumptions | Test where its assumptions describe the empirical setting and where delivery changes the represented behavior |

Two corrections matter.

First, PrefixRL already reports SFT+RL entropy collapse and declining pass@64 after roughly 100 RL iterations. The gap is therefore narrower than “persistence has not been studied”: it concerns the controlled, matched-accuracy, reasoning-valid experiment proposed here.

Second, No Free Lunch does not establish an unconditional impossibility result for arbitrary neural-network training. Its per-prompt tradeoff requires specialization or the corresponding positive-price condition; aggregate pass@k reversal needs an additional fine-coverage condition. A reference-curve shift would require examining those assumptions, not claiming to refute the theorem.

Avoid field-wide “first” claims for accuracy matching, preregistration, or persistence without a systematic search. Preregistration strengthens this study’s evidence; it is not sufficient novelty by itself.

**4. Staged experiment and budget**

Use the frozen OlympiadBench/AIME validation and test sets, with MATH-511 for continuity. Freeze a common training pool and model-specific rarity cohorts using the actual sampling budget. Reuse teacher archives where coverage permits; regenerate only missing traces. The original 109-problem cohort is a pilot, not automatically the population for confirmation.

Use Qwen2.5-Math-1.5B and SmolLM3-3B-Base. Choose response budgets after validation measurements of headroom, truncation, guided reward variance, and memory. Qwen’s published configuration allows **4,096 total positions**; reserve space for questions and hints. The current evaluator rejects requests exceeding that limit. A 4,096-token response cap is therefore not a valid Qwen setting without an explicitly validated context extension. [Evaluator](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/eval_vllm.py:83)

Before Stage 0 proceeds, amend it to:

- Correct the current plan to **1,626,624 completions and a 3,106,406,400 new-token ceiling**, before additions.
- Screen both families on hard validation at feasible budgets; use the agreed 10–70% base pass@128 headroom gate, plus truncation and reward-variance gates.
- Re-measure rarity at the chosen budget; preserve the historical 1,024-token labels separately.
- Add TEACHF-start and TEACHF-RL retention probes. The registered probes currently cover SELF and base.
- Preserve the complete 1,024-attempt harvest pilot and charge failures.
- Treat the 44-problem harvest gate as feasibility evidence, not a power calculation.
- Put training calibration in a separate registration: Stage 0 currently promises generation/evaluation only.

The checked frozen hashes match the report. [Registration](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_39.md) · [Job plan](/Users/mmayman/Desktop/adaptive-rlvr/experiments/sagemaker/plans/p39_stage0.json)

| Stage | Experiment and decision | Planning allowance |
|---|---|---:|
| 0 | Amended census, harvest, headroom, retention probes; separate calibration | $400–700 |
| 1 | Qwen: five core arms, one seed, 150 steps; Smol smoke runs; bounded exploration audit | $300–600 |
| 2 | Qwen: five arms × three independent training seeds, 700 steps; plain references to 1,200 | $4,500–5,500 |
| 3 | Smol: plain plus both sources under the validation-selected delivery; three seeds, 700 steps; references to 1,200 | $5,500–7,000 |
| 4 | Confirm necessary content/exposure controls and reasoning-valid mechanism results | $1,500–3,000 |

These are **conditional planning estimates**, not measured quotes. Using the supplied $32/hour instance rate gives $4 per occupied A100-hour. Assuming four minutes per Qwen step and eight per Smol step at 2,048 tokens, Stages 2 and 3 require **$3,200 and $4,160 in training alone**. The allowances add evaluation and imperfect packing. Longer budgets require recalibration.

Thus $5K is a sensible initial tranche through first-family confirmation. It does not fund this complete program. More calendar time does not remove that constraint.

For confirmation:

- Use three paired, independent training seeds, varying SFT shuffle and RL randomness; decide any expansion to five from validation-based power analysis before test access.
- Use a frozen trace bank and a fixed exposure cap. End all guidance by step 200, leaving 500 unaided RL steps.
- Initially screen SFT dose at 25K and 250K presented tokens; freeze the dose on validation.
- Save dense checkpoints; conduct expensive sealed-test evaluation at registered late points, including steps 400, 550, and 700, with 512 samples/problem.
- Report pass@1/16/64/128/256, total compute, truncation, and reasoning-valid coverage on a frozen subset.
- Require a supported late effect, all-seed reporting, and replication. For the reference-curve claim, retain the ≥2-point relevance threshold and ±2-point equivalence margin; report power separately.
- Freeze checkpoint eligibility and multiplicity rules. An unsupported or wide-interval null remains inconclusive.

Finish confirmation by March; reserve April for mechanism controls, analysis, and writing.

**5. Biggest risks and week-one remedies**

**Curriculum fidelity.** The trainer currently implements linear token-prefix withdrawal, not MFC. MFC samples a guidance length uniformly from each problem’s current frontier, lowers that frontier after sufficient success, and presents hints while the student generates a complete solution. The current code instead generates ordinary rollouts and replaces selected slots with prefix continuations, charging extra generation and potentially mixing conditioning states within a group. [Prefix implementation](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/train_grpo.py:626)

Implement the curriculum explicitly on the common Dr. GRPO core. Check per-problem state transitions, resume behavior, context masking, group baselines, answer leakage, and guided/unguided statistics. Label it an MFC curriculum port rather than reproduction of the paper’s entire optimizer.

**Silent implementation failures.** `train_grpo.py` still prints and ignores unknown arguments. Make flags strict before launching. Test context limits and failure handling with small runs. [Parser](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/train_grpo.py:2145)

**Memory and cost.** Measure a complete generation, backward, and initialized-Adam cycle for both models. The trainer stores parameters in fp32. Interleaved SFT has a second Adam optimizer; on a nominal 3B model, parameters, gradients, and two optimizers’ moment states already approach 72 GB before other allocations. Benchmark that configuration separately. Record acquisition, prefill, generation, reference scoring, and optimization costs. [Interleaved SFT](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/train_grpo.py:983)

**Matched-accuracy support.** Build reference trajectories before confirmation. Select a common accuracy interval using validation and freeze a reference-extension rule. Preserve unavailable bootstrap draws; report support failures. Add seed uncertainty to the current problem-only bootstrap. Accuracy matching is a diagnostic, so accompany it with joint accuracy/coverage trajectories and equal-compute results.

**Reasoning-judge validity.** Calibrate on blinded examples containing valid solutions, answer-correct errors, guesses, truncations, and intermediate-answer leakage. Measure errors separately by source and length. Judge every answer-correct sample on the chosen subset. Use human adjudication and false-positive sensitivity analysis; a few false positives can dominate large-k coverage. CoT-Pass@k’s underlying criterion is joint answer and reasoning correctness—2506.14245.

The existing judge silently removes missing votes and reports k only through 64. Require the specified vote count or retain an explicit unknown classification with bounds; add the registered larger-k metrics. Repeated calls to one judge do not provide independent validation. [Judge implementation](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/cot_judge.py)