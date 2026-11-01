# Broad literature review (2025–Oct 2026): boundary expansion, persistence, evaluation

Scope per brief_fable.md. Excludes the nine items under Codex verification (Yue 2504.13837; Invisible Leash 2507.14843; RL-Squeezes 2509.21128; CoT-Pass@k 2506.14245; DATPO 2609.08650; MFC 2609.13997; Curriculum-RLVR 2606.22317; BOPTR 2609.01274; sparse-token 2610.24663). Every paper below was verified at the abstract level via the arXiv API on 2026-10-31; none is listed unverified. Venues are unconfirmed (arXiv preprints) unless stated. "Comparison regime" is what the abstract states; most abstracts are silent, which itself supports our critique. Background rejection-sampling lineage (RFT, STaR, DART-Math arXiv 2407.13690 — verified; 7B–70B models, difficulty-aware sampling, no matched-budget claim) predates the window and is cited only as lineage.

## 1. Methods claiming to expand the reasoning boundary / pass@k at large k

**LUFFY** (2504.14945). Mixed-policy GRPO folding off-policy expert traces into RLVR; models unnamed in abstract. Claims "+6.4 avg" over six math benchmarks, "+6.2" OOD. No matched-step/compute statement. The in-RL analogue of our teacher-SFT start; no matched-accuracy control.

**QuestA** (2507.13266). Partial-solution hints appended to prompts during RL; 1.5B models built on DeepScaleR/OpenMath-Nemotron (Qwen lineage). AIME24 72.50 (+10.73), AIME25 62.29 (+12.79), HMMT25 41.67 (+10.11). Comparison regime unstated. Hint scaffolding with implicit withdrawal at test time; Qwen-only.

**BREAD** (2506.17211). Branched rollouts from expert anchors: short expert prefixes grafted into failed GRPO rollouts; small LMs; <40% of ground-truth traces, ~3x faster than GRPO. The headline is a speed claim — exactly the speed/coverage confound our instrument separates.

**PrefixRL, "Reuse your FLOPs"** (2601.18795). Conditions RL on very off-policy prefixes from base-model rejection sampling. Reaches the same training reward 2x faster than the strongest baseline *including* the rejection-sampling compute, and 3x final reward vs SFT-then-RL on hard problems; transfers to held-out benchmarks. The only method here with compute accounting; still no matched-accuracy coverage claim.

**MOER** (2610.30873). Decaying expert CoT prefix, selective rollback to full expert trace on failure, GRPO + trust-region SFT loss. No numbers or models in abstract. The cleanest withdrawal design; evidentially thin at abstract level.

**H²R** (2610.39934). Hierarchical hint rewards; Qwen2.5-Math-7B; authors' own HiMath benchmark 30.4%→39.1% (45.6% oracle hints). Qwen-Math only; self-built benchmark; regime unstated.

**RuscaRL** (2508.16949). Checklist rubrics decayed over rollouts plus LLM-judge rewards; claims boundary expansion under Best-of-N; no numbers, models, or budget protocol in abstract.

**Residual Advantage / CoRA** (2610.11519). Teacher-student residual added to verifier advantage; Qwen3-1.7B/4B students, 8B teacher; macro Avg@8 +1.7–3.6, Pass@8 +3.9–6.3 across 24 comparisons. Qwen-only; fixed-step by construction; jointly reports accuracy and pass@8 but not at matched accuracy.

**MagicBeanStalk** (2610.12659). Recursive problem synthesis to escape the zero-advantage trap (~88% of a 78K pool is all-solve or all-fail). gpt-oss-120b 61.2→69.8, DeepSeek v4Flash 72.4→77.9 over 8 math benchmarks at a 16K-token budget. Notably non-Qwen; data-side rather than algorithm-side; no pass@k claim in abstract.

**ProRL** (2505.24864) and **BroRL** (2510.01180). Prolonged RL (KL control, reference resets) claims pass@k gains over base "including where base fails at every attempt"; BroRL scales rollouts per example to hundreds and revives models saturated after 3K ProRL steps. Neither abstract gives numbers or a compute-matched protocol. Our 1,000-step and rollout-allocation arms speak directly to both.

**Pass@k Training** (2508.10751). Pass@k as the RL reward with an analytical advantage; abstract names no models and no numbers.

**ExpDis** (2610.10536). Alternates novelty-bonus explorer policies with distillation into a bonus-free student; beats DAPO at the same wall-clock budget on seven math benchmarks, two families, with better pass@k scaling. Wall-clock matching is the closest any method paper gets to our standard — still not matched accuracy.

**BBG** (2606.15455). Diagnoses diversity collapse as overtraining; restricting updates to zero-observed-success problems lifts Pass@256 above base on hard benchmarks. Models unnamed in abstract. A direct frontier-lift claim from the RL side; we should test whether it survives matched-pass@1 comparison.

## 2. Persistence: do teacher-injected gains survive RL?

**Quagmires in SFT-RL Post-Training** (2510.01624). Hundreds of models to 12B (Llama3, Mistral-Nemo, Qwen3), 7 math benchmarks, up to 256 repetitions. High SFT scores mispredict post-RL outcomes; RL from an SFT model can end below RL from base; generalization loss and pass@large-k predict post-RL performance (R²/Spearman improved "up to 0.5 (2x)"). Multi-family, endpoint-based; the closest large-scale neighbor to our decay result — and their pass@large-k predictor is consonant with "coverage is a property of the starting model."

**When RL Suppresses Its Own Vocabulary** (2605.29190). OLMo3-7B; puzzle-only SFT gives +7pp pass@32 on OlymMATH-Hard, then vanilla GSPO suppresses the SFT-introduced exploratory primitives (hypothesize/backtrack, via motif classifier); a perplexity novelty bonus restores them (+7pp over vanilla GSPO; ceiling 16.0%→36.0%). Mechanistic, non-Qwen evidence that RL erodes injected behaviors — the strongest independent support for our decay finding, though fixed-step and single model.

**Sequential Beats Joint** (2609.04108). OPD-then-RL beats pure OPD, pure RLVR, and joint mixing on logic/math; no numbers. Explicit framing: OPD expands teacher-supported coverage, RL sharpens within it — our thesis stated as design principle, without matched-accuracy measurement.

**RL Starts before RL** (2609.28145). OPD-initialized students beat direct-RL and SFT-then-RL after shared-settings RL; forward-KL OPD overtakes reverse-KL after RL. Claims warm-start ordering persists to the endpoint — in tension with our +0.4-point residual; no numbers, no matched accuracy, models unnamed.

**OPDArena** (2610.25651). Five pretrained models; students of RLVR-trained teachers retain prior capabilities, students of SFT teachers lose more. No numbers.

**Does self-distillation improve reasoning?** (2610.18613). Qwen3 1.7B–14B and OLMo-3-7B; reported SDPO/OPSD gains largely fail on unseen problems; attributed to overfitting/behavior transfer. A skeptical re-evaluation in our methodological register.

Background: **RL's Razor** (2509.04259; forgetting tracks KL to base, no numbers) and **SFT Memorizes, RL Generalizes** (2501.17161; GeneralPoints/V-IRL, no numbers).

## 3. Evaluation, metrics, and support/likelihood analyses

**There Is No Free Lunch in RLVR** (2610.39295, 2026-10-29). Theory: RLVR as shared exponential tilt; via Stiemke duality, any pass@1 gain strictly lowers some prompt's success, crossings bounded by sign variation; includes a log-readable crossover diagnostic. No empirics. Predicts curve-constrained movement — the theoretical counterpart of our one-curve result; we must check whether our frontier satisfies (or refines) its conditions.

**RLVR is a Kernel, Not a Function** (2609.22547). Five public RLVR pairs; models post-RL success conditional on base success as a Markov kernel; confidence bands over k; "first loss" between 11 and 61 samples on fresh 32k-token prompts; no crossing statistically established in initial evals. The nearest statistical-inference standard to our pre-registered, bootstrap-interval design.

**What Fixed-Rollout pass@k Evaluations Can Identify** (2609.09245). Identification analysis on Brown et al.'s 10,000-rollout release; counterfactual n=16 evaluations leave failure at k=1,000 ambiguous by factors 1.5–2,600; sharp identified intervals plus finite-task certificates. Directly bounds what our pass@16-at-matched-pass@1 instrument can claim about large-k boundaries — we must either adopt their intervals or scope our claims to k=16.

**Beyond Pass@k: Cover@tau** (2510.08325) — large-k pass@k rankings unstable; proposes threshold coverage; no numbers. **What pass@k Cannot Measure** (2610.07405) — Qwen2.5-1.5B-Instruct, GSM8K, GRPO vs RFT, 3 seeds; correct-solution lexical diversity 15% lower for GRPO (length-controlled); no consistent pass@8/32 winner; argues GRPO's pass@1 edge is smaller-loss-vs-base, not capability. **Sharpening Tax** (2610.01509) — 42 base/post-trained cases, 4 families, agentic benchmarks; tax metric plus PTGS sampler. **Beyond Compression** (2609.37066) — cross-surface pass@K over prompt variants; post-training compresses sampling cost on AMC but raises the large-K ceiling on AIME; "compression is one regime, not a universal explanation" — a direct complication for a single-frontier story on heterogeneous difficulty.

**Reasoning with Sampling** (2510.14901). MCMC-style sampling from base-model likelihoods matches or beats RL-trained models on MATH500/HumanEval/GPQA with no training and better diversity. **Spurious Rewards** (2506.10947). Random/spurious rewards give Qwen2.5-Math-7B +21.4 on MATH-500 vs +29.1 for ground truth, via GRPO clipping bias amplifying pretrained behaviors; does not transfer to Llama3/OLMo2. Together: strong elicitation evidence and the sharpest warning that Qwen-Math-only results overstate generality — our 1.5B/7B main results are Qwen-Math; the 26-checkpoint cross-family prediction carries that burden.

**ScaleRL** (2510.13786). >400,000 GPU-hours; sigmoidal compute-performance fits predicting a 100,000-GPU-hour run. The compute-matched methodological neighbor; it fits accuracy-vs-compute where we fit coverage-vs-accuracy. **PBA** (2607.20543). Diagnoses pass@k inversion in a 3,000-prompt regime-controlled study; anchors low-evidence prompts to the frozen base; no numbers.

## (a) Five papers we must engage most

1. **2609.09245** (fixed-rollout identification) — bounds the inferential reach of pass@16; ignoring it invites a fatal review.
2. **2610.39295** (No Free Lunch) — theory predicting exactly our frontier phenomenology; engage or be scooped on the explanation.
3. **2510.01624** (Quagmires) — multi-family SFT-then-RL at scale; overlaps our decay result and our "starting model" conclusion; we must differentiate (matched accuracy, identical-prompt teacher-vs-self contrast, pre-registration).
4. **2601.18795** (PrefixRL) — the one compute-accounted claim that structured guidance beats SFT-then-RL on hard problems; the sharpest live counterexample to "nothing lifts the curve but warm starts."
5. **2609.22547** (RLVR is a Kernel) — sets the statistical bar for pass@k comparisons; our instrument should be stated in terms survivable under their critique.

## (b) Open gaps our matched-accuracy, pre-registered instrument can fill

1. **No persistence result at matched accuracy.** Quagmires, 2609.28145, 2609.04108, OPDArena all compare endpoints at fixed steps/budgets; none tracks a warm-start advantage along a matched-pass@1 trajectory. Our decay-to-+0.4 curve has no competitor.
2. **No guidance method evaluated at matched accuracy.** QuestA, BREAD, MOER, RuscaRL, H²R, Residual Advantage report fixed-step gains; ExpDis matches wall-clock, PrefixRL matches compute — none matches accuracy, so none can distinguish frontier shift from acceleration.
3. **No pre-registration anywhere.** 2609.22547 is post-hoc inference on public checkpoints; no training study registers decision rules before runs.
4. **Cross-family frontier prediction is unoccupied.** Boundary-expansion numbers concentrate on Qwen(-Math); the multi-family work (Quagmires, MagicBeanStalk, 2605.29190) never asks whether coverage-at-matched-accuracy is predictable from the base model. Our 26-checkpoint prediction is the only entry.
5. **Strategy-level diversity remains unmeasured on the frontier.** 2610.07405 is lexical-only on GSM8K; 2605.29190's motif classifier is one model, one transfer setting. Linking a strategy metric to position on the matched-accuracy curve is open.
6. **Per-problem locus of teacher gains.** MFC/Curriculum-RLVR (Codex's list) claim pass@256 gains on unsolved problems at fixed budgets; no one shows *where* gains sit at matched accuracy. Our per-problem decomposition can adjudicate expansion vs reallocation.

*Word count: ~1,490.*
