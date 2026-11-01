# Round 1 — Fable (independent reviewer), November 1, 2026

Answered for the updated target: NeurIPS 2027 (~mid-May, ~28 weeks), quality first, staged budget. Facts checked: MFC (2609.13997) full text; AdaBack (2506.18110), 2605.29190, PrefixRL (2601.18795), No Free Lunch (2610.39295), Quagmires (2510.01624) abstracts; train_grpo.py; teacher-archive coverage computed locally.

## 1. Is the direction right? The single question, and a thesis that can come out either way

Yes, with one narrowing. The reviewer's "central question" is two questions: what information the teacher adds (direction 1) and what delivery retains it (direction 2). The gap both lit files agree on is persistence at matched accuracy: no published guidance method — MFC, Curriculum RL, AdaBack, QuestA, PrefixRL — tracks whether its gain survives after guidance is gone, at matched pass@1. v6 already owns the negative half (every start decays to ~+0.4 by step 150 against 700-step references). The single question:

"On problems with zero observed base success, does any delivery of external reasoning — weight-space (SFT then RL) or in-context with withdrawal (MFC-style) — durably lift pass@k at matched pass@1 after guidance is fully withdrawn, and does the answer depend on whether the content comes from a teacher or from the base's own deep sampling?"

Thesis either way:
- Positive: withdrawal-based delivery retains what SFT-then-RL loses. The frontier is movable and delivery, not content alone, is the lever. This overturns v6's closing claim and is a method-level result.
- Negative: every delivery decays to the same curve on hard sets with headroom, on two families; published boundary-expansion claims are fixed-step speed artifacts; coverage at matched accuracy is a property of the starting model, now confirmed against the strongest counterexample family. This is the science paper.

Two conditions make either outcome NeurIPS-grade rather than workshop-grade: hard sets must pass Gate V (headroom), and the MFC-style implementation must be defensibly faithful — a negative with a nerfed baseline is worthless. With 6.5 months the bar moves from feasible to definitive: two families in the headline (not "a second family is being prepared"), persistence to ≥500–700 steps, a registered confirmatory stage. One warning: this subfield moves monthly (MFC and 2610.24663 are weeks old). Six months of silence on exactly the persistence question invites scooping; decide at Stage 3, not at the deadline, whether the result goes on arXiv early.

## 2. Essential / optional / drop; critique of the 2x3

Essential: direction 2 (persistence with MFC-style withdrawal — the one live counterexample family we can actually run) and a reduced direction 1 (teacher vs deep-sampled self at matched prompts; the harvest is queued). Direction 3 (C_k^valid) is an audit, not a direction: run once on {base, Dr. GRPO endpoint, winning arm} on ~50 problems with a human-checked judge; never headline it. Direction 4 is mostly spent: allocation, UCB-Con, prefixes, wide groups are already on the curve in v6. What remains is one strong RL-side arm on the hard sets (Pass@k Training from the November 1 plan, or BBG-style zero-success-only updates, 2606.15455 — trivial to implement), without which "RL-side stays on the curve" is open to the ceiling objection. A faithful DATPO reimplementation (trees, sentence-entropy forks, sibling-diversity advantage) is weeks of engineering whose failure is uninterpretable; drop it and say why. Direction 5: demote to one descriptive analysis.

The 2x3:
- The retention cell is mislabeled. 2605.29190's bonus is not a retention term: it rewards correct rollouts by perplexity under the reference model — a diversity bonus conditioned on correctness (verified from the abstract). This group already showed two outcome-diversity bonuses stay on the curve. If you want retention, use KL-to-the-SFT-start (a real regularizer, cheap). If you want to test 2605.29190, implement it as published and name it correctly. Running both doubles the row for little return.
- "Equal tokens and compute" across delivery is ill-defined. SFT-then-RL presents a 25K-token dose once; a 500-step withdrawal curriculum re-presents trace prefixes hundreds of times. Pre-register what is held fixed (total teacher tokens presented, or compute) and report the other.
- One delivery contrast is already answered: TIME (interleaved SFT during RL) was equivalent to SFT-before-RL once the dose completed (v6/P36). Cite it; do not rerun.
- Missing control the reviewer explicitly asked for: answer-free teacher traces with all intermediate answers removed (CORR leaves intermediate answers in place). Add it as a source variant within the SFT delivery only — not a full row.
- The deep-self row can be empty: Gate H needs ≥44/109 audited-valid. Pre-register the fallback (PEER pool, or teacher-only study) before Stage 0 runs.

Recommended core: 2x2 {teacher, deep-self} × {SFT→RL, MFC-style withdrawal}, plus Dr. GRPO, one RL-side representative, answer-free-teacher SFT, and one retention arm as secondary. Eight arms, better targeted than the effective nine of the 2x3.

## 3. Novelty, paper by paper

- MFC 2609.13997 (full text verified): partial teacher traces appended to the prompt as a hint (not prefill); per-problem monotone frontier g_curr starting at the full trace; training level drawn uniformly from [0, g_curr]; frontier drops to the used level when mean group reward ≥ 0.5, never rises. Students Qwen3-1.7B/0.6B, teacher DeepSeek-V3.2, GRPO + two DAPO stabilizers, 1,200 steps. Evaluation is fixed-step/fixed-budget pass@256; no accuracy matching, no persistence tracking after withdrawal, and no control of SFT on the same teacher traces at equal content. Both our delivery contrast and the matched-accuracy persistence reading are new relative to it.
- AdaBack 2506.18110: per-sample, reward-driven revealed fraction of the target output; same evaluation gaps. MFC reports AdaBack's unguided share plateaus near 0.40 — withdrawal completeness is a variable to measure, not assume.
- Curriculum RL 2606.22317: repeated teacher SFT/RL rounds, +9.8 pass@256 at fixed k; teacher identity and compute unreported; persistence after the final bridge untested — our axis.
- PrefixRL 2601.18795: the sharpest counterexample — self-generated rejection-sampling prefixes, compute-accounted, claimed 3x final reward vs SFT-then-RL on hard problems. Its source is self, not teacher. The design's source × delivery cross is the unoccupied structure: PrefixRL does self only, MFC teacher only. It also bears on v6's "teacher content is the carrier": if deep-self + withdrawal durably matches teacher + SFT, the carrier story needs revision.
- DATPO 2609.08650: fixed-budget, k ≤ 64, 2,048-token cap; novelty should not rest on reimplementing it.
- 2605.29190: OLMo3-7B only, puzzle→math, fixed step; RL suppresses SFT-introduced primitives, reference-perplexity bonus restores +7pp pass@32. Whether restoration is a frontier shift or slower sharpening is exactly what matched accuracy decides — untested there.
- Quagmires 2510.01624: multi-family, endpoint-based; predicts post-RL from pass@large-k; never tracks an advantage along a matched-accuracy trajectory.
- No Free Lunch 2610.39295: theory under shared-strategy exponential tilt plus "exclusive specialization"; concludes RL can only move along the frontier. RL-side arms live inside its hypotheses (the one-curve result is its empirical counterpart); injection changes the strategy family and exits them. Frame the experiment as a test of the theorem's boundary, and run their log-readable diagnostic on our runs — it is cheap and preempts the obvious review question.

Net new claim: the first matched-accuracy, pre-registered test of whether in-context withdrawal and weight-space injection differ in the durability of added coverage, crossed with content source, on zero-observed-success problems, across two model families.

## 4. Staged plan (28 weeks; costs assume $4/A100-h; 1.5B ≈ 2 A100-min/step at 1,024 tokens, so a 500-step run ≈ $67 at 1,024 and ~$140–170 at 2,048; one 256-sample × 200-problem reading ≈ $6–8)

Stage 0 (weeks 1–2, ~$230, queued) — amend before launch:
1. Add a full SmolLM3 training-cycle calibration (generation + backward + optimizer at 2,048 and 4,096 tokens). P39 measures generation throughput only; 3B training cost is the largest unknown in the budget.
2. State Gate V for SmolLM3 explicitly, not only the 1.5B base.
3. Register the Gate-H fallback now.
4. No-compute week-1 work: segment the teacher archive into steps for MFC levels, and fix the coverage hole I found — the archive has verified teacher solutions for only 60 of the 109 c=0 problems (1–8 each). Either deep-sample the 7B teacher on the remaining 49 (cheap; add to the Stage 0 job) or upgrade the teacher for the hard pool. Note MFC's teacher is DeepSeek-V3.2; ours is Qwen2.5-Math-7B-Instruct — a weaker-information regime that must be stated as scope either way. If the Big-Math hard pool becomes the training set, there is no teacher archive for it at all; teacher generation is a real cost line.
5. Commit to running the long Dr. GRPO hard-set references first in the pilot stage — every matched reading depends on their pass@1 range (support failures killed the P36 late readings).

Stage 1 (weeks 2–3, ~$150): trainer work and gates. Implement the MFC-style mode: prompt-hint presentation (the existing prefix_rollouts is assistant-prefill continuation with a global linear anneal — materially different), per-problem monotone frontier, uniform level draw, success-triggered descent; plus KL-to-SFT-start. Registered smoke signature: unguided share rises monotonically; frontier histogram non-increasing; trainer rejects unknown flags. Independent diff review against the MFC paper (Codex). Document deviations (Dr. GRPO core, no DAPO stabilizers; 8 rollouts matches).

Stage 2 (weeks 3–6, ~$1,200–1,500): pilot at 1.5B, 2,048 tokens, hard pool, 1 seed, 300 steps: Dr. GRPO; teacher-SFT→RL; teacher-withdrawal; self-SFT→RL and self-withdrawal (if Gate H); answer-free-teacher SFT; one RL-side arm. Start the 3-seed, 700-step Dr. GRPO references here (~$650, included). Buys: arm selection, a support map, measured costs. Gate: arms advance on a descriptive matched-pass@128 gap ≥ +2 at steps ≥ 150; the two delivery cells advance regardless — they are the paper.

Stage 3 (weeks 7–12, ~$2,500–3,000): confirmation at 1.5B: 3–4 arms × 3 seeds × 500 steps, read against the 700-step references at registered checkpoints (pass@1/16/64/128 from 256 samples; hierarchical seed/problem bootstrap as the reviewer asked), both matched-pass@1 (primary) and equal-compute (secondary) readings; hard_test unsealed here only. Buys: the headline result.

Stage 4 (weeks 12–18, ~$1,500–2,500 pending calibration): SmolLM3-3B. Quality version: the 2x2 core × 2–3 seeds × 300–500 steps — with the NeurIPS timeline this is what makes the claim general. Fallback if 3B costs ≥3x the 1.5B: winner + Dr. GRPO only. This is the user's main budget decision: total ≈ $5.5–7K for the quality version, ≈ $4.5K with the fallback and one pilot arm cut.

Stage 5 (weeks 16–18, ~$300 + judge API): C_k^valid on ~50 hard problems × {base, Dr. GRPO, winner}; two independent judges, final answers masked, 30-trace human audit with agreement reported. Strategy-cluster descriptive pass only if time remains.

Weeks 18–26: analysis, writing, external review rounds; two weeks of buffer for capacity delays and reruns.

## 5. Biggest risks and week-1 de-risking

1. MFC infidelity (highest). Divergences to manage: prompt-hint vs our prefill machinery; per-problem frontier vs global anneal; Dr. GRPO vs GRPO+DAPO stabilizers; teacher strength. The prior internal null (human/self prefixes −0.05/−0.11) says in-context guidance was inert in this recipe once before — if the MFC arm shows no start-offset even early in RL, treat that as an implementation alarm, not a finding. Week 1: smoke-test signature, independent diff review, and log the unguided share so withdrawal demonstrably completes (AdaBack's plateaued at ~0.40).
2. Teacher coverage hole: 60/109 verified. Fix in week 1 (teacher deep-sampling or teacher upgrade) or the information-boundary comparison is underpowered at exactly c=0, the stratum where v6 locates teacher gains.
3. Support at matched accuracy: teacher arms overshoot the reference's pass@1 range (this made P36's late readings uninterpretable). De-risk: 700-step hard-set references before any arm; denser checkpoints where arms will land; pre-registered support rule plus an equal-compute secondary reading that cannot fail support.
4. Compute and operations: hours-long capacity delays, idle-reservation burn, one coordinator, and the known duplicate-SageMaker-job billing failure (keep the watcher alive; add a job-dedup assertion to the launcher). Hard dollar caps per stage; rebudget on Stage 0 throughput.
5. Judge validity: LLM judges pass answer-correct-but-invalid chains. Mask final answers, use two judges, human-audit 30 traces, report agreement; if agreement < 80%, publish the counts and drop the C_k^valid claim.
6. Six-month drift and scooping: pre-registration discipline decays over 28 weeks and evaluation variants sprawl. Freeze the estimator code and the registration ledger now; label all new readings exploratory. Revisit arXiv timing at Stage 3.
