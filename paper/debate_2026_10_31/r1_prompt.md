# Debate brief, October 31, 2026: which NOVEL method makes RLVR rollouts durably more diverse?

You are a strict, independent ICML reviewer and research strategist. The user wants a METHOD paper (ICML 2027,
deadline late January 2027; about 12 weeks; compute: 8 x A100 machines on SageMaker, about $300 per 3-seed 150-step
comparison) whose contribution is a novel way to make RLVR's rollouts more diverse, measured as pass@k at matched
pass@1 (coverage), not just an analysis. Read the evidence and judge the candidates below. Be concrete and skeptical.

## Setting
Qwen2.5-Math-1.5B, MATH train (3,920 problems with answers; no human solutions in our data file), 511 MATH test
problems, Dr. GRPO (16 prompts x 8 rollouts, 150 steps, lr 1e-6), 32 samples per test problem, 1,024 tokens. Estimator:
a checkpoint's pass@16 minus the same-seed reference trajectory's pass@16 interpolated at its pass@1, averaged over
eligible checkpoints and 3 RL seeds, problem bootstrap. Files (repo ~/Desktop/adaptive-rlvr):
experiments/training/results_p26/p36_analysis_partial_20261031.txt, p36_phase2_detail.txt, p36_interim_job1.txt,
p36_interim_strata.txt, p36_interim_diversity.txt, p30_stage2_p34_analysis.txt, p30_2k_threeseed.txt;
docs/brief/brief.tex (the 5-page brief); experiments/training/PREREGISTRATION_3{0,2,3,4,5,6}.md; PREREGISTRATION_38_DRAFT.md
and codex_p38_review.md; paper/LIT_2026_10_28.md (literature notes).

## Evidence (registered unless marked descriptive)
1. RL-algorithm changes (>10, incl. UCB-Con/Batch outcome-diversity bonuses, Flat-RLVR, PSR/NSR, allocation, human
   prefixes, replay) never met the curve-raised criterion; several were equivalent within +/-2.
2. SFT starts (25K tokens) before RL: H25 (teacher solutions, hard problems) +1.92; TEACHF (teacher, P35 prompts,
   length-matched to SELF) +2.10 "curve raised"; CORR (teacher reasoning with corrupted final box) +2.14 "curve raised";
   SELF (base's own verified solutions, same prompts) +1.78 inconclusive; TEACHF beats SELF's trajectory +1.53
   [95% +0.34, +2.76], all seeds positive. OPD (on-policy distillation) start: no gain (-1.02 vs SFT-R25 trajectory).
3. Descriptive dynamics: self starts (SELF, CWS, UNI, TAIL) +2 to +3 at RL steps 1-25, about 0 by steps 100-150, ending
   at Dr. GRPO's endpoint (pass@1 57, pass@16 87). Teacher starts keep about +2 through step 50 and end at pass@1 63-65,
   pass@16 about 88.5 (steps 100-150 lie beyond the reference's pass@1 range; 700-step references finish tomorrow).
4. Descriptive stratum decomposition: self starts' early gain sits on test problems the base solves 1-8 of 64 and
   vanishes; H25's gain shifts toward problems the base never solved (c = 0: +0.86 early, +1.52 at step 50).
5. P36 (interim): coverage-weighted self-SFT (CWS) vs uniform self-SFT: +0.21 [-0.30, +0.69]: no targeting benefit.
   Interleaving the SFT during RL (TIME) equals before-RL once the dose is complete.
6. Diversity diagnostics: self starts increase distinct wrong answers (+1.3/problem) and spread success (pass@2 +1.2);
   teacher start: pass@2 +2.4, distinct wrong -1.5.

## Candidates
C1. Hint-conditioned self-solutions (teacher-free): for problems the base never or rarely solves, condition the base
    on the reference answer (or a partial/human solution, if obtained) to write its own full solution; keep verified
    ones (and possibly unverified), SFT before RL. Reaches c = 0 problems (the self pool cannot). Related: STaR's
    rationalization (2022), QuestA (question augmentation with partial solutions during RL), hint/prefix RL.
C2. Unverified teacher reasoning (label-free): since corrupted final answers work, SFT on teacher traces without
    verification, usable on problems without answer keys.
C3. Zero-signal-targeted injection during RL: when a group's rollouts all fail, add a hint-conditioned or teacher
    trace (off-policy) for that problem. Related: LUFFY, ReLIFT, prefix-guided RL; our human-prefix RL test was null and
    TIME suggests timing matters little.
C4. Anything better you can propose that the evidence supports.

## Questions
(a) Which candidate is most likely to give a durable, registrable coverage gain over Dr. GRPO AND over the closest
    baseline (SELF; and, for C1, a teacher)? Why, given evidence 1-6?
(b) Novelty: what does the 2025-2026 literature already cover (STaR rationalization, QuestA, LUFFY, ReLIFT, BREAD,
    hint-GRPO, "learning from failures", cold-start SFT before RL, distillation for pass@k)? What precise claim would
    be new and defensible at ICML?
(c) The minimal decisive experiment (arms, controls, primary contrast, decision rule) for the top candidate, within
    about $600 and two weeks; what would make you abandon it.
(d) Risks: confounds (pass@1 movement, format/finishing, answer leakage via hints, problem overlap with test),
    hardware, and whether the matched-pass@1 instrument can show the gain.
Answer in at most 900 words. No praise.
