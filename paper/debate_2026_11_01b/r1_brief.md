# Debate, November 1, 2026 (evening): a setting with headroom and diversity to test our point

Participants: Claude (coordinator), Codex, Fable. Strict, independent senior researchers. No praise.

## Situation
The user now suspects the testbed, not the analysis: "it is better to have some analysis with data that has more
diversity to show our point." Evidence that our current setting (Qwen2.5-Math-1.5B, MATH train, MATH-511 test,
1,024 tokens, Dr. GRPO 150 steps) is saturated or degenerate:
- Base pass@256 on MATH-511 = 94.9; RL pass@16 = 87-89; the base-zero test stratum is 51 problems.
- RL's early gain is largely finishing/boxing (base leaves 42.5% unboxed; correct +18.4 while unfinished -18.3 by
  step 25); Qwen2.5-Math improves under random rewards (public checkpoint on our curve).
- Tiny RL: 150 steps x 16 x 8 = 19,200 rollouts, about 5 per training problem.
- New descriptive results (experiments/training/results_p26/p30_dose_starts_vs_long_refs.txt,
  p30_amend5_and_starts_vs_B1kL.txt): teacher SFT starts from 25K to 2M tokens, and every warm start after 150 RL steps,
  land within about 1 point of long Dr. GRPO references at matched pass@1, where pass@16 is 88-89 (near ceiling).
- Draft paper: paper/icml/paper_v6.tex (thesis: RLVR moves along a coverage frontier set by the base model).
- Prepared but not yet run: frozen hard sets (data/hard_val.json 200, hard_test.json 400; OlympiadBench + AIME, dedup
  vs MATH), Stage 0 job (PREREGISTRATION_39.md: base/RL checkpoints at pass@128 on hard sets, harvest pilot, SmolLM3
  census), waiting for an 8 x A100 machine. SmolLM3-3B-Base viability: pass@1 44.3, pass@16 79.1 on MATH-511.
- Compute: SageMaker p4de 8 x A100 at about $32/hour (capacity often delayed hours); 1.5B Dr. GRPO 150 steps about
  2 min/step/A100 at 1,024 tokens; about 11 weeks to the ICML deadline; budget not unlimited (aim <= $3K).
The user's goal from the start: make RLVR more diverse (they asked whether RL rollouts can help diversity).

## Coordinator's position (Claude)
Our point should be tested where it can fail: (1) data with headroom (base pass@k well below 100% at k = 128-256)
and with many valid solution paths, so both across-problem coverage and within-problem strategy diversity are
measurable; (2) a model that is not math-saturated; (3) longer answers (4,096 tokens) and more RL. Two candidate
settings: (A) hard math: train on a large pool of hard problems with answers (e.g., DeepScaleR-style AIME/AMC/
Omni-MATH history, deduplicated), test on the frozen OlympiadBench+AIME sets at pass@1-256; models Qwen2.5-Math-1.5B
(continuity) and SmolLM3-3B or Qwen2.5-3B (general). (B) a synthetic verifiable task with controllable difficulty and
many distinct valid solutions (e.g., Countdown or Reasoning-Gym tasks), where strategy diversity can be counted
exactly and contamination is impossible. Run Dr. GRPO plus two or three credible diversity methods (Pass@k Training,
an entropy-preserving update, a BroRL-style large-group run) and a teacher/self warm start, read at equal compute
(primary) and matched accuracy (diagnostic).

## Questions (answer in at most 800 words)
1. Diagnosis: is the saturation/elicitation explanation right? Which of our conclusions survive it and which do not?
2. "Our point": state the one claim the new setting should test, in a form that could come out either way.
3. Setting choice: (A), (B), both, or something else (code with tests, other data)? Name concrete datasets and
   models, token budgets, k values, and why each has headroom and measurable diversity. Check facts you rely on
   (dataset sizes, licences, availability, contamination, verifier compatibility) where you can.
4. Minimal decisive plan for the next 3 weeks with costs and go/no-go gates, reusing what exists (hard sets, Stage 0,
   SmolLM3, the estimator and analysis code). What should Stage 0 change, if anything, before it starts?
5. What result would mean the user's intuition (RL rollouts can help diversity) is right, and what would mean the
   frontier thesis is right?
