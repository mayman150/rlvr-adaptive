# Schema debate, November 1, 2026 (three-way: Claude (coordinator), Codex, Fable)

The user feels the current direction is wrong and wants a good research SCHEMA: the question, the hypotheses, the
method, the experiments, the decision points, for an ICML 2027 paper (deadline late January 2027, about 11 weeks).
The user's goal from the start: make RLVR more diverse. They want a METHOD paper, and they are not convinced that RL
rollouts cannot help diversity. Be a strict, independent senior researcher; question our assumptions, including the
metric and the setting, not only the candidate methods.

Evidence and files (repo ~/Desktop/adaptive-rlvr): docs/brief/brief.tex (5-page brief, Oct 30);
experiments/training/results_p26/p36_analysis.txt (final P36 + P35 phase 2), p36_phase2_detail.txt, p36_interim_strata.txt,
p36_interim_diversity.txt, p30_stage2_p34_analysis.txt, p30_2k_threeseed.txt, smollm3_viability.txt;
paper/debate_2026_10_31/ (yesterday's method debate, r1/r2 of Codex and Fable); paper/LIT_2026_10_28.md;
experiments/training/PREREGISTRATION_3*.md. Compute: SageMaker 8 x A100 (p4de) machines, about $32/hour, capacity
sometimes delayed by hours; a 3-seed 150-step 1.5B comparison costs about $300; budget is not unlimited.

Summary of evidence: Qwen2.5-Math-1.5B on MATH (3,920 train problems; 511 test problems; base pass@1 21.6, pass@16 78,
pass@32 85.5), Dr. GRPO 16 x 8 rollouts, 150 steps, 1,024 tokens; instrument: pass@16 at matched pass@1 (32 samples).
- >10 RL-side changes (PSR/NSR, allocation at fixed budget, replay, human prefixes, Flat-RLVR, UCB-Con/Batch
  outcome bonuses, 1,000-step runs): none raised the curve; some equivalent, some inconclusive.
- SFT starts: teacher reasoning (TEACHF, CORR) "curve raised" (+2.1); beats the base's own solutions on identical
  prompts (+1.53 [0.34, 2.76]); self-SFT and coverage-weighted self-SFT give early gains that fade by step 50-150;
  coverage weighting = uniform (+0.03). Teacher starts also reach pass@1 63-65 vs Dr. GRPO's 58 at step 150.
- Novelty check: teacher SFT before RL is standard; answer-conditioned self-solutions = STaR/ExPO/SPLICE/LSPO.
- Untested RL-side levers: much larger rollout budgets per problem (BroRL-style), training temperature / entropy-
  preserving updates (e.g., Clip-Cov/KL-Cov), direct pass@k objectives (Pass@k Training), and measurement at larger k
  on the hard test strata (our pass@16 is near ceiling; the base-zero test stratum has 51 problems).

See the coordinator's own position in r1_claude.md. Then write YOUR independent schema (max 900 words):
1. Diagnosis: is the direction wrong? If yes, what exactly is wrong (question, metric, setting, method family)?
2. The research question and the one-sentence thesis you would bet the paper on.
3. Method: what is the method (RL-side, start-side, or hybrid)? Why should it work, given the evidence? What is new
   relative to 2025-2026 work (name the closest papers)?
4. Setting and metric: model(s), data, budgets, k, fixed-compute vs matched-accuracy readings, and why.
5. Experiment plan in stages with costs and go/no-go gates (first 2 weeks must be cheap and decisive).
6. What result would kill the thesis, and what is the fallback paper.
No praise; be concrete.
