# Coordinator, round 2
- I withdraw my method: adaptive rollouts + pass@k credit is covered by DARS 2508.13755, Knapsack RL 2509.25849,
  Pass@k Training 2508.10751 and DATPO 2609.08650. Codex's correction stands (p = 1/8 gives a 66% chance of a success
  in 8 rollouts); the signal argument holds only for p around 1/64 or lower.
- I lean to Fable's HARVEST as the method candidate: it is cheap (generation only), connects to the user's intuition
  (rollouts help), and tests a clean question (sampled support vs model support). Exposure arithmetic supports it:
  150 steps x 16 prompts x 8 rollouts = 19,200 rollouts over 3,920 problems, about 5 per problem, so RL cannot find
  p of about 1% solutions that 1,000 base samples find. Closest prior art I see beyond STaR/RFT: DART-Math
  (difficulty-aware rejection tuning: more synthesis trials on hard queries, but with a stronger synthesizer, not the
  base, and no RL or coverage reading), ReST/Expert Iteration. Please check novelty against these.
- Codex's rare-success constraint is complementary: if HARVEST's gain fades like SELF's, the fade is either reference
  catch-up or suppression of rare successes by RL updates; Codex's week-1 check (resample rare problems at existing
  checkpoints) distinguishes them, and a constraint would then protect the harvested successes. Proposed hybrid:
  HARVEST, plus constraint only if Stage 0 shows suppression.
- Agreed so far (please confirm or object): harder held-out set with headroom (OlympiadBench-style, about 400-500
  problems) plus MATH-511 for continuity; 2,048-token primary; pass@1 and pass@64/128 with 256 samples; primary
  reading at equal total compute (counting harvest generation), matched-accuracy frontier as the diagnostic;
  Stage 0 eval-only re-evaluation of existing checkpoints at large k; kill criteria and fallback mechanism paper.
