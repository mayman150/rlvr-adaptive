# Coordinator's position (Claude), November 1, 2026

1. Diagnosis. The direction is too narrow in three ways, not wrong in its findings.
   (a) Metric. "pass@16 at matched pass@1" asks whether a method shifts the accuracy-coverage curve. It conditions
   away any gain that comes with higher accuracy, so a method that makes RL both more accurate and more diverse can
   read as zero. The user's goal ("diversity that makes RL better") is closer to fixed-compute outcomes: at equal
   training compute, higher pass@1 AND higher pass@k (large k). Matched accuracy should be a secondary diagnostic.
   (b) Setting. MATH test with a math-specialized 1.5B model leaves little headroom: base pass@32 is 85.5%, our
   pass@16 sits at 86-89, and the hard (c = 0) test stratum is 51 problems. RL-side exploration effects are claimed
   mostly on harder sets (AIME/AMC/Olympiad-level) and at k = 64-1024. Our null may partly be a ceiling.
   (c) Method family. Every RL-side lever we tested kept 8 rollouts per problem at temperature 1. With p_x < 1/8 the
   group rarely contains a success, so RL gets no signal exactly where coverage could grow. We never tested the
   levers that change this: rollout budget, exploration temperature/entropy, pass@k objectives.
2. Thesis I would bet on: "RLVR's diversity collapse is a rollout-signal problem on low-p problems: allocating
   rollouts adaptively (many rollouts only where p is small but nonzero, estimated online) and crediting the group by
   its first success (a pass@k-style advantage) raises pass@k at large k without lowering pass@1, at equal total
   rollout compute." The method would combine adaptive per-problem group size with a pass@k advantage; closest
   work: BroRL (many rollouts), Pass@k Training, outcome-based exploration (UCB-Con/Batch, tested null), GRPO with
   adaptive sampling (DAPO's dynamic sampling filters zero-variance groups but does not enlarge them).
3. Setting: keep Qwen2.5-Math-1.5B (all infrastructure), add a harder train/test split (e.g., MATH level 5 plus an
   Olympiad-style set with answers), evaluate pass@1 and pass@k for k up to 256 on the hard set, primary reading at
   fixed training compute; matched-accuracy curves as diagnostic. Replicate on SmolLM3-3B later.
4. First two weeks: (i) $50: re-evaluate existing RL variants' step-150 checkpoints with 256 samples on hard test
   problems (did we miss an RL effect at large k?); (ii) about $700: Dr. GRPO vs (a) T = 1.2 rollouts, (b) pass@k
   advantage, (c) 32 rollouts per prompt (or adaptive group size), 2-3 seeds, 150 steps, read at fixed compute and
   at large k on the hard stratum. Gate: a lever beats Dr. GRPO on hard-set pass@64+ at fixed compute without a
   pass@1 loss.
5. Kill: all levers equal Dr. GRPO at large k on the hard set at fixed compute. Fallback: the measurement paper
   (curve set by the start; teacher reasoning moves it; RL-side levers do not), now with large-k and hard-set evidence.
