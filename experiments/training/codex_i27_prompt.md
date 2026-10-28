Review experiments/training/INVESTIGATION_27.md (exploratory). Data: results/worth-v15v-all-*/<NAME>/eval/*_eval.json
(counts of 32 per problem; eval_vllm.py uses the same per-problem sampling seeds for every checkpoint). Answer briefly,
numbered: (1) does finding 2 support "methods are indistinguishable from seeds per problem", given common sampling
seeds and the 1.5-point matching window, and what cheap statistic would test it properly (e.g. a permutation or
paired comparison of method-vs-seed differences); (2) is the uniform-logit-shift check (finding 4) a fair test, and
what known per-problem model would be better (e.g. a Beta prior / the project's frontier law in law.py); (3) is the
next test (base at lower temperature) well designed to separate "RL sharpens the base distribution" from alternatives,
and what would each outcome mean; (4) wording fixes.
