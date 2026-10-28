Discussion request (be candid, compact, link every claim; search the web). Context: our project
(docs/full_story.tex "Where we stand"; experiments/training/INVESTIGATION_27.md) finds that near-on-policy RLVR
interventions on Qwen2.5-Math-1.5B/7B lie on Dr. GRPO's pass@16-vs-pass@1 curve and preserve the base model's per-problem
difficulty order. The user asks whether off-policy RL for LLMs is "a different problem". My view: (a) training-inference
mismatch / asynchronous-RL work (e.g. 2510.26788, 2602.01826, 2505.24298, 2607.22186, 2610.09597) addresses stability and
throughput of the same objective, so it is a different problem that at most changes speed; (b) off-policy-DATA methods
(replay/buffers, teacher or stronger-model traces, base anchors: BAPO 2602.20722, LUFFY 2504.14945, ExGRPO 2510.02245,
TOPR 2503.14286, Asymmetric REINFORCE 2506.20520, ARMOR 2607.10481, ReLIFT) overlap with our question, because injecting
samples the policy did not produce is a candidate route off the curve; most report gains at fixed steps, which may be speed.

Answer, numbered:
1. Do you agree with (a)/(b)? Correct me where wrong (e.g. can mismatch/staleness change the frontier, not just speed?).
2. For each off-policy-data method above (and any others you find): what exactly is claimed about coverage/pass@k or
   base-unsolved problems (e.g. BAPO's "resolves 40.7% of problems that base models consistently fail to solve": how
   measured, which k, which base, fixed step or matched accuracy?); are checkpoints public on Hugging Face (give repo ids
   and base models, verify)?
3. Which of them can be put on our curve cheaply (evaluation only, like Pre-registration 28) and which need
   reimplementation; rank by threat to our claim.
4. How should we scope and frame our claim given this (e.g. "near-on-policy, correctness-only RLVR"); does this change
   the paper's story or the next experiments? Recommend concretely.
