# Pre-registration 22: does external guidance move the curve where the policy's own does not?

Declared October 25, 2026, before any run of this design; revised after Codex's review, before launch.

## What was known

At matched pass@1, every tested intervention that uses only the model's own samples lies on Dr. GRPO's curve,
including prefix-guided rollouts from stored base-model successes (PG, Pre-registration 16 pilot: -0.35, 95%
[-1.60, +0.80]) and rollout allocation that gives rare problems 2.5 times more learning signal
(Pre-registration 18). Supervised fine-tuning on human solutions (Pre-registration 19) put the policy above
Dr. GRPO's 512-token curve only because its short, human-style answers escape truncation; at 1,024 tokens it falls
below the curve (Check 21, interim), and RL after it barely moves pass@1. Thesis: the curve is set by what the
policy samples; information it rarely produces (human solutions to problems with few or zero observed successes
in 32 base samples) should move the curve if supplied as context for the policy's own continuations instead of
replacing its answer style. Prior work with partial-solution guidance: QuestA (2507.13266; question augmentation
with partial solutions whose length is reduced during training), LUFFY (2504.14945; importance-weighted
off-policy traces), BREAD (2506.17211; adaptive expert hints). The contribution here is the matched-pass@1
comparison of the guidance's source under one mechanism.

## Mechanism (all arms)

Pre-registration 16's prefix-guided rollouts with two changes: for a problem with at most 4 successes in 32 base
samples (counts from `data/discovery_qwen_train32.json`) and a retained stored solution, 2 of its 8 rollouts are
policy-sampled continuations conditioned on the first f_t of the stored solution's tokens, f_t = 0.5 (1 - t/100)
(no prefix after step 100), **cut before the stored solution's first final-answer box** (`--prefix-before-box`,
so a prefix never supplies the answer; prefix-only verification is logged). Prefixes come only from the archive's stored
solution (`--prefix-fixed`), not from the policy's later successes, so the guidance source stays fixed. Only continuation log-probabilities
enter the loss; the verifier grades prefix plus continuation; prefix plus continuation stay within 1,024 tokens;
Dr. GRPO advantages over the group of 8. Stored solutions are retokenized and filtered as in `load_discovery`
(longer than 1,023 tokens or failing re-verification: dropped, count kept); retained problems by stratum and seed
are logged.

## Arms (seeds 8, 9, 10; Pre-registration 21's recipe: 16 x 8 rollouts, 1,024 tokens, AdamW 1e-6, 150 steps)

- **HG1k (human guidance, all eligible problems):** stored solution = the problem's human-written MATH solution
  (`data/human_prefix_train.json`: the human solutions with the discovery file's base counts; 1,668 rare problems
  before filtering, 487 with zero observed successes).
- **HGm1k (human guidance, matched problems):** as HG1k, restricted to rare problems that also have a retained base
  success (`--prefix-match-archive discovery_qwen_train32.json`), i.e. PG1k's problems.
- **PG1k (self guidance):** stored solution = the base model's first verified success (988 rare problems before
  filtering).
Reference: Pre-registration 21's B1k (same seeds, same problems per step).

## Evaluation

As Pre-registration 21: the 511 problems, 32 samples at T = 1, 1,024 tokens, steps 50, 100, 150 (no prefix at
evaluation), PG1k also at step 25 (reference knots); greedy accuracy on the 3,382 test problems at step 150.
Training logs keep assisted (prefixed) and unassisted rewards apart, with prefix and continuation token counts.

## Decision rules (Pre-registration 21's estimator and guards; 2,000 shared bootstrap draws; Holm over rules 1-2)

Pre-registration 21's overlap, clamping (10%), missing-evaluation and equal-seed-weight rules apply; intervals
condition on these seeds.
1. **Human guidance moves the curve:** HG1k against B1k's trajectory-and-temperature envelope: at least +2, 95%
   interval above 0, positive in all three seeds, Holm-adjusted p at most 0.05 -> moves the curve; average
   equivalence (pooled and per-seed 90% intervals in [-2, +2]) -> on the curve; otherwise inconclusive.
2. **The source matters on the same problems:** HGm1k against PG1k's trajectory: at least +1, 95% interval above 0,
   per-seed signs agreeing, Holm-significant.
Every positive claim is confirmed on the fresh 512 problems (`data/confirm_fresh_512.json`, same sign, 90%
interval excluding 0) before it is reported as established. If B1k's or HG1k's step-150 truncation exceeds 20% in
any seed, verdicts are qualified; a claim that guidance moves an unconstrained curve also requires a 2,048-token
re-evaluation of the step-150 checkpoints. A null rejects this annealed schedule, not external guidance in general.
Scope: 150 steps, 1,024 tokens, T = 1.

## Reported without a rule

PG1k and HGm1k against B1k's envelope; HG1k against HGm1k (the 680 problems only HG can guide); gaps at step 50
(during guidance) and step 150 (50 steps after it ends) separately; pass@16 on problems with 0 and 1-4 base
successes; prefixed rollouts per step, their success rate, prefix-only correct, prefix and continuation tokens;
truncation and answer length; step-150 pass@1 minus B1k; greedy accuracy.

## Cost

9 training runs and 30 checkpoint evaluations on AWS L40S, about $250.
