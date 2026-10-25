# Pre-registration 21: does the one-curve result hold at a 1,024-token response budget?

Declared October 25, 2026, before any run of this design. Revised after Codex's review, before launch.

## What was known

All matched-pass@1 results so far were trained and evaluated with answers up to 512 tokens, where about 48%
of Dr. GRPO's sampled answers at step 150 are truncated. Re-evaluating 512-trained checkpoints at 1,024 tokens
(Checks 16 and 21) moved Dr. GRPO from about 44 / 74 (pass@1 / pass@16, step 150) to about 58 / 88 with 15%
truncation, halved the warm start's step-0 gain (+4.77 to +2.15) and turned the human-solution arm's position
from about +5 above Dr. GRPO's curve to about -3 below it (seed 10; other seeds pending). Arms that change answer
length are therefore confounded with the budget at 512 tokens. Arms with Dr. GRPO-like truncation (advantage
rules, rollout allocation) were not shown to be affected, but they were never trained at a larger budget.
Pre-registration 18 found frontier-value allocation (AC) on Dr. GRPO's curve at 512 tokens (+0.14, 90% interval
inside +-2).

## Arms (seeds 8, 9, 10; Pre-registration 12's recipe with answers up to 1,024 tokens in training and evaluation)

16 problems x 8 rollouts per step, one on-policy update per batch, sequence-sum loss with the constant
normalizer 128 x 1,024, AdamW 1e-6 (betas 0.9, 0.99, epsilon 1e-8, no weight decay), clipping 1, 150 steps, the
same problems and order as the 512-token runs of the same seed. As in Pre-registration 20, the cap and the
denominator change together; this does not isolate the cap's causal effect (a constant scaling of the loss
approximately cancels in AdamW, up to epsilon and clipping). Pre-clip gradient norms, clipped and skipped
updates, tokens and GPU-hours are reported.
- **B1k:** Dr. GRPO (A = r - mean).
- **PSR1k:** A = r. **NSR1k:** A = r - 1.
- **AC1k:** Pre-registration 18's AC with its frozen settings (4 pilot rollouts, 64 extras by the law value,
  lambda 0.37, k = 16, cap 32, fixed normalizer). Lambda 0.37 is transferred unchanged from the 512-token
  frontier; a null concerns this frozen rule.
Claims concern Dr. GRPO, PSR/NSR and the frozen AC intervention, not every correctness-only rule. Training on
Vulcan (one node per seed, one arm per GPU); AWS only if Vulcan stalls for more than 12 hours.

## Evaluation

The 511 problems, 32 samples at T = 1, answers up to 1,024 tokens, at steps 50, 100, 150 for every arm, and
step 25 for B1k and PSR1k (reference knots only, because the base-to-step-50 segment is long at this budget;
comparison points remain steps 50, 100, 150; a sensitivity omitting step 25 is reported); the base model at
1,024 tokens (Check 16, reused); B1k at T = 1.15 and 1.3 at step 150 (envelope); greedy accuracy on the 3,382 test
problems at step 150 (1,024 tokens). Truncation rate and mean answer length are recorded for every evaluation.

## Decision rules (Pre-registration 12's interpolation; 2,000 shared bootstrap draws over problems)

Estimator: eligibility (checkpoints inside the reference's pass@1 range) is fixed on the full data; checkpoints
are averaged within a seed, then seeds weighted equally; interpolation and envelopes are recomputed in every
shared draw. Missing required evaluations yield "incomplete evaluation". "Insufficient overlap" if fewer than two
checkpoints per seed are eligible; more than 10% of bootstrap points clamped in any seed makes a comparison
uninterpretable. Intervals condition on these trained seeds. Four two-sided directional tests (Holm); equivalence
uses unadjusted 90% intervals.
1. **Sign rules on one curve (T = 1 trajectories):** NSR1k against PSR1k's trajectory, PSR1k against B1k's,
   NSR1k against B1k's. **Average equivalence** if the pooled 90% interval and each seed's 90% interval lie
   within [-2, +2]; **departure** (either direction) if |pooled estimate| >= 2, the Holm-adjusted p-value is at
   most 0.05 and the per-seed signs agree; otherwise inconclusive.
2. **Allocation:** AC1k against B1k's trajectory-and-temperature envelope moves the curve if the gain is at least
   +2, the 95% interval is above 0, it is positive in all three seeds and the Holm-adjusted p-value is at most
   0.05 (then Pre-registration 18's confirmation on the fresh 512 problems); average equivalence as in rule 1
   replicates Pre-registration 18's null; otherwise inconclusive. AC1k against B1k's T = 1 trajectory is
   reported next to it.
Simultaneous 90% point bands are reported; checkpoint-average equivalence does not establish whole-trajectory
equivalence. A joint "the core result replicates at 1,024 tokens" statement requires all four contrasts to reach
average equivalence.
3. **Budget qualifier:** if the step-150 truncation of any arm in a comparison exceeds 20% in any seed, that
   verdict is qualified as "truncation still substantial at 1,024". The threshold is a reporting trigger, not
   evidence that truncation no longer binds.

## Reported without a rule

Each arm's position relative to B at 512 tokens (existing runs) next to its position at 1,024; training-time
statistics (mixed groups, answers with nonzero advantage, truncation, length, step time); for AC1k, group sizes by
pilot count and the share of pilot-zero problems with a success; pass@16 by base-count stratum; step-150 pass@1
of each arm minus B1k with its 90% interval; greedy accuracy.

## Cost

12 training runs (about 6 L40S-hours each at 1,024 tokens) and 48 checkpoint evaluations (about 2.4 L40S-hours
each), about 190 L40S-hours on Vulcan.
