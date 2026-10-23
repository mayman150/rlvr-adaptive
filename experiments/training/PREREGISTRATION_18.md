# Pre-registration 18: does allocating rollouts by frontier value move the frontier at equal compute?

Declared October 23, 2026, before any run of this design. The forecasts below were computed with
the frozen frontier model of Pre-registration 17 (`law.py`, with the allocation extension
`law_alloc.py`) before training. Reviewed by Codex before launch.

## Rationale (from Pre-registrations 12-17)

Every correctness-only rule tested behaves like Dr. GRPO on rarely solved problems: a problem's
expected step under Dr. GRPO with group size n is (n - 1) grad p, and a realized step needs a sampled
success, which for small p occurs with probability about n p. Reweighting advantages cannot change
this; the number of rollouts a problem receives can. Groups of 32 (4x rollouts) reached pass@1 52.5
(Pre-registration 14, interim). The frontier model predicts that redistributing a fixed rollout
budget toward problems with high frontier value moves the matched-pass@1 frontier, while
redistributing it toward high per-rollout signal (as speed-oriented allocation methods do) does not.

## Arms (seeds 8, 9, 10; Pre-registration 12's settings: 16 problems per step, AdamW 1e-6, 150 steps, 512 tokens; same problems per step as B)

Both use exactly 128 rollouts per step, as B does: stage 1 draws 4 rollouts per problem (64);
stage 2 distributes the remaining 64 in proportion to a value of each problem's stage-1 count c of 4
(largest remainders, at most 32 rollouts per problem); Dr. GRPO advantages within each problem's group;
the same loss normalizer as B.
- **AC (coverage allocation):** value = posterior mean of 16 p (1 - p)^16, p ~ Beta(c + 1/2, 4 - c + 1/2)
  (expected pass@16 gain per extra rollout). Values for c = 0..4: 0.175, 0.094, 0.021, 0.0025, 0.0001.
- **AV (variance allocation, control):** value = posterior mean of p (1 - p) (expected signal per
  rollout; the target of speed-oriented allocation such as HORA 2605.07114 and Knapsack RL 2509.25849).
References: Pre-registration 12's B (trajectory and temperature envelope); Pre-registration 14's G32
(4x rollouts, reported).

## Forecasts (frozen model; pass@16 gain over B at matched pass@1 0.37 / 0.40 / 0.44)

| Model variant | AC | AV |
|---|---|---|
| Frozen law (32-sample base counts, tau -0.20) | +2.41 / +2.66 / +3.03 (mean +2.70) | -1.20 / -1.26 / -1.31 (mean -1.26) |
| Deep-sample counts (3 of 4 shards), tau_deep +0.20 | +1.46 / +1.59 / +1.74 (mean +1.60) | -0.74 / -0.79 / -0.77 (mean -0.77) |

## Evaluation

The 511 problems, 32 samples, steps 50, 100, 150 at T = 1; greedy accuracy on 3,382 test problems
at step 150. Pre-registration 12's interpolation and envelope; 2,000 shared bootstrap draws; clamping
reported.

## Decision rules

1. **AC moves the frontier** if its gain over B's envelope is at least +2 pp with the 95% interval
   above 0 and positive in all three seeds; otherwise Pre-registration 12's categories. Step-150 pass@1
   minus B is reported with its 90% interval.
2. **Allocation target matters:** AC against AV's trajectory at least +1 pp with the 95% interval
   above 0.
Holm over rules 1 and 2. If rule 1 passes: confirmation on the fresh 512 problems
(`data/confirm_fresh_512.json`), same sign and 90% interval excluding 0.
3. **Forecast accuracy (reported):** the frozen-law forecast is confirmed for an arm if the observed
   gain lies within 1.5 pp of its mean forecast; signs are compared for both arms.

Reported without a rule: allocated group sizes by stage-1 count; pass@16 by base-count stratum;
AC and AV against G32's trajectory where pass@1 ranges overlap; generation time.

## Cost

About $100 training (six runs) and $90 evaluation.
