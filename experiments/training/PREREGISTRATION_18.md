# Pre-registration 18: does allocating rollouts by frontier value move the frontier at an equal rollout budget?

Declared October 23, 2026, before any run of this design (revised after Codex's review, before
launch). Forecasts were computed with the frozen frontier model of Pre-registration 17 (`law.py`,
extended by `law_alloc.py`) before training; output in `PREREGISTRATION_18_forecast_output.txt`.

## Rationale

Every correctness-only rule tested behaves like Dr. GRPO on rarely solved problems: for a problem with
success probability p and a group of N rollouts, Dr. GRPO's expected log-odds step is
E[C (N - C) / N] (= (N - 1) p (1 - p) for fixed N), and a realized step needs a sampled success.
Reweighting advantages does not change this; the number of rollouts a problem receives does. Under
the frontier model, redistributing a fixed rollout budget toward problems where the matched-pass@1
frontier value per rollout is positive moves the frontier; redistributing toward high per-rollout
signal does not; and HORA's per-problem normalization (each problem's terms scaled by G / N_i)
cancels the expected effect of extra rollouts.

## Arms (seeds 8, 9, 10; Pre-registration 12's settings: 16 problems per step, AdamW 1e-6, 150 steps, 512 tokens; same problems per step as B)

All three use exactly 128 rollouts per step (equal rollout budget to B; generation runs in two calls):
stage 1 draws 4 rollouts per problem; stage 2 distributes the remaining 64 (at most 32 rollouts per
problem) as below; stage-1 rollouts are pooled into each problem's group; Dr. GRPO advantages within
each group.
- **AC (frontier-value allocation):** extras in proportion to the frontier value of one rollout under
  the model, v(c) = max(0, E[p^2 (1 - p)^2 (16 (1 - p)^15 - lambda)]), p ~ Beta(c + 1/2, 4 - c + 1/2),
  lambda = 0.37 (Dr. GRPO's observed pass@16-vs-pass@1 slope, steps 50-150). With 4 pilot rollouts only
  c = 0 has positive value, so the extras go to problems with no pilot success. Dr. GRPO's fixed normalizer.
- **AV (signal allocation, control):** extras in proportion to E[p (1 - p)] under the same posterior.
  Fixed normalizer.
- **HR (HORA, 2605.07114, control):** greedy on the marginal posterior hit gain with a Beta(1, 1)
  prior; each problem's terms scaled by 8 / N_i (HORA's per-problem normalization). Adaptation: 4 pilot
  rollouts of 8 (HORA: 8 of 32), at most 32 per problem.
References: Pre-registration 12's B (trajectory and temperature envelope); Pre-registration 14's G32.

## Forecasts (pass@16 gain over B at matched pass@1 0.37 / 0.40 / 0.44; c = 0 stratum at 0.44)

| Model variant | AC | AV | HR |
|---|---|---|---|
| Frozen law (32-sample base counts, tau -0.20) | +3.91 / +4.32 / +4.79 (mean +4.34); +6.7 | -1.09 / -1.14 / -1.17 (mean -1.13) | +0.12 / +0.16 / +0.17 (mean +0.15) |
| Deep-sample counts (3 of 4 shards), tau_deep +0.20 | +2.36 / +2.55 / +2.78 (mean +2.56); +4.8 | -0.70 / -0.72 / -0.73 (mean -0.72) | +0.06 / +0.06 / +0.09 (mean +0.07) |

## Evaluation

The 511 problems, 32 samples, steps 50, 100, 150 at T = 1; greedy accuracy on 3,382 test problems
at step 150. Pre-registration 12's interpolation and envelope; 2,000 shared bootstrap draws; clamping
reported.

## Decision rules

1. **AC moves the frontier:** gain over B's envelope at least +2 pp, 95% interval above 0, positive
   in all three seeds; otherwise Pre-registration 12's categories. Step-150 pass@1 minus B reported
   with its 90% interval.
2. **Allocation target matters:** AC against AV's trajectory at least +1 pp, 95% interval above 0.
3. **Beyond HORA:** AC against HR's trajectory at least +1 pp, 95% interval above 0.
Holm over rules 1-3. If rule 1 passes: confirmation on the fresh 512 problems
(`data/confirm_fresh_512.json`), same sign and 90% interval excluding 0.
4. **Forecast accuracy (reported):** for each arm, the observed gain over B's envelope compared with
   the forecast mean of each model variant; a forecast counts as confirmed if within 1.5 pp; the
   ordering AC > HR > AV is compared with the observed ordering.

Reported without a rule: group sizes by pilot count; pass@16 by base-count stratum; each arm against
G32's trajectory where pass@1 ranges overlap; generation time. Prior work: HORA (posterior hit
utility, per-problem normalization), Knapsack RL (2509.25849; mixed-group probability times
information gain), DARS (2508.13755; extra rollouts to low-accuracy problems at a larger budget, with
Pass@1-Pass@K curves), AR3PO (2509.25808; adaptive stopping and replay).

## Cost

About $150 training (nine runs) and $130 evaluation.
