# Pre-registrations 18 and 19: results so far (October 24, 2026, 18:30 UTC)

Computed with `analyze_v18.py` / `analyze_v19.py` functions (Pre-registration 12's interpolation, 2,000 shared
bootstrap draws). HR, HP, HQ (seeds 8, 10), HS were trained and evaluated on Vulcan (results/vul_*); the rest on
AWS. AV seed 8 is running on Vulcan (the AWS copy waited 22 hours for capacity and was stopped before starting).

## Pre-registration 18 (rollout allocation at 128 rollouts per step)
- Rule 1, AC against B's envelope, seeds 8-10: +0.14, 90% [-0.28, +0.51], 95% [-0.36, +0.58]; per seed -0.10,
  +0.44, +0.07 -> rule 1 not met (needs +2); the 90% interval lies in [-2, +2] (average agreement).
- Rule 3, AC against HR's trajectory: +0.00, 95% [-0.52, +0.46] -> not shown.
- Rule 2, AC against AV's trajectory, seeds 9-10 only: +0.79, 95% [+0.19, +1.38] (needs +1); pending AV seed 8.
- Reported: HR against B's envelope +0.10, 95% [-0.49, +0.66]; AV (seeds 9-10) -0.75, 95% [-1.50, -0.14].
- Rule 4 (forecasts, seed-mean curves): HR observed +0.22/+0.29/+0.25 at pass@1 37/40/44 (forecast +0.12 to
  +0.17: confirmed); AV (two seeds) about -0.75 (forecast -0.7 to -1.1: confirmed); AC +0.09/-0.20 at 37/40
  (forecast +3.9/+4.3 frozen, +2.4/+2.6 deep: not confirmed).
- Allocation as designed: AC gives pilot-zero problems 18.9 rollouts on average (others 4.3-4.4); 50-54% of
  pilot-zero groups then contain a success, against 18-21% under AV (6.4 rollouts). HORA: 9.8 to 6.1 by pilot count.
- Step-150 pass@1, AC minus B: -0.2 to -1.3 per seed; greedy test accuracy AC 47.1-47.4, B 47.5-48.2, HR 47.3-48.3.

## Pre-registration 19 (human-solution fine-tuning, then Dr. GRPO)
- c = 0 solutions retained after the trainer's filters: 434-435 of 487 per seed (rule 1 requires 400).
- Rule 1, HQ against HP's trajectory: **not interpretable (insufficient overlap)**: after fine-tuning, RL moves
  pass@1 by at most 1.6 points in 150 steps (HP 42.7-45.3, HQ 42.6-44.9), so no HQ checkpoint of seeds 9 and 10
  falls in HP's pass@1 range. Step-150 pass@16 on the 151 problems with no base success, HQ minus HP:
  +0.58, 90% [-2.13, +3.43].
- Both arms lie above B's 150-step curve at 512 tokens (pass@16 79-81 at pass@1 43-45; B 74-75 at 44), but only
  6-7% of their sampled answers are truncated at 512 tokens against 48% for B (answers about 150 tokens). Check
  21 tests whether this is a budget effect.
- Greedy test accuracy at step 150: HQ 45.5-46.5, HP 44.1-46.3 (B 47.5-48.2).
