# Pre-registration 18: registered result (October 25, 2026)

All arms and seeds complete (AV seed 8 trained and evaluated on Vulcan, results/vul_D*). Output of
`python analyze_v18.py --node-dirs vul_A_eval,vul_D_named` (bands omitted):

```
problem alignment: 79 evaluation files share one problem order
Rule 1: AC against B's trajectory-and-temperature envelope (511 problems, seeds (8, 9, 10))
   AC: +0.14, 90% [-0.28, +0.51], 95% [-0.36, +0.58], per seed 8: -0.10, 9: +0.44, 10: +0.07 -> average agreement
   step-150 pass@1, AC minus B: -0.76, 90% [-1.04, -0.49]
Rule 2: allocation target matters
   AC vs AV trajectory: +0.51, 90% [+0.11, +0.92], 95% [+0.04, +0.99], per seed 8: -0.05, 9: +0.80, 10: +0.78 -> not shown
Rule 3: beyond HORA
   AC vs HR trajectory: +0.00, 90% [-0.43, +0.39], 95% [-0.52, +0.46], per seed 8: +0.34, 9: -0.66, 10: +0.32 -> not shown
Holm over rules 1-3:
   2. AC vs AV trajectory: p = 0.0410 -> not significant
   1. AC vs B envelope: p = 0.6010 -> not significant
   3. AC vs HR trajectory: p = 0.9620 -> not significant
   Interpolation endpoint clamping:
      1. AC vs B envelope: 395 of 18000 (2.2%)
      2. AC vs AV trajectory: 0 of 18000 (0.0%)
      3. AC vs HR trajectory: 62 of 18000 (0.3%)
Rule 4 (reported): forecast accuracy at matched pass@1 37 / 40 / 44, seed-mean curves
   AC: observed 37: +0.09, 40: -0.20 (mean -0.05)
      frozen law: forecast mean +4.12 -> not confirmed (|diff| 4.17)
      deep-sample counts: forecast mean +2.46 -> not confirmed (|diff| 2.51)
   AV: observed 37: -0.62, 40: -0.27, 44: +0.20 (mean -0.23)
      frozen law: forecast mean -1.13 -> confirmed (|diff| 0.90)
      deep-sample counts: forecast mean -0.72 -> confirmed (|diff| 0.49)
   HR: observed 37: +0.22, 40: +0.29, 44: +0.25 (mean +0.25)
      frozen law: forecast mean +0.15 -> confirmed (|diff| 0.10)
      deep-sample counts: forecast mean +0.07 -> confirmed (|diff| 0.18)
   observed ordering HR > AC > AV (forecast AC > HR > AV)
Reported without a rule
   AC mean group size by pilot count: c=0: 18.9 (1794 groups), c=1: 4.3 (1128 groups), c=2: 4.4 (1157 groups), c=3: 4.4 (1195 groups), c=4: 4.4 (1926 groups)
   AV mean group size by pilot count: c=0: 6.4 (1748 groups), c=1: 9.4 (1108 groups), c=2: 10.3 (1053 groups), c=3: 9.6 (1273 groups), c=4: 6.4 (2018 groups)
   HR mean group size by pilot count: c=0: 9.8 (1744 groups), c=1: 9.2 (1141 groups), c=2: 8.3 (1121 groups), c=3: 7.2 (1218 groups), c=4: 6.1 (1976 groups)
   AV vs B envelope: -0.61, 90% [-1.15, -0.19], 95% [-1.24, -0.09], per seed 8: -0.32, 9: -0.63, 10: -0.87 -> average agreement
   HR vs B envelope: +0.10, 90% [-0.38, +0.55], 95% [-0.49, +0.66], per seed 8: -0.66, 9: +1.26, 10: -0.30 -> average agreement
   step-150 pass@16 by base-count stratum (32 base samples), mean over seeds:
      B: c=0 24.0, c=1-4 90.1, c=5-16 99.9, c>16 100.0 (c=0: 151, c=1-4: 158, c=5-16: 179, c>16: 23 problems)
      AC: c=0 24.8, c=1-4 89.5, c=5-16 99.9, c>16 100.0 (c=0: 151, c=1-4: 158, c=5-16: 179, c>16: 23 problems)
      AV: c=0 26.4, c=1-4 91.1, c=5-16 100.0, c>16 100.0 (c=0: 151, c=1-4: 158, c=5-16: 179, c>16: 23 problems)
      HR: c=0 24.4, c=1-4 90.4, c=5-16 99.9, c>16 100.0 (c=0: 151, c=1-4: 158, c=5-16: 179, c>16: 23 problems)
Greedy test accuracy at step 150 (3,382 problems), mean over seeds
   B: 47.9
   AC: 47.3
   AV: 49.2
   HR: 47.8
```

**Verdict.** Rules 1-3 not met: frontier-value allocation (AC) lies on Dr. GRPO's curve (+0.14, 90% interval
inside +-2), does not beat signal allocation (AV, +0.51, not Holm-significant) or HORA (HR, +0.00), and lowers
step-150 pass@1 by 0.76 points. No confirmation run (rule 1 failed). Forecasts: HR and AV confirmed, AC not
(observed -0.05 against +4.12 frozen, +2.46 deep-sample).
