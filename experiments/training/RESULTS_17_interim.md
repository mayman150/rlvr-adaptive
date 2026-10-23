# Pre-registration 17: interim results (law frozen at commit e92edc0 before these were computed)

Output: `RESULTS_17_interim_output.txt`. Fit on Pre-registration 12's B only: tau_32 = -0.20, affine beta 2.10.

| Test set | Checkpoints | pass@16 RMSE: law / uniform shift / affine | Verdict | Log-lik law - affine (wins) |
|---|---|---|---|---|
| Fit (B, in-sample) | 9 | 1.06 / 21.29 / 1.28 | (fit) | +2,282 (9/9) |
| **T1 other methods, Qwen** (CM, M, U, D32, U32, C05, G32, MXR, WRF where available) | 56 | **1.21** / 20.07 / 2.45 | **supported** | +17,894 (54/56) |
| **T2 warm start** (from its own step-0 counts) | 9 | **3.56** / 9.04 / 7.99 | **falsified** | +4,232 (9/9) |
| T3 longer training | – | pending (450-step runs training) | – | – |
| **T4 OLMo-2-1B** (tau transferred from Qwen) | 50 | **1.90** / 9.85 / 12.58 | **supported** | +13,975 (50/50) |
| **T5 1,024 tokens** | 4 | **0.85** / 11.48 / 1.27 | **supported** | +2,434 (4/4) |

Strata (mean observed minus law, pass@16): T1 within 0.2 in every stratum; T4 never +6.0, rare
-6.8 (errors cancel in the aggregate); T2 never -11.9 (the warm start's never-solved problems gain
much less during RL than the law predicts). Reported without a rule: tau refitted on the warm
start's trajectory -0.65 to -0.70 in all three seeds (vs -0.20 for RL from the base model); OLMo
refitted tau -0.10 and +0.10.

Reading so far: from a base model, the law predicts the accuracy-diversity frontier of 11 training
rules, a second model family, and a larger budget within about 1-2 points of pass@16 using only the
base model's per-question counts and each checkpoint's pass@1. It fails after a warm start, where
RL lifts never-solved problems less than near-independent learning predicts (a strongly negative
fitted transfer). Pending: tau_deep (deep base sampling) and T3.

## Deep base sampling (3 of 4 shards: 232 of the 309 problems with c <= 4; shard 3 pending)
- Of the problems with 0/32 base successes, 53% are solved at least once in 256 further base
  samples; their mean base success rate is 0.93% (the 32-sample empirical-Bayes prior assumed 1.77%).
  Problems with 1-4/32: 99% solved at least once, mean rate 8.25%.
- **tau_deep = +0.20, 90% [+0.10, +0.35]** (bootstrap over problems, 200 draws): with the base
  difficulty measured precisely, the fitted transfer term is positive and its interval excludes 0.
