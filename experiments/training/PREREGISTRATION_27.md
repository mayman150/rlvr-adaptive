# Pre-registration 27: does longer or stronger training leave the short-run Dr. GRPO frontier? (reason 3)

Declared October 28, 2026, before any run of this design; reviewed by Codex before launch.

## Why

Every evaluated arm lies, over its observed pass@1 support, on Dr. GRPO's pass@16-vs-pass@1 curve, and at matched
pass@1 different methods solve
nearly the same problems at nearly the same rates (Investigation 27), a profile that a per-problem law from base counts
predicts (law.py, 0.82 pp RMSE on 65 checkpoints). Three explanations were proposed: (1) on-policy learning has no
signal on problems the model never solves; (2) the gains are shared skills that lift all problems in base order;
(3) our training is short and gentle (150 steps at learning rate 1e-6), so the policy stays near the base. This
pre-registration tests (3). Longer RL training is reported to expand coverage (ProRL 2505.24864, BroRL 2510.01180),
with KL control and reference resets that we do not use.

## Arms (Qwen2.5-Math-1.5B, Dr. GRPO, seed 8, Pre-registration 21's recipe otherwise)

- **LH1:** learning rate 1e-6 (the baseline's), 1,000 steps (16 problems x 8 answers per step, about 4.1 passes over
  the 3,904 training problems), 1,024 tokens.
- **LH3:** learning rate 3e-6, 1,000 steps.
Trainer: train_dist.py (FSDP2, 4 A100 per arm, fp32 weights with bf16 autocast; its 1.5B parity check passed,
Pre-registration 20). Checkpoints at steps 50, 100, 150, 300, 500, 700, 1,000. No KL term, no reference reset.

## Evaluation (vLLM, 511 test problems, 1,024 tokens, T = 1)

- 32 samples at every checkpoint; 256 samples at steps 150, 500 and 1,000.
- The base model with 256 samples (job worth-p27-base) and at T = 0.4, 0.6, 0.8 with 32 samples (Investigation 27).
Reported for every checkpoint: pass@1, pass@16, truncation, mean length, boxed rate; training reward and length.

## Readings fixed in advance (training seed 8; intervals condition on it; problem bootstrap, 2,000 draws, all
## evaluations resampled together; Holm correction over the two arms within each reading)

Notation: base counts are the base model's successes on each test problem from the 256-sample evaluation
(worth-p27-base, T = 1, 1,024 tokens); a checkpoint's counts come from its 256-sample evaluation where available
(steps 150, 500, 1,000) and its 32-sample evaluation otherwise. The reference "B1k8 step 150" is Dr. GRPO seed 8 at step
150 (32 samples; pass@1 58.15).

A. **Coverage (primary).** For each arm, pass@256 at step 1,000 minus the base's pass@256 (both 256 samples, the
   fraction of problems with at least one success). Met if the point estimate is at least +1 point and the
   Holm-adjusted 95% interval lies above 0 (the +1 floor applies to the estimate; the interval only establishes
   positivity). Reported: the same contrast at step 150; the arm's change from step 150 to step 1,000; the numbers of
   problems with zero successes in 256 base samples but at least one in 256 arm samples, and the reverse. Equal
   finite-budget coverage does not establish new support: a problem with zero successes in 256 samples is not
   impossible for the base.
B. **Short-run law (prediction test, reported).** The frozen frontier law (law.py, tau = -0.20 as frozen for
   Pre-registration 20; theta_0 from law.initial_theta on the 32-sample base counts of the original base evaluation
   (BASE15), the prior from law.fit_prior on them) predicts pass@16 at each checkpoint's 32-sample pass@1
   (strict: no prediction outside the law's reachable range). A departure is a checkpoint at step 500 or 1,000 with
   pass@16 at least 2 points above the prediction and a Holm-adjusted 95% interval above 0. Beyond the pass@1 range on
   which the law was validated, a departure may reflect extrapolation error; it is reported, not counted as support.
   Where a checkpoint's pass@1 lies within B1k's 150-step support, Pre-registration 21's matched comparison against
   B1k's trajectory and temperature envelope is also reported.
C. **Difficulty order (reported).** Spearman correlation across problems between the base's 256-sample success rate and
   the arm's 256-sample success rate, at steps 150 and 1,000; mean success on problems with zero successes in 256 base
   samples.

**Verdict per arm:** "supported" if reading A is met; "not supported" if A is not met and the arm's step-1,000 pass@1
exceeds B1k8 step 150's by at least 3 points (measured gain reported); "inconclusive" otherwise. **Overall:** reason 3
is supported (for this regime) if either arm is supported; not supported if neither arm is supported and at least one
is "not supported"; otherwise inconclusive. Non-significance does not establish equivalence.

## Scope

Plain Dr. GRPO without KL control or resets, 4 passes over MATH training problems, one seed. A negative result limits
the claim to this regime; it does not rule out ProRL-style training.

## Cost

One 8 x A100 machine (ml.p4de.24xlarge, $31.56 per hour): training about 9-12 hours (the 512-token smoke run took
18-19 s per step at 4 GPUs; 1,024 tokens is slower), evaluation 2-3 hours on the same GPUs: 12-15 hours, about
$380-475 plus setup; maximum runtime set to 20 hours. Base evaluations on one L40S: about $5.
