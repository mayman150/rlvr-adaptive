# Pre-registration 27: does longer or stronger training leave the base model's curve? (reason 3)

Declared October 28, 2026, before any run of this design; reviewed by Codex before launch.

## Why

Every tested intervention lies on Dr. GRPO's pass@16-vs-pass@1 curve, and at matched pass@1 different methods solve
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

## Readings fixed in advance (problem bootstrap, 2,000 draws, all checkpoints resampled together)

1. **Curve (the frontier law as the reference).** For each checkpoint, the frozen frontier law (law.py, tau = -0.20,
   fitted before this study) predicts pass@16 from the base counts and the checkpoint's pass@1. Reason 3 is supported on
   this reading if, at step 500 or 1,000, an arm's pass@16 exceeds the law's prediction by at least 2 points with a 95%
   interval above 0. Where a checkpoint's pass@1 lies within B1k's 150-step support, Pre-registration 21's matched
   comparison against B1k's trajectory and temperature envelope is also reported.
2. **Coverage beyond the base (Yue et al.'s crossover).** At step 1,000, pass@256 of the arm minus pass@256 of the
   base. Short training leaves the base ahead at large k; reason 3 is supported on this reading if the arm's pass@256
   exceeds the base's by at least 1 point with a 95% interval above 0. Reported: the number of problems solved at least
   once in 256 samples by the arm but never by the base, and the reverse.
3. **Difficulty order (reported).** Spearman correlation across problems between the base success rate (256 samples)
   and the checkpoint's success rate, at steps 150 and 1,000; success on problems the base never solved in 256 samples.
**Overall:** reason 3 is supported if reading 1 or reading 2 is met by either arm; it is not supported if neither is
met and both arms reach a step-1,000 pass@1 at least 3 points above B1k's step 150 (the policy did move further); if
neither arm moves that far, the test is inconclusive.

## Scope

Plain Dr. GRPO without KL control or resets, 4 passes over MATH training problems, one seed. A negative result limits
the claim to this regime; it does not rule out ProRL-style training.

## Cost

One 8 x A100 machine (ml.p4de.24xlarge, $31.56 per hour): training about 8 hours (about 20-30 s per step at 4 GPUs),
evaluation about 2 hours on the same GPUs: about $300-400. Base evaluations on one L40S: about $5.
