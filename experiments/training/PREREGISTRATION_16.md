# Pre-registration 16 (pilot): which push to rare questions moves the curve inside RL?

Declared October 22, 2026, before any run of this design. Runs on Tamia (H100), so every arm,
including the Dr. GRPO reference, is trained and evaluated on the same hardware. Reviewed by Codex
before launch.

## What was known

Pre-registration 12: own-success replay +1.19 at matched pass@1. Pre-registration 14 (interim, not
final): seeded replay about +1 and slower; the warm start (fine-tuning on the base successes, then
Dr. GRPO) is far ahead (step-150 pass@1 51-53 vs 44, pass@16 78-80 vs 75), with its pass@16 gain on
problems the base never solves (24 -> 34), but it is outside Dr. GRPO's accuracy range and the
512-token budget truncates 48% of Dr. GRPO's answers (Check 16 running). The PI asked for a method
inside RL, not a fine-tuning stage. Diagnosis: diversity at matched accuracy moves only with new
signal on rarely solved questions; replay delivered far less push than the warm start.

## Arms (seed 12, fresh; Pre-registration 12's settings: 16 questions x 8 answers, AdamW 1e-6,
150 steps, 512 tokens; the shared discovery archive of Pre-registration 14)

Candidates (inside RL; eligible for selection):
- **AR, annealed replay:** seeded coverage-weighted log-mass replay, lambda 4 falling linearly to 0
  by step 100, 16 replayed questions per step (8 times Pre-registration 14's strength at the start).
- **GC, group completion:** a zero-success group whose question has a stored success (base or own)
  gets it appended as a ninth, correct answer; Dr. GRPO advantages over the 9 answers. Intentionally
  without an importance weight: an off-policy likelihood push on a stored success, not an unbiased
  on-policy estimator (LSPO and LUFFY use importance weighting; their estimator justification does
  not carry over).
- **PG, prefix-guided rollouts:** for questions the base solved at most 4 of 32 times and with a stored
  success, 2 of the 8 rollouts continue from the first f_t of the stored success's tokens
  (f_t = 0.5 (1 - t/100), zero after step 100); the prefix is context (no gradient), the verifier
  grades prefix plus continuation, and prefix plus continuation stay within 512 tokens.
- **MXGC:** MaxRL's advantages with group completion (the strongest push on rare questions).

References (not eligible): **B** (Dr. GRPO), **S32** (warm start), **SD** (warm start plus seeded
coverage replay at lambda 0.5), **MXR** (MaxRL alone, to isolate completion in MXGC).

## Evaluation

The 255 validation problems (labels audited), 32 samples at T = 1, steps 50, 100, 150 (S32 and SD
also 0) and the base model; B also at T = 1.15 and 1.3 at step 150. Not the 511 test problems.

## Selection rule (pilot)

For each candidate: mean pass@16 gain over B's trajectory-and-temperature envelope at matched
pass@1 (Pre-registration 12's interpolation, 2,000 shared bootstrap draws), and step-150 pass@1
minus B's. **Eligible:** all registered evaluations present and at least two of the candidate's
checkpoints inside B's pass@1 range (otherwise "insufficient overlap", not eligible). **Selected:**
the eligible candidate with the largest gain among those with a gain of at least +1.5 and step-150
pass@1 no more than 2 points below B's; candidates within 0.5 of that largest gain count as tied,
and a tie goes to the simpler rule (GC, then PG, then AR, then MXGC). MXGC is also reported against
MXR's trajectory, so that a MaxRL effect is not credited to completion. If none qualifies: no candidate; reported, and the method search
continues. Training logs keep assisted and unassisted rewards apart (prefix successes, capped guided
continuations, completed groups and their tokens). Reported without a rule: each arm against S32's trajectory where ranges overlap; pass@16
on validation problems the base never solves and solves 1-4 of 32 times; completed groups, prefix
successes, replay strength; truncation rates; compute.

The selected candidate goes to a separately registered confirmation (3 seeds, 511 test problems,
fresh 512-problem set, controls including B, S32, and the 2026 methods). One seed and 255 problems
cannot establish an effect; this pilot only ranks.

## Cost

Free (Tamia allocation): about 2 nodes x 3 h of training and 3 nodes x 1.5 h of evaluation.
