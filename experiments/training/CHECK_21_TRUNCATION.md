# Check 21: is the human-solution arms' position above Dr. GRPO's curve a 512-token truncation effect?

Declared October 24, 2026, before HP and HQ are evaluated at 1,024 tokens, after seeing Pre-registration 19's
512-token results (HP, HQ: pass@16 79-81 at pass@1 43-45, against Dr. GRPO's 74-75 at 44; 6-7% of their sampled
answers truncated at 512 tokens, against 48% for Dr. GRPO) and Check 16's partial 1,024-token results. Revised
after Codex's review. Also completes Check 16, whose spot evaluations timed out (missing: B seed 9 steps 50 and
100; S32 seed 9 steps 0 and 50; S32 seed 10 all steps); Check 16's own registered readings are kept separate.

**Why.** After fine-tuning on human solutions the policy answers in about 150 tokens; Dr. GRPO's answers
average about 410 and half are cut at 512. A higher pass@16 at matched pass@1 can then mean "finishes within
the budget" rather than "the curve moved". At 1,024 tokens Dr. GRPO's step-150 truncation is 15.5% (Check 16,
seed 10): still material, so a 1,024-token result does not describe an unconstrained curve.

**What.** Existing checkpoints re-evaluated with answers up to 1,024 tokens (no new training), same 511
problems in the same order, 32 samples, T = 1, the same checker and sampling seeds as at 512: B seeds 8, 9, 10
at steps 50, 100, 150 (seed 10 and seed 9's step 150 exist from Check 16; the base model exists); HP and HQ
seeds 8, 9, 10 at steps 0, 50, 100, 150; Check 16's missing S32 lanes. A manifest records, per evaluation, the
checkpoint, budget, problem-order hash, decoding settings, sampling seed and whether the file is reused;
timed-out evaluations stay missing.

**Measure.** For arm X in {HP, HQ}, seed s, checkpoint t (steps 0-150) and budget b: the gap
g(X, s, t, b) = pass@16 of X minus B's pass@16 interpolated along B's trajectory of seed s at budget b (base
model, steps 50, 100, 150; Pre-registration 12's interpolation, sorted by pass@1, ties by step) at X's pass@1.
**Common support:** (s, t) is eligible if X's pass@1 falls inside the same-seed B range at both budgets; the
eligible set is fixed on the full data before bootstrapping. G_b is the mean of g over eligible (s, t), each
with equal weight (an equal-seed-weight version is reported as a sensitivity). Intervals: 2,000 shared
bootstrap draws over problems across budgets, arms and seeds, recomputing the interpolation in each draw and
clamping only resampled points that leave the range; clamping is reported per seed and budget, and more than
10% clamped makes the comparison uninterpretable. Intervals are conditional on the trained seeds.

**Reading fixed in advance.** Applies only if G_512 is established (90% interval above 0); otherwise
"no 512-token gain to explain". **Largely a budget effect** if the paired contrast G_1024 - 0.5 G_512 has its
90% interval wholly below 0; **gain largely retained at 1,024** if G_1024 - 0.8 G_512 has its 90% interval
wholly above 0 and G_1024's interval is above 0; otherwise **intermediate or inconclusive**. A three-seed
reading requires at least two eligible checkpoints in every seed; otherwise "insufficient overlap".
If B's step-150 truncation at 1,024 exceeds 20% in any seed, every reading is qualified as "truncation still
binding at 1,024". If the gain is largely retained at 1,024, the step-150 checkpoints of B, HP and HQ are
evaluated at 2,048 tokens as a sensitivity before any claim that human solutions move the curve.

**Reported.** Per seed, checkpoint and budget: pass@1, pass@16, the interpolation segment used, eligibility,
the gap, truncation and mean answer length; truncation by base-difficulty stratum; HQ against HP at 1,024 with
Pre-registration 19's rule-1 procedure and interpretation guards and the c = 0 stratum contrast (without a
verdict). Scope: conclusions are relative to Dr. GRPO's 150-step trajectory at T = 1; the comparison with the
longer Dr. GRPO run at 1,024 tokens is a separate decision.

**Cost.** About 30 single-checkpoint evaluations (the human-solution arms' short answers are cheap), on
Vulcan (checkpoints stored there) and AWS (the others).
