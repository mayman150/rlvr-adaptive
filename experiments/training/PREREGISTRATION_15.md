# Pre-registration 15: do two 2026 diversity methods move the curve at matched accuracy?

Declared October 21, 2026, before any training run of this design. Designed with an independent
reviewer (Codex).

## What was known

Pre-registration 12's final result (coverage memory +1.19, average agreement). A literature search
(`paper/RELATED_2026.md`): no 2026 method paper compares at matched pass@1; BBG reports lower pass@1
than GRPO on every benchmark with higher pass@256. Pre-registration 14 is training; none of its
arms has been evaluated.

## Why

Our diagnosis says correctness-only rules, which only reweight questions and change noise, move a
run along one accuracy-diversity curve, so fixed-step diversity gains can be speed. Barakat et al.
(NeurIPS 2026) argue that reweighting toward hard prompts can rotate the update through shared
parameters, so this is an empirical question. Two prominent 2026 methods are tested at matched
accuracy:
- **BBG** (Yuan et al., arXiv 2606.15455): Bayesian boundary gating on W-REINFORCE.
- **MaxRL** (Tajwar et al., arXiv 2602.02710, ICML 2026): maximum-likelihood RL.

## Arms (seeds 8, 9, 10; Pre-registration 12's settings: 16 questions x 8 answers, AdamW 1e-6, 150 steps, 512 tokens; same questions per step as B)

- **BBG:** W-REINFORCE advantages (+0.3 for a correct answer, -1 for a wrong one) times a
  per-problem coefficient alpha(m) = u(m; 8, 16) = 16 prod_{j=1}^{15} (8 - m + j)/(9 + j) if
  u >= 0.01 x 9, else 0 (m = correct answers of 8). All-wrong problems keep their negative signal.
  Adaptation: the paper uses the normalized bucket weights alpha / sum(alpha) and accumulates 512
  active prompts per update; we rescale the coefficients to mean 1 over each step's 16 problems and do
  not accumulate active prompts (inactive problems simply contribute nothing). Conclusions are scoped
  to this recipe.
- **WRF: ungated W-REINFORCE** (positive weight 0.3), BBG's own base, so that BBG's gate is isolated
  (BBG against WRF's trajectory, reported) and BBG against B is not read as the gate's effect alone.
- **MXR (MaxRL):** advantages (r - mean) / mean per group; no gradient for groups without a success
  (the paper's Algorithm 1).
- Adaptations, disclosed: 1.5B model, 150 steps, 512 tokens, on-policy single update (no PPO
  clipping or KL), our verifier and normalizer.

## Evaluation

Pre-registration 12's 511 problems, 32 samples, steps 50, 100, 150 at T = 1; greedy accuracy on
3,382 test problems at step 150. References: Pre-registration 12's B (trajectory and temperature
envelope) and base model.

## Decision rules

Pre-registration 12's procedure with shared bootstrap draws (2,000 over problems), clamping
reported, problem alignment checked.
1. **Each arm against B's envelope** (BBG, MXR, WRF), Pre-registration 12's categories: moves the
   curve (at least +2, 95% interval above 0, all three seeds), average agreement (90% interval within
   +-2), worse, inconclusive. Holm over the three directional tests; "moves the curve" and "worse"
   are claimed only if Holm-significant (otherwise inconclusive); equivalence uses the uncorrected 90%
   interval.
2. **The fixed-step reading next to the matched-accuracy reading** (the point of the audit):
   step-150 pass@16 minus B's and step-150 pass@1 minus B's (90% intervals), reported for each arm.
3. If an arm moves the curve (Holm-significant): confirmation on `data/confirm_fresh_512.json` (Pre-registration 14,
   amendment 3), same sign and 90% interval excluding 0.

Our theory predicts average agreement for both. Reported without a rule: BBG's active-problem share
per step; MaxRL's share of groups without a success; D32 (Pre-registration 14) against each arm's
trajectory when both exist; compute.

## Cost

Training about $135 (three arms), evaluation about $180, smoke test and reserve $40.
