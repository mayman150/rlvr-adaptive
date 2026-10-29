# Pre-registration 32: outcome-based rarity bonuses and repetition penalties on the matched-pass@1 curve

Declared October 28, 2026, before any run of this design; reviewed by Codex before launch (its corrections applied).

## Why

Earlier experiments included token-entropy bonuses and the unlikeliness reward (Pre-registration 12); this study adds
terms based on the repetition of verifier-defined final-answer outcomes. Song, Kempe and Munos (arXiv 2509.06941,
"Outcome-based Exploration for LLM Reasoning") report, for Qwen2.5-7B base trained on the MATH training split (final
checkpoint, three seeds, Table 4), pass@1/pass@32 of 38.1/59.3 for GRPO and 38.2/61.0 for Batch exploration; UCB-Con
reaches 39.1/57.8. This endpoint comparison motivates testing a matched-pass@1 contrast in our setting.

## Arms (Qwen2.5-Math-1.5B, Dr. GRPO, Pre-registration 21's recipe, seeds 8, 9, 10)

Notation: a problem x, a group of G = 8 answers y_1..y_G, A_i = r_i - mean(r) (Dr. GRPO), and the outcome class a_i
of answer i: verifier-accepted responses share one class; responses with no parsed final box share a synthetic class
("no parsed final box"); every other response's class is its last boxed answer after the verifier's normalization,
rationals compared by value. The loss is

    -(1/Z) sum_i [ A_i sum_{t} log pi(y_i,t) + c B_i sum_{t < s_i} log pi(y_i,t) ],

with Z Dr. GRPO's constant normalizer and s_i the number of response tokens wholly before the last "\boxed" marker
(all tokens if there is none). Only the exploration term excludes tokens from the last box marker on; correctness
credit covers the full response. This adapts the paper's final-answer masking (sec. 3.1), whose purpose is to keep the
bonus from being earned by the answer tokens themselves.
- **OU (UCB-Con, sec. 3.2):** B_i = min{1, N(x, a_i)^(-1/2)} - b0, with N(x, a) the number of times the run has
  sampled outcome a on problem x, this group included; B_i = 0 for every answer of a group whose answers are all
  correct (the paper's exception).
- **OB (Batch, sec. 3.3):** B_i = -(1/G) #{j != i : a_j = a_i}, in every group. The paper states the all-correct
  exception for the UCB variants only, so OB keeps it: in an all-correct group every reasoning prefix receives
  c B_i = -0.175 while the correctness advantage is zero. The frequency of such groups is logged.
- c = 0.2 (the paper's Qwen value) in both arms; b0 = 0.5 in OU.

Differences from the paper, stated in advance:
- **No history.** 150 steps x 16 problems = 2,400 visits to 3,904 training problems in one shuffled pass, so each
  problem is visited at most once and N(x, a) is the count within the group. OU is then a within-group rarity rule
  (+0.5c for a unique outcome, +0.21c for two copies, +0.08c for three, 0 for four, negative from five). Historical
  exploration is not tested. The paper trains 700 steps of 256 problems (about 24 passes over MATH's 7,500).
- **b0.** The paper uses b0 = 1 for MATH (its easy dataset) and 0.5 for its medium dataset. Without history, b0 = 1
  makes UCB-Con a penalty-only rule close to Batch; b0 = 0.5 keeps the positive signal for rare outcomes that
  distinguishes the two arms. This is a deliberate departure from the paper's MATH setting.
- **Coefficient.** c = 0.2 is transferred, not strength-matched. Dr. GRPO does not divide advantages by the group
  standard deviation, so in a mixed group with success rate p the bonus coefficient relative to the advantage scale is
  1/sqrt(p(1-p)) >= 2 times the paper's; no gradient-norm match is attempted.
- **Outcome equivalence** uses our verifier's normalization, narrower than the paper's math_verify: equivalent
  symbolic forms can form separate wrong classes and receive rarity credit.
- Dr. GRPO instead of GRPO with KL 0.001; 1.5B instead of 7B; 16 problems x 8 answers per step instead of 256 x 8;
  Pre-registration 21's optimizer settings.
A null therefore concerns this adaptation; it cannot refute the paper's broader claim.

## Evaluation and decision (511 test problems, 32 samples, T = 1, 1,024 tokens, vLLM; checkpoints 25, 50, 100, 150)

Reference: B1k seeds 8-10 (trajectory and temperature envelope; the trajectory-only contrast is also reported).
Primary metric: pass@16 in percentage points at matched pass@1, averaged over eligible checkpoints within each seed,
then over seeds; Pre-registration 21's estimator and guards (fixed eligibility, at least two eligible checkpoints per
seed, missing evaluations, clamping); significance requirements Holm-adjusted over the two arms (two-sided tests).
- **Checkpoint-averaged gain detected (moves the curve):** pooled gain >= +2, 95% interval above 0, positive in all
  three seeds.
- **Checkpoint-averaged loss detected (lowers the curve):** the mirror rule.
- **Checkpoint-average equivalence within +/-2 points:** the pooled and each seed's 90% interval within +/-2,
  conditional on these trained seeds.
- Otherwise inconclusive; more than 10% clamped in a seed: not interpreted.

**Manipulation check (decides how a null is read).** For each checkpoint and problem, the number of distinct valid
incorrect outcome classes among the 32 test samples ("no parsed final box" excluded), averaged over problems, minus
B1k's value interpolated along the same seed's trajectory (base and steps 25-150, T = 1) at the checkpoint's pass@1;
checkpoints outside that trajectory's pass@1 range are ineligible; averaged as the primary metric; 90% interval from
2,000 problem-bootstrap draws shared with the primary analysis. Reported with it: the conditional collision
probability (for problems with at least two valid incorrect samples, the chance that two of them drawn without
replacement share a class; lower means more diverse), and the no-parsed-box rate. If the pooled interval of the
distinct-class contrast is not above 0, an increase in the registered diversity metric has not been demonstrated, and
the arm's curve result is not read as a test of more diverse answers. Matching aggregate pass@1 does not match the
number of incorrect samples per problem; the collision probability is the measure that conditions on it. These
secondary checks are not multiplicity-adjusted.

**Against the reported magnitude.** pass@32 at matched pass@1 (same estimator and guards), with +1.7 (Batch minus
GRPO in the paper's Table 4) as a fixed reference magnitude: an upper 90% bound below +1.7 excludes that magnitude for
our registered contrast; it does not establish a cross-study difference in effects. Raw step-150 pass@1 and pass@32
minus B1k's are also reported.

Reported: pass@16 gain by base-count stratum (base zero of 32 included); the training logs' bonus statistics (share of
nonzero and positive bonuses, mean bonus, all-correct groups and those skipped by OU, distinct outcomes per group and
in all-wrong groups, no-parsed-box count); the first-block probe entropy (comparable across arms; the update-population
entropy is not).

## Cost (estimate)

Six single-GPU runs plus their evaluations on one 8 x A100 machine (about 5-6 hours, as Pre-registration 31; memory
and time not yet verified for the added bonus pass), with the two remaining GPUs generating Pre-registration 30's
teacher pool (Amendment 1 there): about $170-200 in total.
