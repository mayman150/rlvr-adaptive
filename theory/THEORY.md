# Reward-adapted rollout collection with binary rewards: the two-moment theorem

## Setting

A prompt x has policy pi(y | x) with score l(y) = grad log pi(y | x) and binary reward
R(y) in {0, 1}; its success probability is p = P(R = 1). A **reward-adapted rule** draws
answers y_1, y_2, ... independently from pi, observes their rewards, and

- stops at a time N that is a stopping time for the reward sequence and is bounded by
  n_max (this covers fixed groups, probes that abandon, sampling until k successes,
  balanced exits, and filtering), then
- returns g = sum_{i <= N} A_i l(y_i), where each advantage A_i is a function of
  (r_1, ..., r_N). Abandonment and filtering are A = 0.

## Two facts about one answer

Write kappa(1) = 1/p and kappa(0) = -1/(1 - p), and F_r = E[|l|^2 | R = r].

1. E[l 1{R = 1}] = sum_y pi(y) grad log pi(y) 1{R(y) = 1} = grad p, and E[l] = 0, so
   E[l | R = r] = kappa(r) grad p.
2. Conditional on its reward, an answer is independent of every other answer and its
   reward.

## Theorem

For every reward-adapted rule,

    E[g] = c(p) grad p,                      c(p) = E[ sum_{i<=N} A_i kappa(r_i) ],
    E|g|^2 = F1 D1(p) + F0 D0(p) + |grad p|^2 X(p),

with D1 = E[sum_{i<=N, r_i=1} A_i^2], D0 = E[sum_{i<=N, r_i=0} A_i^2], and
X = E[sum_{i != j <= N} A_i A_j kappa(r_i) kappa(r_j)]. The functions c, D1, D0, X
and the expected cost E[N] depend only on the rule and p. So the first two moments of
the update depend on the prompt only through p, |grad p|^2, F0, and F1.

**Proof.** Couple the rule to an infinite i.i.d. sequence y_1, y_2, ... and let r be its
reward sequence. N and the A_i are functions of r. By fact 2, conditional on r the
answers are independent with E[l(y_i) | r] = kappa(r_i) grad p (fact 1) and
E[|l(y_i)|^2 | r] = F_{r_i}. Hence
E[g | r] = sum_{i<=N} A_i kappa(r_i) grad p and
E[|g|^2 | r] = sum_{i<=N} A_i^2 F_{r_i} + sum_{i != j} A_i A_j kappa(r_i) kappa(r_j) |grad p|^2.
Taking expectations over r, whose law depends only on p, gives both statements. QED.

No optional-stopping argument is needed: conditioning on the whole reward sequence turns
the random group size into a constant.

## Batches

For independent prompts, E|sum_x g_x|^2 = sum_x E|g_x|^2 + sum_{x != x'} c_x c_x' <grad p_x, grad p_x'>,
so the batch second moment also needs the cross-prompt gradient inner products. Dynamic
sampling with refill is covered prompt by prompt (keeping is a reward function, so it is
part of A); dividing the batch by the random number of kept groups is not exactly covered,
because the denominator couples prompts.

## Where the structure breaks

- **Token-level length normalization.** Weights such as 1/|y_i| or |y_i| / sum_j |y_j|
  depend on answer lengths, which are not functions of the rewards, so
  E[l w(|y|) | R] is not along grad p in general (exact counterexample in `exact.py`).
  The mean gains a term from the covariance of length and score given the reward.
- **Non-binary rewards.** E[l | R = v] = grad P(R = v) / P(R = v), one direction per reward
  level, so a rule whose coefficients are nonlinear in the reward leaves the span of
  grad E[R] (exact counterexample in `exact.py`; prior statements in the literature).
- **Clipping, importance ratios, stale samples.** Fact 1 uses on-policy sampling.

## Prior work this extends

Davis and Recht give the mean for fixed-size leave-one-out weightings, with exact
implicit objectives; ARCUS proves fact 1 and the dynamic-sampling corollary; RL2ML gives
the covariance of success-only estimators with fixed group size. The theorem above adds
random group sizes and the full second moment with both F0 and F1.
