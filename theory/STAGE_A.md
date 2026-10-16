# Stage A: exact theory and the optimal collection rule (first results, October 16, 2026)

Reproduce: `python3 exact.py`, `python3 lattice.py`, `python catalog.py`, `python sensitivity.py`
(the last two need numpy).

## Exact machinery

- `exact.py` enumerates every answer sequence for softmax prompts and checks, for rules
  with fixed and random group sizes, that the mean update is c_S(p) times grad p with c_S
  depending only on p, and that the second moment equals the four-number formula. It also
  confirms the two breakers (token-mean length normalization, non-binary rewards).
- `lattice.py` represents any reward-adapted rule on the lattice of (failures, successes)
  counts and computes its mean coefficient, second-moment coefficients, and expected cost
  as exact finite sums. It matches the brute-force enumeration exactly (18 checks).
- `optimal.py` finds the rule with the highest training value per generated answer.

## Criterion

Prompts come from a difficulty distribution pi(p) and are processed by one rule. With
gradients of different prompts orthogonal and the step size chosen optimally, the
improvement of mean accuracy per generated answer is proportional to

    Eff = Sig^2 / (Cost * M2),  Sig = E[c(p) |grad p|^2],  M2 = E[second moment],  Cost = E[answers].

Scaling advantages leaves Eff unchanged, so the optimum is found by maximizing
Sig - M2/2 - theta * Cost (additive over lattice states, solved by dynamic programming over
stopping and advantages) and choosing theta by bisection (Dinkelbach). A consistency check
turns the solved policy back into a rule and evaluates it with the independent lattice
engine; the two agree to six significant digits.

Declared model of the per-prompt numbers: F0 = F1 = R (constant per-answer Fisher; real
models have R much larger than 1) and |grad p|^2 following one of three shapes.

## Results (R = 10^4; R = 100 gives nearly the same numbers)

Efficiency relative to the optimal rule, |grad p|^2 proportional to p^2 (1-p)^2:

| Rule | uniform | mostly hard | U-shaped | mostly easy |
| --- | --- | --- | --- | --- |
| fixed group of 8 (GRPO, Dr. GRPO, DAPO alike) | 0.61 | 0.38 | 0.40 | 0.38 |
| fixed group of 16 | 0.65 | 0.41 | 0.43 | 0.41 |
| fixed 8, std-normalized | 0.57 | 0.35 | 0.36 | 0.35 |
| probe 4, abandon if agree, fill to 16 (SARA-style) | 0.77 | 0.60 | 0.67 | 0.60 |
| until 1 correct, cap 16 (sample-until-correct) | 0.26 | 0.09 | 0.12 | 0.30 |
| until 2 correct and 2 wrong, cap 16 | 0.41 | 0.22 | 0.22 | 0.22 |

DAPO's dynamic sampling has exactly the per-answer efficiency of the fixed group it
filters: dropped groups cost their answers and contribute nothing, and kept groups are
unchanged. Its practical benefit must come from batch composition or step normalization.

Sensitivity to the shape of |grad p|^2:

| Shape | Best rule in use | Its efficiency | Optimal gain over it |
| --- | --- | --- | --- |
| p^2 (1-p)^2 | probe 4, fill to 16 | 0.60 to 0.77 | 1.29x to 1.66x |
| p (1-p) | probe 4, fill to 16 | 0.75 to 0.87 | 1.15x to 1.33x |
| constant | fixed 8, std-normalized | 0.24 to 0.44 | 2.3x to 4.2x |

**Structure of the optimal rule** (main model, uniform prior): sample at least 3 answers;
continue while the number of successes stays inside a band that widens with the number of
answers; stop when outcomes are lopsided. Advantages use a posterior baseline rather than
the group mean, so all-wrong groups still push their failures down. Optimal advantages
on a fixed group of 8 recover only about 5% of the gap; the stopping band does most of the
work.

## Gate assessment

The Stage A gate asks whether the optimal rule beats the best current rule by a material
margin across a realistic range. Under all three signal shapes it does, by at least 15%
and usually 30% to over 100%. Which current rule is best, and the size of the gain,
depend on how |grad p|^2 varies with difficulty, which Stage B measures directly.

## Caveats

Orthogonal gradients across prompts; one-step improvement; mean accuracy as the target;
a known difficulty distribution (estimable from training history); cost counted in
answers, not tokens; on-policy updates without clipping or token normalization. The
optimal advantages depend on the difficulty distribution.

## Next

Stage B: measure p, |grad p|^2, F0, and F1 per prompt on Qwen2.5-Math-1.5B (unbiased pair
estimator for |grad p|^2, exact per-answer gradients from the STOP tooling), fit the shape
of |grad p|^2 against p, and rerun the comparison on the measured numbers. In parallel:
write the proofs for random group sizes, a two-parameter "band" rule that approximates the
optimum for practical use, and cost in tokens.
