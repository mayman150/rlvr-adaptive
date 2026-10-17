# Stage B results (40 MATH training prompts, 16 answers each, Qwen2.5-Math-1.5B)

Reproduce: `python analyze_b.py results/summary_all.json --block combined`
(first 16 prompts in `results/`, remaining 24 in `results_part2/`; the first run was
stopped by the operating system for memory while a training smoke test ran alongside).

**Measured reliably.** Wrong answers carry larger gradients than correct ones:
F1 = 4.21e5 [3.61e5, 4.82e5] and F0 = 1.17e6 [1.04e6, 1.30e6] (90% bootstrap over prompts),
so F1 / F0 = 0.36 in the declared last-layer blocks, while wrong answers are only 9%
longer (421 versus 386 tokens). Training calibration over all 1.5B parameters gives
F1 / F0 = 0.395 (29 correct, 99 wrong answers), so the asymmetry holds across the model.

**Not measured reliably.** |grad p|^2 is small against per-answer noise (noise ratio of
order 10^3 to 10^4). The three pair-type estimates that theory requires to agree do not
(correct pairs 31, wrong pairs -532, mixed pairs 380), and two difficulty bins have
nominally negative means, which a squared norm cannot have. Answers show no dependence
(mean cosine between answers -0.0006 +/- 0.0006 in the head block, -0.0001 +/- 0.0010 in
the MLP block); the disagreement comes from heavy tails, where a few long wrong answers
with very large gradients dominate raw sums and make prompt-level bootstrap intervals too
narrow. The shape of |grad p|^2 against difficulty is therefore unresolved at this sample
size and needs many more answers per prompt or a tail-robust estimator.

**Consequence.** Which current rule is best, and the size of the optimal rule's gain,
remain open. The advantage fix tested in training does not depend on that shape: in the
high-noise regime the optimal failure-to-success advantage ratio on mixed groups is about
F1 / F0 whatever the shape, and F1 / F0 is the number measured reliably.

## Shape of |grad p|^2 against difficulty (24 prompts, 96 answers each, MLP block)

Reproduce: `python analyze_shape.py results_shape/summary_all.json` (prompts 0 to 9 in
`results_shape/`, 10 to 23 in `results_shape_part2/`; the first run was stopped by Claude
Code for low memory and resumed with smaller generation batches).

Using the unbiased pair statistic with delete-one-answer jackknife errors, prompts with
success rate at most 0.15 or at least 0.85 give a pooled 0.73 +/- 0.60 and prompts between
0.3 and 0.7 give 15.5 +/- 5.4. The ratio of 0.05 rejects a flat shape (predicted 1) and
disfavors p(1-p) (0.36); the weighted fit prefers p^2 (1-p)^2 (chi-square per prompt 0.35,
against 0.54, 0.77, and 0.90 for zero signal). The median-of-means estimates are too noisy at
12 answers per group to be informative and are not used. Conclusion: the signal fades for
very easy and very hard prompts, the regime in which the optimal lookup-table rule is robust.
