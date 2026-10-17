# The story of the paper (living document, October 16, 2026)

Draft: `worth_paper.pdf` (source `worth_paper.tex`). Red "pending" markers show where
running results go.

## One-paragraph pitch

RL for reasoning spends most of its compute generating answers, and every modern
pipeline decides adaptively how many answers each prompt gets: drop prompts whose
answers all agree, abandon after a short probe, sample until a correct answer. These
rules are compared by heuristics. We show that with binary rewards, any such rule is
fully characterized, in its effect on the training signal, its noise, and its cost, by
four numbers per prompt. That turns rollout collection into an exact optimization
problem, ranks the rules in use against the optimum, and, after measuring the four
numbers in a real model, yields a concrete prediction: wrong answers carry about 2.5
times larger gradients, so their advantages should be scaled down by that ratio.

## The arc

1. **Problem.** Adaptive rollout rules are everywhere and silently change objective,
   noise, and cost at once.
2. **Insight.** For binary rewards, an answer's expected gradient given its reward points
   along the gradient of the success probability. Conditioning on the reward sequence
   makes random group sizes trivial, so the first two moments of any reward-adapted rule
   depend on four numbers per prompt.
3. **Consequences.** Exact efficiency for every rule: DAPO filtering equals the fixed
   group it filters; sample-until-correct is worst; probe-then-fill is best in use; the
   optimal rule beats it by 1.6 to 2.2 times at the measured signal shape and gradient
   ratio.
4. **Measurement.** In Qwen2.5-Math-1.5B, wrong answers have 2.5 to 2.8 times larger
   squared gradients (last layer and full model), while being only 9% longer, and the
   gradient signal fades for very easy and very hard prompts, which makes the optimal
   rule robust.
5. **Prediction and test.** Scale failure advantages by F1/F0: predicted 1.12 to 1.56
   times efficiency in 11 of 12 settings. Pre-registered two-seed training test.

## How the story changes with the pending results

| Training result | Shape measurement | Paper |
| --- | --- | --- |
| Scaled arm wins (pre-registered "promising") | fading shape | Strongest version: theory, optimal rule deployable, measured asymmetry, and a training gain. Run the optimal lookup-table rule as a third arm and scale to more seeds. ICML candidate. |
| Scaled arm wins | flat or unresolved | Theory plus a validated practical fix; the optimal rule is presented as conditional on the shape. Solid ICML or TMLR paper. |
| No evidence | fading shape | The accounting and the optimal rule stand; the one-step criterion is shown not to predict training here. Run the optimal rule anyway before concluding. TMLR. |
| No evidence | flat or unresolved | A characterization paper: exact implicit objectives, variances, and costs of every rule in use, plus the measured asymmetry. TMLR or workshop. |

## Honest weaknesses to address in the writing

- The mean result for fixed groups is known (Davis and Recht; ARCUS); our novelty is random
  group sizes, the full second moment with both F0 and F1, the optimal rule, and the
  measurements. State this plainly.
- The efficiency criterion ignores exploration; work on learning from wrong answers
  suggests they may matter for pass@k beyond one-step progress.
- The optimal rule is fragile if the signal shape is flat; the scaled-failure fix is
  robust in 11 of 12 settings, and the exception (mostly easy prompts, flat signal) should
  be stated.
- One model, a two-seed pilot, and greedy accuracy that partly measures format learning.

## Status

| Item | State |
| --- | --- |
| Two-moment theorem and proof | Done, exactly verified |
| Lattice engine, optimal rule, catalog, sensitivity, robustness | Done |
| F1 / F0 asymmetry (last layer and full model) | Measured |
| Shape of the gradient signal against difficulty | Done: fades at the extremes (ratio 0.05; flat rejected), optimal rule robust |
| Training pilot, seed 1 | Done. Baseline 47.5%, scaled 45.3% (minus 2.2 points, within evaluation noise). Training reward higher for scaled in the first 100 steps, tied afterwards (0.586 both over the last 50 steps) |
| Training pilot, seed 2 | Baseline 46.7%; scaled arm running |
| Bibliography author names | Several placeholders to fill before submission |
