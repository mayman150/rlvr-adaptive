# The story of the paper (living document, updated October 18, 2026)

## October 18: external review and the revised claim

Codex, asked to review as an ICML reviewer: **reject as it stands (4/10), credible path**;
about 6/10 if OLMo shows one curve with intervals, 5/10 if curves robustly differ.
Accepted corrections:
- Equal means for PSR and NSR are classical baseline invariance (Williams 1992; Greensmith,
  Bartlett, Baxter 2004). Unbiased pass@k objectives exist (PKPO, Walder and Karkhanis 2025).
  Ours: the second-moment accounting with F0 != F1 and adaptive group sizes, the measured
  asymmetry, and matched-accuracy tests of the NSR explanation. No "nobody measured" claims.
- Qwen "one curve" was absence of detected separation (128 problems, two seeds), not
  equivalence. The NSR paper used 4B to 8B models and k up to 256; we do not overturn it.
- Noise predicts speed; that it explains NSR needs an intervention.
- Disclosure: in Pre-registration 6 (verl-like), gradient clipping at 1 was active on 26 to
  28% of NSR steps and 7 to 8% of PSR steps; never in other on-policy runs (max norm 0.48).

Plan: Pre-registration 9 (OLMo, equivalence with intervals; running) and Pre-registration
10 (PSR plus random-sign noise matched to NSR's predicted noise: same mean, same noise, no
systematic push on wrong answers; if as slow as NSR, noise size suffices to produce the lag,
if clearly faster, size alone does not account for it, without identifying the cause; NSR
and PSR at 2x learning rate; fresh seeds 5 to 7 for a fresh-only Qwen equivalence test, seeds
3 and 4 as a disclosed secondary; same-policy NSR shadow; F0/F1 re-measured; Adam statistics).

Claim to submit if both hold: *positive and negative reinforcement share their expected
update; NSR's extra term is zero-mean noise concentrated on wrong answers, which carry
larger gradients; the size of that noise, predicted from four numbers per question, sets how
fast a run moves along an accuracy-versus-diversity curve that reward shaping does not move.*
Fallback: TMLR, as a scoped diagnostic paper.

## Final results (October 18): one curve

All eight pre-registered experiments are finished. Every rule we tried (PSR, NSR, five
baselines in between, the NSR paper's verl-like implementation, and an exact pass@4 objective)
traces the same accuracy-versus-diversity curve (pass@32 at a given pass@1), also when
reasoning is graded (CoT-Pass@k, Claude Haiku 5.5 judge). Rules differ only in speed along
the curve, and speed follows the theorem's predicted noise (dose-response, both seeds,
rank correlation 0.9 to 0.97). The optimal answer-collection rule moves faster along it at
equal cost (+1.1, +3.3 points). WORTH-k is dropped: changing the objective did not move the
curve at this scale. Remaining gap: a second model family, longer training, larger k.

## Current direction (October 17): positive and negative samples, same mean, different noise

**Why we changed.** A literature review (three agents, key claims verified by hand) found
that the critical-budget framing is already published: Bansal, Mohri, Abreu,
Alvarez-Melis, Kakade, "Critical Batch Size for LLM Policy Optimization" (HiLD at ICML
2026, verified on OpenReview), which splits GRPO noise into inter- and intra-prompt terms
and finds the intra-prompt term dominant; Li et al. (arXiv 2608.29296) on square-root
learning-rate scaling; IsoCompute (arXiv 2603.12151) on rollouts per problem.

**What is ours.**
1. Wrong answers carry about 2.5 times larger squared gradients than correct ones. No
   paper measures gradient norms by correctness, and two 2026 variance papers assume the
   opposite (VSR, arXiv 2610.15888, Assumption B.3, verified verbatim; BICC 2603.13134).
2. The exact second moment of any reward-adapted rule with separate F0 and F1 and random
   group sizes (the mean part is known: Davis and Recht; ARCUS Corollary 1).
3. A consequence nobody states: on-policy, positive-only and negative-only reinforcement
   have exactly the same expected update, n grad p. The famous difference between them
   (Zhu et al., NeurIPS 2025: negatives preserve entropy and pass@k) must live in the noise,
   which NSR puts on hard prompts and makes 2.5 times larger per answer, or in
   implementation choices.

**Pitch.** *Rewarding right answers and penalizing wrong ones push the model the same way.
Everything that separates them is noise, and wrong answers carry most of it.*

**Tests.** Pre-registration 4: Dr. GRPO, PSR, NSR, W-REINFORCE, strictly on-policy, two
seeds, with in-training measurement of each arm's mean and noise, an on-policy entropy
probe, and pass@64. Next, depending on the outcome: implementation factors (token-mean
aggregation, clipped reuse) or optimizer (Adam versus SGD), and a second model family
(Qwen2.5-Math is suspect for small effects; Spurious Rewards, arXiv 2506.10947).

**Results so far (October 17, afternoon).**
- Pre-registration 4, seed 1 (seed 2 running): pass@64 NSR 83.0 vs PSR 83.2 (minus 0.3;
  pre-registered threshold 3 points, not met); pass@1 PSR 42.4 vs NSR 37.7; entropy over
  steps 100 to 150 NSR / PSR = 1.11 (threshold 1.20, not met). Base model pass@64 79.9.
- The second-moment theorem holds in the real model: measured per-prompt second moment /
  theorem = 0.71 to 1.00 for all arms and seeds (pre-registered tolerance 30%).
- NSR = PSR minus sum_i grad log pi(y_i), a zero-mean term (exact identity). Its noise is
  6.4 times PSR's (theorem: about 7), and it learns more slowly (first-block training reward
  33.6 vs 51.3 over steps 25 to 49), as a noisier estimator should.
- Equal-mean test (projection on the population gradient) is underpowered: 90% intervals
  include both 1 (theory) and 0.
- Working claim: negative reinforcement is positive reinforcement slowed by extra noise.
  Pre-registration 5 tests it directly: at matched pass@1, equal pass@32 (seeds 3 and 4).
- Rollout rules at matched answers (Pre-registration 3): optimal rule +1.1 (0.4) and +3.3 (0.5) points on
  the 3,382-problem set in seeds 1 and 2 (pre-registered 1.5-in-both rule not met because of seed 1); probe-then-fill +2.0 and +0.8 on
  MATH-500; scaled failures -0.4 and +1.1. Noise scale grows from about 900 answers early
  to over 10,000 later.

**Second section.** Exact accounting of adaptive rollout rules (dynamic sampling has the
per-answer efficiency of the group it filters; sample-until-correct is worst), set against
the literature's accounting flaws: gains plotted against steps while spending 2 to 3 times
more answers, learning rate not retuned, single seeds, 30-problem benchmarks.

---

## Earlier framing (kept for the record)

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
   times efficiency in 11 of 12 settings. Pre-registered two-seed pilot: no evidence (minus 2.2 and minus 2.7 points).
6. **Why a better rule can be worth nothing (new, October 17).** Without orthogonal
   prompts, a rule's one-step progress at B answers per step is
   <grad J, mu>^2 / (2L (s + nV/B)). The critical budget B* = nV/s (the rule's gradient
   noise scale in answers) splits two regimes: below it a rule with k times the efficiency
   is worth k times the answers (about sqrt(k) under Adam at a fixed learning rate); above
   it no reweighting helps and the unweighted estimator is best. Every term is measurable
   inside a training run, so the theory says not only which rule is best but whether it
   matters at your budget.
7. **Tests of the regime theory.** A budget ladder (Dr. GRPO at 32 to 256 answers per step)
   with in-training measurements of B*, the optimal rule at 64 and 128, and the scaled rule
   at 128, two seeds (Pre-registration 3).

New one-line pitch: *a theory of what each rollout is worth, and of when being smarter
about rollouts is worth anything at all.*

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
| Training pilot, seed 2 | Done. Baseline 46.7%, scaled 44.0% (minus 2.7). Pre-registered verdict over both seeds: NO EVIDENCE (mean minus 2.5 points) |
| Critical-budget theorem (no orthogonality) and Adam reading | Written (theory/THEORY.md, paper Section 4) |
| Budget ladder and in-training diagnostics, 14 runs | Launched October 17; 3 training, the rest waiting for GPU capacity |
| Bibliography author names | Several placeholders to fill before submission |
