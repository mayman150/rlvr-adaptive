# Pre-registration 12: a method that moves the accuracy-versus-diversity curve

Declared October 19, 2026, **before any result of Pre-registration 11** (see this file's commit
time; Pre-registration 11's evaluations were still running). Which branch runs is decided by
Pre-registration 11's pre-registered verdict, so the method choice is itself fixed in advance.
Revised once before launch after an independent review (Codex): floor enforcement and failure
handling, bounded memory in replay, a label audit of the evaluation and validation sets, pilot
details, executable envelope inference, Holm family, an untouched-half confirmation, cost
reporting, and what a null result may claim.

## Why

Pre-registrations 4-10 found that on-policy rules that use only correctness (PSR, NSR, constant
baselines, a pass@4 objective, NSR's implementation, added noise) trace one curve of pass@16
against pass@1. Our theorem says every such rule only reweights questions (expected update
sum_x c(p_x) grad p_x) and changes the noise. So a method that moves the curve has to work
outside that class: limit the sharpening that spreads success across questions (if Pre-
registration 11 finds sharpening separable), or add information through memory of verified
successes (otherwise). Published methods of each kind are run as audit arms under the same
protocol: matched answers, matched accuracy, several seeds, intervals.

## Branch rule

- Pre-registration 11 verdict **separable** -> **Branch S** (limit sharpening).
- **Average agreement** or **below** -> **Branch M** (memory).
- **Inconclusive**, **insufficient overlap**, **incomplete evaluation** or **crossing** -> no
  Pre-registration 12 run; Pre-registration 11's evaluation is enlarged first.

## Arms

All arms are Dr. GRPO (advantage r - group mean) with one change:

| Arm | Change | Branch |
| --- | --- | --- |
| B | none (within-class reference) | S, M |
| E | fixed entropy bonus on every token of the batch, coefficient beta_E (pilot) | S, M |
| A | adaptive entropy control: the bonus coefficient follows beta <- max(0, beta + eta (H0 - H_t)), H_t the batch's mean token entropy, H0 its value at step 0; eta from the pilot | S |
| F | **entropy floor (candidate method, Branch S):** mean token entropy on 64 frozen base-model answers must stay at least rho times its starting value; after each Adam step, if the first-order prediction H + <h, u> (h the entropy gradient before the step, u Adam's actual parameter change) is below the floor, the parameters move along h by the missing amount over abs(h)^2, the measured bank entropy is checked after every step, and while it is below the floor by more than 1% the move is doubled (up to eight times); a step still below after that counts as a floor failure (reported; if more than 5% of steps fail, F's verdict carries the label "floor not enforced"); rho from the pilot | S |
| M | **log-mass memory (candidate method, Branch M):** every verified, stop-terminated, distinct correct training answer joins an archive (at most 4 per question) after the update; each step adds the gradient of lambda x the sum over 8 archived questions (uniform) of log sum over that question's archive of pi(y given question) (MAPO's marginal likelihood term), on the same scale as the reward loss; lambda from the pilot | M |
| U | unlikeliness reward (He et al. 2025), beta_rank = 0.25 as published: in groups with mixed rewards, a correct answer's reward becomes 1 - 0.25 (G - rank) / G, rank 0 the most probable of the 8 by sequence probability | M |

Branch S runs B, E, A, F; Branch M runs B, M, U, E. The candidate method is F (Branch S) or M
(Branch M); the others are audits of published ideas.

## Protocol

Qwen2.5-Math-1.5B base, as Pre-registrations 4-10: 3,920 training problems, AdamW 1e-6, 16 questions
by 8 answers, strictly on-policy, sequence-sum loss, gradient clipping at norm 1, answers up to
512 tokens, **150 steps**, bf16 checkpoints at steps 50, 100 and 150. The learning rate is fixed
for every arm (Pre-registration 10 found that the learning rate changes speed along the curve);
this is a stated limitation. **Fresh seeds 8, 9, 10**; the four arms of a seed run in parallel on
one 4-GPU instance and see the same questions at every step.

**Labels.** Before any Pre-registration 12 result, every reference answer of the evaluation and
validation sets is audited: Claude Haiku 5.5 solves each problem; where it disagrees with the
reference, Claude Opus 5.5 adjudicates between the two; references it rejects are corrected (or
the item excluded if no single correct answer exists), and the list is committed before the
primary runs (`label_audit.py`). Result (October 19): 3 of 768 references rejected (a sign error,
corrected; two multi-answer questions whose reference kept one value, excluded), so the
evaluation set has 511 problems and the validation set 255 (`data/label_corrections.json`). The
256-problem set of Pre-registrations 9 and 11 contains one of the excluded items; their
pre-registered analyses are unchanged and a sensitivity analysis without it is reported. The same list gives a sensitivity analysis for Pre-registrations
9 and 11, which share the 256-problem set.

**Pilot (strength choice, before the primary runs).** Seed 11, 100 steps, each tunable arm at two
strengths (beta_E 0.001 and 0.003; eta 0.01 and 0.03; rho 0.8 and 0.6; lambda 1 and 4) plus B;
pass@k with 32 samples on 256 separate validation problems (`data/val_frontier_256.json`, none
in training, MATH-500 or the evaluation set) at step 100. For each arm, the strength whose
step-100 validation pass@1 is within 3 points of B's is chosen, the higher validation pass@16
breaking a tie; if neither is within 3 points, the one closer to B's pass@1; a remaining tie goes to the weaker
strength; an intervention that never activates in the pilot (the floor never binds, the archive
stays empty) gets the stronger setting. A starts at beta 0. Choices are committed before the
primary runs start.

**Evaluation.** 512 subject-stratified held-out MATH test problems (`data/eval_frontier_512.json`,
which contains Pre-registration 11's 256), 32 samples at temperature 1, for the base model and
every arm's checkpoints at steps 50, 100, 150; B's step-150 checkpoint also at temperatures
1.15 and 1.3 (the temperature-calibrated baseline). pass@1 and unbiased pass@16 (primary),
pass@32 (secondary); answer lengths, truncation and parseable-box rates; texts saved.

## Decision rules

For an arm X and seed s, the comparison points are X's checkpoints (steps 50, 100, 150) whose
pass@1 lies inside the reference's pass@1 range; the difference at a point is X's pass@16 minus
the reference's pass@16 at the same pass@1 (linear interpolation, reference points ordered by
pass@1, ties by step; outside the resampled range the end value, with its frequency reported).
The arm's value is the mean over its comparison points, pooled as the mean over seeds;
uncertainty from 2,000 bootstrap draws of the 512 problems (one draw for all arms, seeds and
temperatures), comparison points fixed from the full data; a simultaneous 90% band over all
points is reported.

- **References.** (i) B's T = 1 trajectory (base, steps 50, 100, 150); (ii) **B's envelope**: at
  each pass@1, the higher of the trajectory and of B's step-150 temperature curve (T = 1, 1.15,
  1.3), each used only inside its own pass@1 range (points ordered by pass@1, ties by step or
  temperature); the envelope is recomputed in every bootstrap draw. Claims are against this
  restricted envelope (three temperatures).
- **Primary (candidate method F or M, against B's envelope):**
  - **moves the curve:** pooled difference at least +2 points, 95% interval above 0, and positive
    in all three seeds;
  - **average agreement:** 90% interval inside [-2, +2];
  - **worse:** at most -2 points and 95% interval below 0;
  - otherwise **inconclusive**; insufficient overlap (fewer than two comparison points in any
    seed) and incomplete evaluation as in Pre-registrations 9-10.
- **Equal cost:** reported separately: X's step-150 pass@1 minus B's, paired over problems,
  pooled over seeds; "the same accuracy at equal answers" only if its 90% interval lies inside
  [-1, +1].
- **Audit arms** (against B's trajectory, the same three-way rule). Holm's correction applies to
  the two-sided directional tests ("moves" or "worse") over the audit arms of the branch (two
  hypotheses in branch S, two in branch M); equivalence claims use the uncorrected 90% interval.
- **Untouched-half confirmation.** Half of the 512 problems also served Pre-registration 11's
  branch choice. The primary verdict is confirmed only if the candidate's pooled difference on
  the 256 untouched problems has the same sign and its 90% interval excludes 0; otherwise it is
  reported as "not confirmed on untouched problems".
- **What a null may claim.** "Average agreement" for an intervention requires its equivalence
  interval and evidence that it was active (floor corrections on at least 5% of steps; archive
  used on at least 90% of steps after step 20; entropy bonus with a nonzero coefficient);
  otherwise "inactive".

Reported without a rule: GPU time and processed tokens per arm (including bank and replay
passes), so "equal cost" (equal generated answers) is not read as equal compute; entropy traces, the floor's activity (share of steps with a correction,
size), the archive's growth, reasoning validity of correct answers for the candidate and B at
step 150 (LLM judge, as Pre-registration 8) if the primary verdict is "moves the curve".

## Contribution by outcome (stated in advance)

- **The candidate moves the curve:** a method outside the correctness-only class that shifts
  the accuracy-versus-diversity curve under a strict protocol, with the class boundary as its
  justification.
- **Only an audit arm moves it:** which mechanism escapes the shared curve; the candidate fails.
- **None moves it:** the shared curve also holds for these outside-class interventions at this
  scale (with intervals), a stronger diagnostic result.

## Cost

Pilot about $80-100; primary training about $160; evaluation 43 checkpoint-evaluations of
16,384 answers, about $180-250.
