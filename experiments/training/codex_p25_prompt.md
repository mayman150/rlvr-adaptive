Draft experiments/training/PREREGISTRATION_25.md (output the full file text only): a causal test of WHY the
matched-pass@1 curve does not move. Read first: RESULTS_18.md, RESULTS_20.md, RESULTS_21.md, RESULTS_23.md,
CHECK_17_TRANSFER.md, PREREGISTRATION_21.md (estimator and guards), train_grpo.py (Runner, backward_weighted,
mh_credit, prefix_rollouts), eval_vllm.py, docs/full_story.tex (diagnosis). Facts: every tested intervention lies on
Dr. GRPO's pass@16-vs-pass@1 curve (1.5B at 512 and 1,024 tokens; 7B at 2,048); rollout allocation gave rare problems
2.5x more successes without moving it; minimum-help credit and human-solution guidance also did not.

Hypothesis to test: from a fixed checkpoint, single (or few-step) policy updates built from DIFFERENT gradient sources
all move (pass@1, pass@16) along the local tangent of Dr. GRPO's curve (delta pass@16 = s * delta pass@1, s the
curve's slope at that point), with no transverse component; equivalently, rare-problem-sourced updates do not raise
rare problems' success more than ordinary updates of equal size raise it through shared parameters.
Design requirements: checkpoints B1k seeds 8-10 at steps 25 and 75 (or 50) at 1,024 tokens; donor update types at
matched update norm (one AdamW step from a fresh optimizer state is not comparable to training; specify a controlled
update: e.g. a normalized gradient step of fixed parameter-space norm, or k steps with the trained optimizer state):
(1) ordinary Dr. GRPO batch, (2) rare-problem-only batch (base c <= 4 of 32), (3) medium-only batch, (4) rare problems
with human-guided successes (P22's prefixes), (5) minimum-help credit on rare failures, (6) a random-direction control
of the same norm; several independent replicates per type (fresh rollouts); held-out evaluation on the 511 test
problems with eval_vllm.py (32 samples, T = 1, 1,024 tokens) plus per-stratum changes. Primary quantity: the
transverse displacement d_perp = delta pass@16 - s * delta pass@1, with s estimated from B1k's own trajectory around the
checkpoint; rules: tangent hypothesis supported if every donor type's mean d_perp has a 90% interval within [-1, +1]
point (per unit pass@1 movement scaled appropriately) and rare-sourced types do not exceed ordinary in rare-stratum
gain per unit norm; falsified if some type has d_perp >= +1 with 95% interval above 0. Include a cost estimate
(A100 hours; vLLM ~13 min per 1.5B evaluation), the update-construction details, what each outcome would mean for the
paper, and prior work this connects to (cross-prompt interference 2602.21189, GradAlign 2602.21492). Precise terms,
human prose, compact (about 1.5 pages).
