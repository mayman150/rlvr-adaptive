You are a senior ML researcher doing a focused literature review (search the web; prioritize 2025-2026 arXiv,
NeurIPS 2025/2026, ICLR 2026, ICML 2026, COLM 2026). Read first, for our state: docs/full_story.tex (sections from
"Allocating rollouts by frontier value" to "Where we stand"), experiments/training/INVESTIGATION_27.md,
experiments/training/PREREGISTRATION_27.md, _28.md, _29.md, paper/LIT_2026_10_27.md (our previous scan; do not
repeat it, extend it), paper/RELATED_2026.md.

Our current findings (Qwen2.5-Math-1.5B/7B, MATH, Dr. GRPO, 150 steps): at matched pass@1, ~20 interventions
(advantage rules, entropy, replay, allocation, human hints, minimum-help credit, update-direction changes, whitening,
teacher distillation at a small dose) lie on Dr. GRPO's pass@16-vs-pass@1 curve; methods solve nearly the same
problems at the same rates (base difficulty order preserved) although weight updates differ; a frozen per-problem law
from base counts predicts the curve; about a third of the 1,024-token gain is learning to finish answers; problems the
base finishes wrongly stay unsolved. Candidate explanations: (1) no on-policy signal on unsolved problems; (2) RL learns
shared skills that lift all problems in base order; (3) training too short/gentle. Running: 1,000-step runs (reason 3),
21 public checkpoints on the curve, TD at 7B with a 72B teacher.

Deliver (markdown, compact, every claim with a link, mark what you verified in the paper vs abstract only):
1. For each explanation (1)-(3): papers that support, refute, or refine it, with their key numbers and setting.
2. Papers on: RL as sharpening/reweighting vs capability expansion (and where expansion is reported: horizon,
   rollouts, KL/resets, distillation dose, data); per-problem/difficulty structure of RL gains; length/truncation/
   format effects ("learning to finish", budget forcing); spurious/random rewards and format learning; evaluation
   pitfalls (pass@k at fixed steps, matched-accuracy comparisons).
3. Direct threats to our claims and to novelty (anything showing a matched-accuracy frontier move by an RL method, or
   already stating our "one curve"/per-problem law).
4. Verdict: should we change our thinking? Which explanation does the literature favour; which of our running
   experiments are likely redundant or mis-designed given prior results; 2-3 concrete changes to the plan (experiments
   or framing) with expected outcome. Be candid; do not flatter.
