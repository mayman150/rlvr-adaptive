# Pre-registration 9: results (OLMo-2-0425-1B-SFT, October 19, 2026)

All 10 training runs and 59 evaluations complete. Full output: `RESULTS_9_analysis_output.txt`;
without the one audit-rejected problem: `RESULTS_9_analysis_output_label_sensitivity.txt` (no
verdict changes). Verdicts checked independently against the rules (Codex, read-only).

| Rule | Result | Verdict |
| --- | --- | --- |
| 1. F0 / F1 above 1.2 (95%, bootstrap over questions) | 2.14 [1.55, 2.93]; 2.54 [1.73, 3.65] | pass, both seeds |
| 2a. NSR / PSR second moment above 1 | 19.45 [17.37, 22.05]; 22.22 [19.82, 25.29] | pass, both seeds |
| 2b. NSR slower to the midpoint | seed 1: 52 vs 52 (tie); seed 2: 88 vs 41 | fail (needs both seeds) |
| 3. NSR vs PSR, pass@16 at matched pass@1 | -0.30, 90% [-1.28, +0.77]; 6 points per seed | average matched-accuracy equivalence |
| 3. pass@4 vs Dr. GRPO | +1.32, 90% [+0.24, +2.26] | inconclusive |
| 4. Optimal rule minus Dr. GRPO (greedy, 3,382 problems; budgets 38,424 and 38,396) | +0.65 [-0.24, +1.54] | inconclusive |
| Formatting flag (>10 points) | largest gaps 8 (boxed) and 7 (truncation) points, NSR seed 1 | no flag |

Accepted wording (Codex): Across two OLMo-2-0425-1B-SFT seeds trained for 300 steps, NSR and PSR
satisfied preregistered average matched-accuracy pass@16 equivalence (difference -0.30 points,
90% interval [-1.28, +0.77]). This extends average equivalence to a second model family under
the tested protocol, while checkpointwise equivalence remains unresolved and uncertainty
covers problem sampling rather than training-run variability. Wrong-answer squared gradient
norms were larger in both seeds, and NSR's per-question second moment was 19-22 times PSR's,
but the registered speed prediction failed because seed 1 tied at 52 steps. Pass@4 training
showed a suggestive positive matched-accuracy difference (+1.32, 90% [+0.24, +2.26]), with an
inconclusive registered verdict. The transferred optimal collection heuristic also remained
inconclusive (+0.65 greedy accuracy points, 95% [-0.24, +1.54]) at valid answer budgets. NSR's
late formatting differed (seed 1: boxed 89% versus 97%, truncated 8% versus 1%) without
crossing the registered flag, and excluding the audit-rejected problem changed no verdict.

Codex's updated score with Pre-registrations 9 and 10: 6/10 (weak accept). Narrow the story:
reweighting in the theorem does not by itself imply a shared frontier.
