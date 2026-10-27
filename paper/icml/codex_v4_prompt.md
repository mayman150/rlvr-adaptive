Co-write draft v4 of paper/icml/paper.tex (read it; draft v3 has \pending{} placeholders). New registered results:
experiments/training/RESULTS_20.md (7B), RESULTS_21.md (1.5B at 1,024 tokens), RESULTS_23.md (minimum-help pilot);
method details in PREREGISTRATION_20.md, _21.md, _23.md. P22 (prefix guidance) and Check 21 are still incomplete:
keep \pending{} for them. Same style rules as before: natural human prose, precise terms, numbers exactly as in the
files with intervals, no bullet lists in the body.
Produce LaTeX snippets, each preceded by "%%% SNIPPET k: <where it goes>":
1. A new section "Replication at a larger budget and a larger model" (after "The answer budget as a confound"):
   P21's four contrasts and the joint statement; truncation 15-17%; the step-25 knot sensitivity; P20's three
   replicated contrasts, the NSR7-PSR7 contrast not interpreted under the clamping guard, the hard cohort (selected by
   screening, 253 problems, labels audited), pass@16 compression near 94-95 at 7B; rule 3's aggregate RMSE 0.23 with
   offsetting stratum errors; the evaluation-engine change (vLLM) and its equivalence check. About 450 words.
2. A new short section "Process credit from minimum hints" (after the frontier-model section): the method (credit on
   the policy's own failed prefixes by the smallest hint that completes them), its averaged-value control, the pilot
   result (-0.01, AVH -0.05, MH vs AVH +0.05; mechanism gates met: 520 credited prefixes, contrast in 61%), and that the
   registered selection drops it. About 250 words.
3. A revised abstract (<= 200 words) without \pending for P20, P21, P23, stating the paper's thesis: correctness-only
   RLVR interventions change the rate of progress along a shared pass@16-versus-pass@1 curve, not the curve, across
   the tested rules, budgets and two model sizes; the answer budget can create apparent frontier gains.
4. The introduction's contribution paragraph, revised accordingly.
5. Replacement rows for the appendix table for registrations 20, 21 and 23.
6. A revised Discussion section (about 300 words): what the invariance means, what it does not establish (other task
   families, longer training, non-correctness rewards), and why the causal update-direction test (forthcoming,
   \pending{P25}) is the natural next step.
