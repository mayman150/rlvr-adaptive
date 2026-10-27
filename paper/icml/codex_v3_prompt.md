Co-write draft v3 of our ICML paper (paper/icml/paper.tex, draft v2 of Oct 23). Write in natural, human academic
prose (no bullet lists in the body, no hype), with precise research terminology and formal definitions where a term is
first used (matched pass@1, trajectory, envelope, truncation rate). Every number must come from these files, quoted
exactly with its interval: experiments/training/RESULTS_18.md, RESULTS_18_19_interim.md, CHECK_16_TOKEN_BUDGET.md
(result section), CHECK_21_TRUNCATION.md, PREREGISTRATION_18.md, _19.md, _21.md, _22.md, _23.md. Do not report
anything from Pre-registrations 20-23 as a result; they are pending (use \pending{...} placeholders).

Produce LaTeX (compatible with paper.tex's macros) for:
1. A new section "Rollout allocation at a fixed budget" (Pre-registration 18): design (AC law value, AV, HORA), the
   registered verdicts (rules 1-3 not met; AC +0.14 [-0.36, +0.58]; AC vs AV +0.51; AC vs HR +0.00), the mechanism
   statistics (pilot-zero problems receive 18.9 rollouts; 50-54% then contain a success vs 18-21% under AV), the forecast
   results (HR and AV confirmed, AC not: observed -0.05 against +4.12/+2.46), and what this implies (more binary signal
   on rare problems does not move the curve; the per-problem frontier model fails on the intervention it was designed
   for). About 450 words.
2. A new section "The answer budget as a confound" (Checks 16 and 21, Pre-registration 19): truncation rates at 512
   tokens (Dr. GRPO 48%, human-solution arms 6-7%); the warm start's step-0 gain +4.77 at 512 vs +2.15 at 1,024
   (registered reading: largely a budget effect); the human-solution arm above the curve at 512 and below it at 1,024
   (seed 10; others \pending); P19 rule 1 not interpretable; the lesson for matched-accuracy comparisons. About 400
   words.
3. Replacement text for the warm-start paragraph(s) in section "Replay, larger groups, and a supervised warm start"
   that no longer claims the warm start moves the frontier without the budget qualification.
4. A revised abstract (<= 200 words) and the contribution paragraph of the introduction, consistent with 1-3 and with
   \pending{} for the 1,024-token replication (P21), the 7B replication (P20) and the minimum-help method (P23).
5. A one-paragraph revision of the frontier-model section's conclusion in light of AC's failed forecast.
Output only the LaTeX snippets, each preceded by a line "%%% SNIPPET k: <where it goes>".
