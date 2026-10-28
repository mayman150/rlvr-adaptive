Review the design before implementation (candid, numbered, concrete): experiments/training/PREREGISTRATION_30.md. It
follows your strategy advice in paper/codex_strategy.md. Check: (1) whether the decisive test is well posed and
falsifiable (law forecast from the start's own counts; observed shift vs predicted shift; thresholds; the known P17
warm-start failure); (2) selections R/H/W definitions, the teacher pool from P29a, doses, cycling, subject/length
matching, TailSFT adaptation (state the paper's actual filtering criterion so I can implement it faithfully);
(3) competing predictors (pass@1, pass@32, held-out NLL on human solutions; should we add pass@64 or BOPTR?);
(4) confounds and missing controls; (5) cost/time; (6) wording. Read law.py, train_grpo.py (sft_phase,
load_discovery), PREREGISTRATION_14.md (warm start) and _17.md (law) as needed.
