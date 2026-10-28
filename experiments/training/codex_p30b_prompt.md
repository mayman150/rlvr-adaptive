Re-review (concise, numbered): the revised experiments/training/PREREGISTRATION_30.md (revision 1, which answers
your codex_p30_review.md) and the new experiments/training/sft_start.py (starting-model SFT with solution rotation,
token milestones, TailSFT adaptation). (1) Which of your 13 points are resolved, partially resolved or open; (2) bugs in
sft_start.py that would crash, silently change the intended SFT (e.g. TailSFT margins/drop rule, loss normalisation,
milestone accounting, rotation), or exhaust GPU memory on a 1.5B model on one A100 80GB; (3) remaining design flaws
that must be fixed before Stage 1. Do not re-raise resolved points.
