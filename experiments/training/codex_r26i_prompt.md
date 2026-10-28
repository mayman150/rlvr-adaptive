Check experiments/training/RESULTS_26_interim.md against experiments/training/results_p26/dirdiag_w.json and
PREREGISTRATION_26.md (Step 1 and Amendment 1), and direction_diag.py (weight_part). Answer briefly, numbered:
(1) any number in the file that does not match the JSON; (2) any reading that misstates the registered rule or
overclaims; (3) whether the bf16-rounding caveat is right in direction (attenuation of cosines and of the rank-1
share) and how to quantify it cheaply from the existing bf16 checkpoints (no new training), e.g. the fraction of
elements of Delta that are zero or one bf16 spacing, or an estimate of the rounding-noise energy; (4) wording fixes.
