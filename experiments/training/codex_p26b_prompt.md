Re-review, before launch, the Pre-registration 26 changes in commit 9a12e3d (git show 9a12e3d). Your previous review is
experiments/training/codex_p26_review.md. Read: experiments/training/PREREGISTRATION_26.md (Amendment 1 and Step 2b),
train_grpo.py (OffDirection incl. top_pair/project_grads, teacher_backward, whitening_weights, the setup guards in
train_arm, and the step body where the teacher term is accumulated BEFORE the reinforcement gradient), and
direction_diag.py (rewritten: lazy safetensors loading, trajectory Gram, par/perp/other, bf16 retention, two-group
cross Gram). Also load_discovery (teacher solutions reuse it; the teacher archive will list every training problem
with c, n and a response only for base-rare problems with a verified 7B solution).

Answer briefly and concretely, numbered: (1) bugs that would crash or silently corrupt a run or diagnostic (with the line);
(2) whether each of your ten earlier points is resolved; (3) whether the teacher arm (TD, RTD) is a sound test of
"teacher supervision outside the shared RL direction", including the projection's placement (teacher grad alone in .grad,
then RL gradient accumulated), lambda = 1 with the constant normalizer, warm-up, and the selection rule; (4) any amendment
text that is wrong or overclaims. Do not propose new experiments beyond what is needed to fix a flaw.
