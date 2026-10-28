Review before launch: experiments/training/PREREGISTRATION_27.md, experiments/sagemaker/plans/p27_long.json, and the
new evaluate_after/launch code in experiments/sagemaker/entry.py (torchrun mode, "then" evaluations split across the
run's GPUs). Also check train_dist.py supports every flag in the plan for Qwen2.5-Math-1.5B (tied embeddings) for 1,000
steps (data order wrap-around past one epoch, checkpoint saving, memory), and that law.py can produce reading 1's
prediction (pass@16 from base counts at a checkpoint's pass@1). Answer briefly, numbered: (1) bugs that would crash or
corrupt the run/evaluation (with lines); (2) design flaws in the readings (is the frozen law a fair reference at pass@1
beyond its fitted range? is pass@256 vs the base with 256 samples the right coverage test? thresholds?);
(3) whether the cost/time estimate is plausible; (4) wording fixes. No new experiments beyond fixing flaws.
