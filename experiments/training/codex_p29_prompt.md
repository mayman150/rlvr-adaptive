Review before launch, quickly and concretely: experiments/training/PREREGISTRATION_29.md; the teacher term added to
experiments/training/train_dist.py (teacher_bank, the t_items block in train_arm: teacher items (j, -1) added to adv,
lengths, deal, own; FSDP collectives must match across ranks); the --tp option in experiments/training/eval_vllm.py
(vLLM tensor parallelism launched from a `python -I` worker subprocess with CUDA_VISIBLE_DEVICES set to 4 GPUs);
the multi-GPU lanes and "torchrun" step kind in experiments/sagemaker/entry.py (mixed_lane/run_lane/run_step); and
experiments/sagemaker/plans/p29a_teacher72.json. Answer numbered: (1) bugs that would crash, hang (e.g. unequal
collectives) or silently corrupt the run (with lines); (2) design flaws in the pre-registration (dose, criterion
"group <= 2 successes", reference, decision rule, smoke test adequacy); (3) wording fixes.
