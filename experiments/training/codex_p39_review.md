Launch blockers remain. Against the supplied inventory, all 23 eval specifications resolve: steps, offsets/seeds and base handling, including SmolLM3’s explicit override. [Runner](/Users/mmayman/Desktop/adaptive-rlvr/experiments/sagemaker/entry.py:208)

- **Blocking—data:** OB-2655 (validation) and OB-2967 (test) require missing A/B inputs. AIME-1999--4 references an absent PNG diagram. Replace these, audit remaining questions for standalone solvability, and update frozen hashes. [Validation](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/data/hard_val.json:514), [test](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/data/hard_test.json:1368).

- **Blocking—verifier:** All 600 gold answers parse, but equivalent answers using `\frac12`, a leading minus before `\frac{1}{2}`, or `+44` can fail normalization. Extend rational normalization, then re-score archived censuses before refreezing rare cohorts. [Verifier](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/verifier.py:22)

- **Blocking—yield/headroom gates:** Yield × audited solution-validity does not establish problem-level usable yield. Audit by problem; require ≥44 distinct audited-valid problems for H. Gate V must use RE_BASE `counts[208:]`, rather than its pooled 408-problem metric. [Preregistration](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_39.md:25)

- **Blocking—retention:** Define the two-rate 95% interval precisely. Compare SELF8 with SELFstart to test decline during RL, and B1k8 with the fresh base. The current SELF gate can miss loss relative to its SFT start. [Definition](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_39.md:38)

- **Non-blocking—dedup/contamination:** Pin the builder, tokenization and overlap denominator. Two hard_test rows exactly match `train_all_subjects.json`: AIME-1987--4 and AIME-2012-I-14. Broaden exclusions to full MATH and cluster OB/AIME duplicates before splitting. Local dedup cannot establish pretraining cleanliness. [Report](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/data/hard_sets_report.json:2)

- **Non-blocking—capacity/outputs:** The plan requests 1,526,272 completions, with a 2.90-billion-new-token ceiling. Smoke-test n=1024 and shard problems: the worker retains generations and copied token IDs; `--gen-batch` is ignored. Presize RAM/disk and recalibrate the 36-A100-hour/$170 estimate. Save per-sample token lengths, finish flags and IDs. [Evaluator](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/eval_vllm.py:119)

- **Cheap additions/schema:** Add the schema’s missing S32 checkpoint. Report source-specific headroom and harvest discovery at 64/256/1024: the frozen census used 1,024 tokens, while harvest uses 2,048. Fixed 1,024 sampling deviates from stop-after-two; use capped rounds or explicitly charge every pilot attempt. Consider restricting Smol’s extra census to the 110 PEER targets. [Schema](/Users/mmayman/Desktop/adaptive-rlvr/paper/debate_2026_11_01/SCHEMA.md:17)

No files changed.