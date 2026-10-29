# Pre-registration 29: teacher distillation at 7B with a 72B teacher (TD7)

Declared October 28, 2026, before any run of this design; reviewed by Codex before launch.

## Why

The interpretable registered comparisons were consistent with Dr. GRPO's matched-pass@1 curve at 1.5B and 7B.
Teacher distillation (TD, Pre-registration 26) adds information the policy may lack; at 1.5B with a 7B teacher it gave
+0.31 [-1.05, +1.35] (one seed; it failed selection). This is a motivated follow-up, not a confirmation: the
hypothesis, from reports that distillation can add reasoning patterns (Yue et al. 2504.13837), is that a much stronger
teacher at a larger student moves the curve. The design changes teacher size, student size, archive coverage, budget
and gating relative to Pre-registration 26, so it cannot isolate the effect of teacher strength.

## Step 1: teacher solutions (job P29a)

Teacher: Qwen/Qwen2.5-Math-72B-Instruct (Hub revision pinned in the plan), vLLM with tensor parallelism over 4 A100,
our prompt (the 1.5B tokenizer and chat template substituted after the vocabulary check of Pre-registration 28),
T = 1, 4 samples, 2,048 tokens, on all 3,920 training problems (two shards). Archive: every training problem with its
verified samples in sample order (`make_teacher.py --max-c 32`); the trainer keeps the first that survives
canonicalization (Pre-registration 14: retokenized, at most 2,047 tokens, verifies, final box). Reported: coverage
(problems with at least one verified solution), by the 1.5B base count. The same job runs a 4-step smoke test of the
multi-GPU teacher term at 1.5B (train_dist.py, 4 GPUs, the 1.5B teacher archive at its own 1,024-token canonicalization,
gate 2). Launch gate for step 2: the smoke run completes, logs at least one applied teacher solution, and has finite
gradient norms. Frozen-rollout gradient parity for teacher items is not run; the teacher items use the answers' code
path, whose parity was checked in Pre-registration 20.

## Step 2: training (job P29b)

- **TD7:** Dr. GRPO on Qwen2.5-Math-7B with Pre-registration 20's recipe (train_dist.py, FSDP2, 4 A100 per run, fp32
  weights with bf16 autocast, 16 problems x 8 answers, 2,048 tokens, AdamW 1e-6, 150 steps, constant normalizer
  16 x 8 x 2,048, the same data order per seed), plus, for each of the step's problems that has a teacher solution and
  whose group has at most 2 successes of 8, the gradient of -lambda times the sequence-summed log-likelihood of the
  teacher solution (including the appended stop token) / normalizer, lambda = 1 (one answer with advantage 1; note that
  a zero-success group otherwise contributes no RL gradient). Seeds 8 and 9 (in parallel on one 8 x A100 80 GB
  machine). The training plan is frozen before launch with the archive's SHA-256 checksum.
- **Dose (reported, and the scope of any null):** 150 steps x 16 problems visit 2,400 of the 3,904 training problems
  once per seed; the post-canonicalization archive coverage, the number of applied teacher solutions and their tokens,
  and their split by the group's success count (0, 1, 2) are reported. The gate is a noisy online signal (a problem
  with success probability 0.5 passes it on about 14.5% of visits); it does not identify problems whose solutions the
  student lacks.
- **Reference:** B7 seeds 8 and 9 (Pre-registration 20; vLLM evaluations; trajectory steps 0, 25, 50, 100, 150 and the
  step-150 temperature trajectory T = 1, 1.15, 1.3).

## Evaluation and decision (Pre-registration 21's estimator and guards; 511 test problems, 32 samples, T = 1, 2,048 tokens)

Checkpoints 50, 100, 150 of TD7 (evaluated on the training machine after training), on the 511 test problems
(primary) and on Pre-registration 20's 253-problem hard cohort (registered secondary; B7 reaches pass@16 of about
94-95% on the 511, leaving little headroom). Gains are in pass@16 percentage points at matched pass@1, averaged over
eligible checkpoints within each seed, then over seeds with equal weight; at least two eligible checkpoints per seed
and all required evaluations are needed, otherwise "insufficient overlap". Intervals condition on the two trained seeds.
- **Moves the curve:** pooled gain >= +2, 95% interval above 0, positive in both seeds (primary set).
- **Average equivalence:** the pooled 90% interval and each seed's 90% interval within +/-2.
- Otherwise inconclusive. More than 10% clamped bootstrap points in a seed: not interpreted.
- The hard cohort is read with the same rules and reported; it cannot by itself establish a "moves" verdict.
Reported: step-150 pass@1 minus B7's; teacher solutions per step; pass@16 by base-count stratum (Pre-registration 20's
7B base counts); truncation and length. A "moves" verdict triggers a confirmation on the fresh Omni-MATH set
(Pre-registration 24's design) before any claim.

## Cost (estimates)

P29a: one 8 x A100 machine, about 1.5 hours (two 145 GB downloads, two tensor-parallel shards, smoke test): about $50.
P29b: one machine, about 6.5 hours of training and 30 minutes of evaluation: about $220. Total about $270.

## Amendment 1 (October 28, before any teacher sample was produced)

The two tensor-parallel-4 vLLM engines of job worth-p29a-teacher-s0-20261028-180336 failed at initialization
(conflicting distributed ports); no samples were produced. The smoke test passed in that job (4 steps, 2-8 applied
teacher solutions per step, finite gradients). Teacher generation reruns as one tensor-parallel-8 engine over all
3,920 problems (job worth-p29a2-teacher-s0-20261028-183430); per-problem sampling seeds are unchanged (they depend on
the global problem index), so the design is unchanged.

Amendment 1 addendum: the tensor-parallel-8 rerun (worth-p29a2-teacher-s0-20261028-183430) failed at vLLM
initialization with "custom_all_reduce.cuh: invalid argument"; the first failure had the same cause (vLLM's custom
all-reduce needs CUDA IPC, unavailable in SageMaker containers). eval_vllm.py now disables it for tensor parallelism
(NCCL all-reduce, same numerics up to reduction order); third launch worth-p29a3-teacher-s0-20261028-184900. Lost time
about 9 + 10 minutes of one machine (about $10).
Second addendum: the third launch generated for about 15 minutes, then failed with "CUDA error: an illegal memory
access" in the tensor-parallel-8 engine (no samples written). Fourth launch: CUDA graphs disabled (enforce_eager) and
the 3,920 problems in four sequential parts sharing one download, so a failure loses at most one part; per-problem
sampling seeds unchanged.

Third addendum: the fourth launch produced degenerate text (fluent openings followed by random tokens; 71% truncated, pass@1 28.7% on the first 980 problems), consistent with corrupted tensor-parallel communication; it was stopped and its samples are discarded. Fifth launch: NCCL peer-to-peer disabled for tensor-parallel engines, and a 48-problem gate that aborts the lane if pass@1 is below 0.6.

Fourth addendum: the fifth launch (worth-p29a5-teacher-s0-20261028-213520) stopped at its gate: pass@1 0.34 on the
first 48 problems, again with degenerate text, so tensor-parallel generation of the 72B model is unreliable in this
container. Only unofficial quantized single-GPU 72B builds exist. The 72B teacher is abandoned; Pre-registration 30
(Amendment 1) uses Qwen2.5-Math-7B-Instruct, which runs on one GPU. Whether P29b (TD7) runs with that teacher is a
separate decision, not made here. Lost cost of the five launches: about $70.

## Decision (October 29, 2026): P29b is not run

The project lead decided not to run P29b (TD7) with the substitute 7B-Instruct teacher: a teacher of the trained
model's own size is a weaker test of the registered question than the 72B teacher, and Pre-registration 30 tests the
higher-priority question (whether the starting distribution moves the curve). No P29b result exists; the registration
is closed without data.

Correction (October 29): the degenerate 72B text was attributed above to corrupted tensor-parallel communication. The
same degeneration (fluent openings, then random multilingual tokens, no final box) appears in single-GPU generation
by Qwen2.5-Math-7B-Instruct at T = 1 (a third of its samples; Pre-registration 30, Amendment 2 addendum), whose own
generation config is greedy. Sampling at T = 1 is therefore the more likely cause of the fourth and fifth launches'
failures; the cause was not established. The third launch's illegal-memory-access error is a separate failure.
