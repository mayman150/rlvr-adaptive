1. **Bugs and launch safety**

   **Confirmed: teacher tokens corrupt the rollout length metric.** [train_dist.py:215](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/train_dist.py:215) adds teacher lengths, but line 261 divides their combined total by the number of student answers. Compute `mean_length` from `stats` alone; report rollout, teacher, and total tokens separately.

   **Confirmed: changing `--tp` can reuse stale evaluations.** [eval_vllm.py:235](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/eval_vllm.py:235) excludes `tp` from the resume fingerprint. I reproduced a TP=4 invocation skipping its worker after a TP=1 result existed. Include `tp` in the fingerprint and result metadata.

   **No obvious collective mismatch:** teacher items enter the shared plan; every rank executes the same `n_micro`, including zero-weight dummy passes, and the `world` factor correctly compensates for gradient averaging ([train_dist.py:204](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/train_dist.py:204), lines 204–237). The supplied GPU masks are disjoint, and both TP values equal four visible GPUs. The `torchrun` command and isolated evaluator subprocess wiring look consistent ([entry.py:155](/Users/mmayman/Desktop/adaptive-rlvr/experiments/sagemaker/entry.py:155), [eval_vllm.py:249](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/eval_vllm.py:249)). CUDA startup remains unverified.

2. **Design flaws**

   **Dose is sparse and uncontrolled.** [PREREGISTRATION_29.md:25](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_29.md:25): 150 × 16 visits covers only 2,400 of 3,904 eligible training problems per seed, each once. Four teacher samples, canonicalization, and the success gate reduce exposure further. Register post-canonicalization coverage, applied teacher counts/tokens, and counts split by student successes 0/1/2. A null concerns this dose and gate. Also, “one answer with advantage 1” is substantial when zero-success groups have **zero RL gradient**.

   **“≤2 successes” is a noisy online gate.** For a problem with true success probability 0.5, it activates about 14.5% of visits. It does not identify problems whose solutions the learner lacks. This design changes teacher size, learner size, archive coverage, budget, and gating relative to P26; it cannot isolate the effect of a stronger teacher.

   **Reference:** reusing B7 is reasonable if base weights, recipe, prompt, and evaluation protocol match. But B7 already has pass@16 around 94–95% on these 511 problems ([RESULTS_20.md:13](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/RESULTS_20.md:13)). Add the existing 253-problem hard cohort as a registered secondary evaluation with more headroom.

   **Decision rule is ambiguous.** [PREREGISTRATION_29.md:33](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_29.md:33): specify percentage-point units, checkpoint averaging followed by equal seed weighting, at least two eligible checkpoints per seed, complete required evaluations, and pooled **and each seed’s** equivalence intervals. “Average equivalence” and “pointwise” should not be conflated. State that bootstrap intervals condition on these two trained seeds.

   **Smoke test is insufficient as a correctness gate.** [p29a_teacher72.json:60](/Users/mmayman/Desktop/adaptive-rlvr/experiments/sagemaker/plans/p29a_teacher72.json:60) uses the old archive, 512 tokens, and threshold 8. It has no assertion that teacher updates occur. Add frozen-rollout gradient parity with teacher-only and uneven-load cases, plus a maximum-length case and finite-update checks. Before TD7, freeze the actual training plan with explicit `--teacher-max-success 2`, checkpoint saves, and the new archive checksum. P29a currently produces teacher samples; archive construction remains a separate step.

3. **Wording fixes**

   [PREREGISTRATION_29.md:7](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_29.md:7): replace “Every tested … lies on …” with “The interpretable registered comparisons were consistent with …”; some earlier contrasts failed interpretation guards.

   Lines 9–11: call TD a **motivated follow-up**, since the pilot interval includes zero and it failed selection. Frame transfer of reasoning patterns as a hypothesis.

   Lines 28–29: say “sequence-summed teacher log-likelihood, including the appended stop token, with coefficient 1”; this makes the dose precise. Specify **A100 80 GB**.

   [make_teacher.py:47](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/make_teacher.py:47): `mean_correct_of_8` is wrong for this four-sample archive; record the actual sample count.

   Validation was static and through isolated stubs; no GPU execution was performed.