The shared Hub-directory race is the clearest launch blocker. Static/source review only; no GPU run.

1. **Bugs and launch risks**

   - **Concurrent deletion/replacement:** [entry.py:95](/Users/mmayman/Desktop/adaptive-rlvr/experiments/sagemaker/entry.py:95) gives every evaluation of a repository the same directory; line 162 deletes it afterward. The plan repeats repositories across lanes—for example LUFFY at [plan:33](/Users/mmayman/Desktop/adaptive-rlvr/experiments/sagemaker/plans/p28_public.json:33) and line 101. One lane can replace tokenizer files or delete weights while another loads them. `_hub_lock` protects only `hub_tokenizer`. Use a separate directory per evaluation, or an immutable shared snapshot with protected lifetime.

   - **Stale-result reuse:** [eval_vllm.py:220](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/eval_vllm.py:220) fingerprints neither model weights/revision nor tokenizer identity or problem-file contents. Changing the model or `tokenizer_from` can silently reuse an old result. Pin Hub revisions and include those identities/hashes. Adding `force_answer` to the fingerprint correctly distinguishes forcing from ordinary evaluation.

   - **Unchecked tokenizer substitution:** [entry.py:97](/Users/mmayman/Desktop/adaptive-rlvr/experiments/sagemaker/entry.py:97) destroys the checkpoint tokenizer before checking compatibility. Remapped IDs silently change token meanings; IDs beyond the embedding table can crash. Validate token→ID mappings, added tokens, and embedding bounds before replacement. Base/LUFFY/ReLIFT/ExGRPO/zbeeb spot checks matched; differing padded `config.vocab_size` alone does not establish incompatibility.

   - **Disk peak:** [entry.py:96](/Users/mmayman/Desktop/adaptive-rlvr/experiments/sagemaker/entry.py:96) downloads full checkpoint files before loading/casting. The five first 7B downloads total approximately **122 GB**, before B1k checkpoints, the base channel, and the vLLM environment. [launch.sh:33](/Users/mmayman/Desktop/adaptive-rlvr/experiments/sagemaker/launch.sh:33) defaults to 60/120 GB. Check actual free space on the destination mount; cleanup afterward does not reduce this initial peak.

   - **Fallback without a model channel breaks:** [entry.py:122](/Users/mmayman/Desktop/adaptive-rlvr/experiments/sagemaker/entry.py:122) discards `snapshot_download`’s returned path. Line 155 subsequently treats the Hub ID as a local tokenizer directory, copies nothing, and evaluation fails. Retain the snapshot path.

   - **Requested checks that pass:** [eval_vllm.py:135](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/eval_vllm.py:135)’s `TokensPrompt(prompt_token_ids=...)` is valid in vLLM 0.11. The current context budgets also fit: reconstructed longest prompt **806 tokens**, plus 3,072 = **3,878**, below every planned public config’s minimum of 4,096. The guard at line 88 fails explicitly for longer inputs. Raw continuation is sound for these Qwen tokenizers; for broader support, preserve actual generated IDs rather than removing every pad ID at line 123, and use returned `prompt_token_ids` instead of retokenizing at line 136.

2. **Design flaws in the readings**

   - [Preregistration:10](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_28.md:10): forcing combines an answer cue, extra computation, greedy decoding, and removal of termination. It measures that intervention’s effect, not the fraction of RL improvement caused by learning to finish. Add an equal-budget continuation control before making a mechanism claim.

   - [Preregistration:35](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_28.md:35): define Holm families explicitly across sizes, budgets, references, and supervision groups. Specify shared problem-bootstrap indices, reference uncertainty, and single-point support/clamping rules. Reference A’s trajectory and temperature envelope need separate definitions. Unadjusted equivalence intervals should be labelled pointwise.

   - [Preregistration:41](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_28.md:41): freeze the exact baseline-count files; distinguish extrapolation across token budgets from extrapolation across pass@1. Also, forced points spend additional computation when compared with Reference A. Save forced texts and continuation IDs: [eval_vllm.py:149](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/eval_vllm.py:149) currently saves only original samples, preventing later audit.

3. **Wording fixes**

   - Lines 7–11: replace “property of the field’s methods,” “any such checkpoint,” and “tests that causally” with claims limited to the listed checkpoints, fixed protocol, and forced-answer intervention.
   - Line 25: random-reward training is not correctness-only RL. Label LUFFY’s on-policy baseline separately from its off-policy variants.
   - Line 56: specify **7 inserted suffix tokens plus up to 32 newly generated tokens**.
   - Line 60: the plan contains **39 specifications but 44 evaluation points**: B1k contributes four checkpoints and the low-temperature specification contributes three temperatures. Label runtime/cost as estimates.