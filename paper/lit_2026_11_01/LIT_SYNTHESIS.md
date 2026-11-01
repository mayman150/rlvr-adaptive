# Literature synthesis, November 1, 2026

Sources:
- codex_verify.md: verification of the nine papers cited by the external review.
- fable_lit.md: a broad search, every paper checked against its abstract.
- the coordinator's own abstract fetches, covering 12 key papers.

## 1. The external review's citations
All nine exist and nearly all numbers are correct. The corrections:
- 2610.24663 studies Qwen3-4B-Base, with a similar result on 8B. The 61.45 figure is pass@256 for a response mixture
  whose pass@1 is matched (19.00 vs 18.98), not a pass@1 value.
- MFC's 128-vs-2,000 result is a nine-benchmark average.
- DATPO evaluates at k = 8 (MATH500) and k = 64 (AIME/AMC) with a 2,048-token cap.
- Curriculum RL (2606.22317) is a preprint; its teacher identity and training compute are not reported.

## 2. What the field claims, and how it measures it
Methods that claim to expand the boundary, all compared at fixed step, budget, wall-clock or compute; none at
matched accuracy:
- teacher-guided curricula on zero-success problems: MFC 2609.13997 (pass@256 70.0 vs 60.0), Curriculum RL
  2606.22317 (+9.8 pass@256);
- prefix and hint scaffolds: QuestA 2507.13266, BREAD 2506.17211, PrefixRL 2601.18795, AdaBack 2506.18110,
  MOER 2610.30873;
- tree and branched rollouts: DATPO 2609.08650;
- rollout scaling and prolonged training: BroRL 2510.01180, ProRL 2505.24864;
- restricting updates to zero-success problems: BBG 2606.15455;
- pass@k objectives: 2508.10751, 2505.15201;
- explorer distillation: ExpDis 2610.10536, the only one matched on wall-clock.

Persistence of injected behavior:
- Quagmires 2510.01624 (multi-family): high SFT scores mispredict post-RL results, and pass@large-k before RL
  predicts them.
- 2605.29190 (OLMo3-7B): RL suppresses reasoning primitives introduced by SFT; a perplexity-novelty bonus restores
  them (+7 points pass@32).
- 2609.28145 (on-policy distillation before RL) and 2609.04108 (sequential beats joint) claim the warm-start ordering
  persists, but at fixed endpoints only.

Theory and evaluation that bear on our instrument:
- No Free Lunch 2610.39295: a theorem that every pass@1 gain has a pass@k price under shared-strategy tilting. It is
  the theoretical counterpart of our one-curve result, and we must engage with it.
- 2609.09245: fixed-n counts identify pass@k only for k <= n, so any large-k claim needs n >= k.
- 2609.22547: statistical inference for pass@k crossings.
- CoT-Pass@k 2506.14245: reasoning-valid coverage judged by an LLM; RL keeps an advantage at k up to 1,024 at 32B.
- 2610.24663: at matched pass@1, a hybrid that swaps about 4% of tokens raises pass@256 from 61.45 to 67.5. This is
  a decoding-space counterexample to a fixed frontier.
- Elicitation: Spurious Rewards 2506.10947 (Qwen-Math only); Reasoning with Sampling 2510.14901.

## 3. Gaps our matched-accuracy, pre-registered instrument can fill
1. No method that claims boundary expansion has been evaluated at matched accuracy, so none separates a frontier
   shift from acceleration.
2. No persistence result at matched accuracy. Our decay of teacher starts to about +0.4 has no competitor.
3. No pre-registered training study.
4. The per-problem locus of teacher gains (expansion versus reallocation) is unmeasured at matched accuracy.
5. Strategy-level diversity has not been linked to position on the frontier.

## 4. Implications for the paper
- Retitle and rescope. Something like "What moves the RLVR coverage frontier?": standard on-policy optimization
  changes preserve it in our tested settings; information and exploration structure may not.
- Engage closely with MFC, Curriculum RL, DATPO, No Free Lunch, Quagmires, 2605.29190, 2609.09245 and CoT-Pass@k.
- Fix: say "zero-observed-success", not "unreachable"; the per-problem law's coupling parameter is fitted on RL runs;
  the corrupted-answer control leaves intermediate answers in place; add a seed-level bootstrap; use n >= k for large
  k; add a second family.
