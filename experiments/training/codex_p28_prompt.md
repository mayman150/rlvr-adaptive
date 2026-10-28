Review before launch (be quick and concrete): experiments/training/PREREGISTRATION_28.md, the new --force-answer code
in experiments/training/eval_vllm.py (evaluate: raw ids, TokensPrompt continuation, max_model_len, fingerprint), and the
hub_model / tokenizer_from support in experiments/sagemaker/entry.py (hub_download, fetch_hub_model, hub_tokenizer,
mixed_lane eval branch), plus experiments/sagemaker/plans/p28_public.json. Answer numbered: (1) bugs that would crash
or silently corrupt results (with lines) - e.g. tokenizer replacement when a checkpoint's vocab differs, models whose
max_position_embeddings is below prompt+3,072, vLLM TokensPrompt usage in 0.11, thread safety of downloads, disk;
(2) design flaws in the readings; (3) wording fixes.
