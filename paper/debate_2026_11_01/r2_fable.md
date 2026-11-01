# R2 (Fable), November 1, 2026

## 1. Critiques

**Codex.** Facts check out (CWS−UNI +0.03 [−0.37, +0.44], support failed; TEACHF +2.10; TEACHF−SELF +1.53; the 65.6% arithmetic). Strongest objection: the constraint's core quantity is unmeasurable. Rare prompts have p≈1% (measured mean 0.93%); detecting declines needs hundreds of fresh samples per prompt per checkpoint, and importance-weighted estimates from stored probes lose effective sample size within a few updates on 1,024-token sequences. Repo evidence also cuts against suppression on rare problems: Check 17 margins *rise* under Dr. GRPO (+0.023–0.025). Update projection in a distributed Adam stack is heavy engineering for 11 weeks. MCPO/UCPO (2604.16972, 2605.00365) unverified by me.

**Claude r2.** The exposure arithmetic (~5 rollouts/problem) is a discovery argument, and P18 refutes discovery as the in-RL binding constraint: AC raised success-bearing pilot-zero groups to 50–54% vs 18–21% and gained +0.14 [−0.36, +0.58] against a +4.12 forecast. The hybrid's conditional constraint hangs on a one-week resample test — right test, too weak to gate a method; keep it as mechanism evidence.

**Self-correction.** My r1 inherited the false "untested" premises: G32, MaxRL/BBG, and allocation were tested. Still untested: BroRL-scale groups, T>1, and harvesting at 512–2,048 samples on c=0.

## 2. HARVEST re-evaluated; which method

P18 changes HARVEST's story, not its verdict. It kills the starvation mechanism — success signal delivered as in-RL *gradient* (≤32 rollouts) does nothing — which sharpens the question to the **channel**: S32 (SFT on base successes, 2.46M tokens) moved exactly the stratum the gradient channel could not (c=0-of-32 test pass@16 ≈24→34; +2.38 vs the 450-step reference, exploratory; step-0 excess +2.15 [+1.10, +3.20] at 1,024). Honest prior update, r1 ~50% → ~35%: S32's 512→1,024 drop was "largely a budget effect" and 2,048 is unmeasured; P36's 25K self-SFT faded (98× dose confound); Check 21's human solutions on never-solved problems sat 2.4–3.9 *below* B's curve at 1,024 — out-of-support content is not sufficient; style/compatibility matters (harvested solutions are base-styled, a point for HARVEST).

53% fact: verified (full_story.tex:658). Lucky guesses are real at p≈1% on short answers: mitigate by blind reasoning audit (~50), preferring problems with ≥2 independent successes, flagging guessable formats. CORR shows the carrier is reasoning, so garbage-reasoning-lucky-answer is the audited failure mode, not label leakage.

Novelty: STaR 2203.14465, RFT 2308.01825, ReST/Expert Iteration, DART-Math 2407.13690 (verified: difficulty-allocated synthesis trials, open 7B synthesizer, SFT-only — no RL stage, no pass@k/coverage reading) and BroRL jointly own "sample hard problems, fine-tune on successes." The defensible claim is channel-and-depth: *the same rare-success information moves the post-RL coverage frontier when delivered as pre-RL SFT from depth-512+ sampling, where the gradient channel (P18) and the same budget spent as longer RL do not.* **Method: HARVEST.** Constraints: only if Stage 0 demonstrates suppression.

## 3. Equal-compute accounting

110 c=0 train problems × 1,024 samples ≈ 112,640 rollouts; plus 150-step RL (19,200) ≈ 132K ≈ the existing 1,000-step reference (128,000 rollouts; +1.96 unadjusted at 256 samples, failed criterion) — the equal-rollout RL arm is effectively already run. Count harvest and SFT in the arm's budget; primary at equal GPU-hours (inference rollouts are ~3–5× cheaper per token; report both), sensitivity at equal rollout count. A BroRL-style large-G arm only if Stage 0 surprises.

## 4. Agreed schema

Thesis: RLVR coverage is bought by what enters the policy's training distribution, not by the RL algorithm: deep-sampled, verified rare successes distilled pre-RL durably raise hard-set pass@64/128 at matched pass@1 and equal total compute; the same information as gradient, and the same compute as longer RL, do not. Arms: HARVEST (3 seeds confirmatory); existing B1k/SELF/TEACHF/1,000-step controls; corrupted-label HARVEST (1 seed); token-matched TEACH-M. Setting: Qwen2.5-Math-1.5B, 2,048 tokens, OlympiadBench-style ≥400 + MATH-511, pass@1/64/128 at 256 samples, frontier as diagnostic; SmolLM3-3B replication. Stage 0 (wk 1, ≤$150, eval-only+generation): harvest + audit (gate: audited-valid solutions on ≥40% of c=0); Codex's fresh-sample resample of SELF/B1k rare problems (suppression vs catch-up); S32 checkpoints re-evaluated at 2,048; 256-sample re-eval of step-150 checkpoints. Stage 1 (wks 2–3, ~$450): HARVEST ×2 seeds; gates: −SELF ≥ +1 all seeds, hard-set pass@64. Stage 2 (wks 4–6, ~$900): 3 seeds, preregistered, controls. Stage 3 (wks 6–8, ~$450): SmolLM3, 300–500-step persistence. Kill: fades like SELF, vanishes at 2,048, or audit gate fails. Fallback: the mechanism paper (channel result: gradient null vs SFT positive vs teacher positive), already largely evidenced.

Disagreement (one line): Codex gates at ≥3 pp — too high against ±1–1.5 pp intervals; I gate at ≥1 pp vs SELF, ≥2 pp vs B1k.
