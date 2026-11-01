# Round 2 — Fable sign-off

**(1) One shared cost model.** I accept Codex's rates: $4/A100-hour; 4 A100-min/step (Qwen) and 8 (SmolLM3 — a placeholder until Stage 0 calibration) at 2,048 tokens. Evaluation: Stage 0 implies ~32K completions/A100-hour, so a 400-problem × 512-sample reading costs ~$26 (Qwen), ~$50 (Smol). My round-1 $5.5–7K was a trimmed design (500 steps, 256 samples, 200-problem readings, Smol fallback), not a different rate model. Full program, train + eval per stage:

- Stage 0 + separate training calibration: $400–700.
- Stage 1 pilot (six Qwen arms × 150 steps, Smol smokes, screening evals): ~$500.
- Stage 2 Qwen confirmation: training ~$3,200 (15 runs × 700 steps ≈ $187 each, references extended to 1,200), evaluation ~$1,300 (45 sealed late readings plus validation): ~$4,500.
- Stage 3 Smol: training ~$3,900 (9 runs × 700 steps ≈ $373 each, plus references), evaluation ~$1,400: ~$5,300.
- Stage 4 content/exposure controls and mechanism: $1,500–2,500.

Total ≈ $12–13.5K. Codex's envelope stands. The original $3–5K funds roughly through first-family confirmation only; say so to the user now.

**(2) Minimal credible version, ~$8.5K.** Cuts that do not weaken the main claim: arms to 500 steps with references to 700 (v6 decay completes by ~150; 300 unaided steps after the step-200 cut suffice); sealed readings at 256 samples, two late points, k ≤ 128 (512 × three points is over-specification); Stage 4 reduced to the answer-free-teacher and visible-prefix SFT controls, run only if a positive effect requires them; Smol arms cut to plain plus both sources under the validation-selected delivery. Not cuttable: two families, three seeds, the full 2×2 on Qwen, and reference support built before the arms — each is load-bearing. Stage 2 ≈ $3,300, Stage 3 ≈ $3,700, Stages 0/1/4 ≈ $1,500.

**(3) Remaining disagreement.** Forcing complete withdrawal by step 200 is a deviation from MFC's success-driven frontier and must be registered as such, with the frontier state at the cut reported — otherwise the arm tests neither MFC nor our schedule.
