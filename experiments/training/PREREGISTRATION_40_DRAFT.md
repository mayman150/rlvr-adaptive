# Pre-registration 40 (DRAFT): pilot of reasoning source x delivery (teacher vs self; SFT before RL vs hints withdrawn during RL)

Drafted November 1, 2026, from the agreed plan (paper/debate_2026_11_02/PLAN.md), after Stage 0's results and Gate H
(PREREGISTRATION_39.md), before any run of this design. The placeholders [B] (answer budget) and [cost] are fixed from
Stage 0b's headroom screens and smoke tests before freezing; Codex reviews the frozen version.

## Question (pilot: direction and size, not confirmation)
On hard training problems, does reasoning from a teacher give a coverage gain that the model's own valid solutions do
not, and does delivering it as hints withdrawn during RL keep the gain after withdrawal better than SFT before RL?

## Data
- Training pool: MATH train (3,920). Hard schedule for every arm: 4 of each step's 16 problems come from the guided set
  G (shuffled passes), the rest from the usual order (train_grpo --hard-problems G --hard-per-step 4).
- Guided set G: problems with at most 4 of 64 base successes (verifier v2) whose matched teacher trace and self trace
  (p40_banks.py: the length-closest pair) are both audited "valid" by the blind audit (results_p26/p40pre_audit_*;
  final banks add Stage 0b's teacher solutions and are audited the same way). Teacher: Qwen2.5-Math-7B-Instruct.
  Self: the base's own verified, finished solutions (Stage 0 census for c = 1-4; the audited harvest for c = 0).
- Evaluation: hard_val (200; OlympiadBench + AIME) with 64 samples, and MATH-511 with 32 samples, at [B] tokens, T = 1;
  hard_test stays sealed.

## Arms (Qwen2.5-Math-1.5B; Dr. GRPO, Pre-registration 21's recipe except [B] answer tokens and 300 steps; checkpoints
25, 50, 100, 150, 200, 250, 300)
1. DR: Dr. GRPO with the hard schedule (seeds 8 and 9).
2. T-SFT: SFT on the teacher bank (one pass: every guided problem's trace once, AdamW 1e-5, batches of 8), then DR
   (seeds 8 and 9).
3. T-HINT: DR with teacher hints on G (MFC-style curriculum port: 4 levels, uniform level draw up to each problem's
   frontier, frontier lowered when the group mean reward >= 0.5, every hint withdrawn from step 200; seeds 8 and 9).
4. S-SFT: as T-SFT with the self bank (seed 8).
5. S-HINT: as T-HINT with the self bank (seed 8).
Each pair of arms shares seeds, so contrasts are paired. Exposure (guidance tokens presented) is logged per arm.

## Readings (descriptive; fixed now)
- R1 post-withdrawal coverage: at steps 250 and 300, each arm minus DR (same seed) on hard_val pass@1, pass@16 and
  pass@64, with 95% problem-bootstrap intervals (2,000 draws, seed 40).
- R2 matched accuracy: each arm's checkpoints against DR's same-seed trajectory, hard_val pass@64 and MATH-511 pass@16
  at matched pass@1 (analyze_p36's estimator: fixed checkpoint set, support rule); unsupported readings are reported as
  such.
- R3 persistence: R1's gap at steps 250-300 minus the gap at steps 150-200 (during hints), per arm.
- R4 source: T-SFT minus S-SFT and T-HINT minus S-HINT (seed 8), R1 and R2.
- R5 diagnostics: hint-curriculum state (unguided share, frontier histogram at step 200), guided-problem training
  reward, truncation, exposure.

## Advance rule (to a three-seed confirmatory stage, separately registered)
An arm advances if its post-withdrawal gap (R1 at step 300, or R2 where supported) on hard_val pass@64 is at least +2
points in every available seed. The teacher-versus-self contrast advances if a teacher arm exceeds its self
counterpart by at least +2 at step 300. Otherwise the pilot is reported as it stands.

## Cost
8 lanes x 300 steps on one 8 x A100 machine, plus evaluations: [cost] (planned about $800).
