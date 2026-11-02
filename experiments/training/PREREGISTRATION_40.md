# Pre-registration 40: pilot of reasoning source x delivery (teacher vs self; SFT before RL vs hints withdrawn during RL)

Declared November 2, 2026, from the agreed plan (paper/debate_2026_11_02/PLAN.md), after Stage 0 and Stage 0b
(PREREGISTRATION_39.md, Amendment 1) and before any run of this design; reviewed by Codex (code: codex_p40_code_review.md
and two re-checks; hints: two disclosure audits).

Fixed from Stage 0b: the answer budget is B = 3,072 tokens (the registered rule: base hard_val pass@128 71.2% with 9.9%
truncation at 3,072, against 18.7% truncation at 2,048). Hardware: one Nibi node, 8 H100 80GB GPUs (all arms on the same
hardware; experiments/tamia/p40_*.sh). Deviations from Pre-registration 21's recipe, all fixed before any run: 3,072
answer tokens; no calibration hold-out (--calib-prompts 0 --w 1; the baseline arm does not use w), so all 3,920 training
problems are in the pool; the hard schedule; 300 steps.

## Question (pilot: direction and size, not confirmation)
On hard training problems, does reasoning from a teacher give a coverage gain that the model's own valid solutions do
not, and does delivering it as hints withdrawn during RL keep the gain after withdrawal better than SFT before RL?

## Data
- Training pool: MATH train (3,920). Hard schedule for every arm: 4 of each step's 16 problems come from the guided set
  G (shuffled passes), the rest from the usual order (train_grpo --hard-problems G --hard-per-step 4).
- Guided set G (data/p40_guided.json, 145 problems: 140 with 1-4 and 5 with 0 of 64 base successes; hashes in
  data/p40_final_report.json): problems whose length-closest teacher/self pair (p40_banks.py) is audited "valid" for both
  traces by the blind audit (results_p26/p40pre_audit_labels.json), whose traces both give a non-empty answer-safe hint
  prefix (hint_utils.safe_hint_lines: cut before the first box and before any line revealing the answer by value), and
  whose top-level hints pass the narrow disclosure audit (results_p26/p40_leak_audit2_labels.json; 5 problems excluded).
  Teacher: Qwen2.5-Math-7B-Instruct; self: the base's own verified, finished solutions (Stage 0 census for c = 1-4, the
  audited harvest for c = 0). Banks are token-matched: 71,145 (teacher) and 71,252 (self) tokens. Stage 0b's new
  teacher solutions are not used in this pilot (they would need another validity audit).
- Evaluation (vLLM, separate job after training): hard_val (200; OlympiadBench + AIME) with 64 samples, and MATH-511 with 32
  samples, at 3,072 tokens, T = 1, at every saved checkpoint, plus the base model and both SFT starts (step 0);
  hard_test stays sealed.

## Arms (Qwen2.5-Math-1.5B; Dr. GRPO, Pre-registration 21's recipe with the deviations above; checkpoints
25, 50, 100, 150, 200, 250, 300)
1. DR: Dr. GRPO with the hard schedule (seeds 8 and 9).
2. TREE: DR with tree rollouts at an equal rollout count (DATPO-inspired, not a reproduction: 4 roots per problem; if
   any root fails, the first failed root is forked at its highest-entropy sentence start between 10% and 80% of its
   length and before its box, and 4 branches continue from that prefix; otherwise 4 ordinary rollouts; Dr. GRPO
   group-relative advantages over all 8; train_grpo --tree-branches 4; seed 8). The structured-exploration arm the
   external review asked for.
3. T-HINT: DR with teacher hints on G (MFC-style curriculum port: 4 levels, uniform level draw up to each problem's
   frontier, frontier lowered when the group mean reward >= 0.5, every hint withdrawn from step 200; seeds 8 and 9).
   Registered deviations from MFC: the top level shows at most 75% of a trace's answer-safe lines, not the whole trace;
   withdrawal is forced at step 200 whatever the frontier (its state at withdrawal is saved); a level is lowered further
   only if the hinted prompt would not fit the 4,096-token context with a 3,072-token answer.
4. T-SFT: SFT on the teacher bank (71,000 presented tokens, just under one pass over the bank; AdamW 1e-5, batches of 8),
   then DR (seed 8).
5. S-HINT: as T-HINT with the self bank (seed 8).
6. S-SFT: as T-SFT with the self bank (seed 8).
Each pair of arms shares seeds, so contrasts are paired. Exposure (guidance tokens presented) is logged per arm.

## Readings (descriptive; fixed now)
- R1 post-withdrawal coverage: at steps 250 and 300, each arm minus DR (same seed) on hard_val pass@1, pass@16 and
  pass@64, with 95% problem-bootstrap intervals (2,000 draws, seed 40).
- R2 matched accuracy: each arm's checkpoints against DR's same-seed trajectory, hard_val pass@64 and MATH-511 pass@16
  at matched pass@1 (analyze_p36's estimator: fixed checkpoint set, support rule); unsupported readings are reported as
  such.
- R3 persistence: R1's gap at steps 250-300 minus the gap at steps 150-200 (during hints), per arm.
- R4 source: T-SFT minus S-SFT and T-HINT minus S-HINT (seed 8), R1 and R2.
- R4b exploration: TREE minus DR (seed 8), R1 and R2; branch statistics (problems branched, branch success).
- R5 diagnostics: hint-curriculum state (unguided share, frontier histogram at step 200), guided-problem training
  reward, truncation, exposure.

## Advance rule (to a three-seed confirmatory stage, separately registered)
An arm advances if its post-withdrawal gap (R1 at step 300, or R2 where supported) on hard_val pass@64 is at least +2
points in every available seed. The teacher-versus-self contrast advances if a teacher arm exceeds its self
counterpart by at least +2 at step 300. Otherwise the pilot is reported as it stands.

## Cost
One Nibi node (8 H100s) for training (time limit set from the smoke test's measured step time) and one for
evaluation; Alliance allocation, no AWS cost.
