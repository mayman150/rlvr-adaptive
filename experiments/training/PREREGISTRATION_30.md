# Pre-registration 30: change the starting model, forecast the post-RL curve

Declared October 28, 2026 (revision 1, after Codex's design review in codex_p30_review.md; Stage 0 launched with
worth-p31-flat-s0-20261028-182104 before this revision, unaffected by it); Stages 1-2 run only after Codex's review of
this revision and of sft_start.py.

## Why

Interpretable comparisons of near-on-policy RLVR recipes on Qwen2.5-Math-1.5B/7B approximately follow Dr. GRPO's
pass@16-vs-pass@1 curve, and a frozen per-problem law computed from the starting model's counts predicts that curve
conditional on pass@1 (0.23-0.82 points RMSE; first public 7B checkpoints within about 1 point). It does not predict the
RL endpoint or compute, and no such claim is made. Prior work shows that the start matters (OctoThinker 2506.20512;
2512.07783; TailSFT 2608.25756) and that pre-RL pass@large-k and held-out loss predict RL outcomes (Quagmires
2510.01624; BOPTR 2609.01274). Pre-registration 17 already established that unrestricted transfer of the frozen law to
a warm start fails (S32: RMSE 3.56; its base-zero stratum 11.9 points below the forecast; refitted tau -0.65 to
-0.70). This study asks a scoped question: for verifier-correct teacher-SFT starts, does the registered forecast agree
with the observed post-RL curve shift, and does it rank starts better than simpler features?

## Stage 0: definitions (running)

The 1.5B base, 64 samples per training problem (3,920), T = 1, 1,024 tokens. Per problem: success count c (of 64),
unfinished count, and final boxed answers grouped by the verifier's normalization (`verifier.normalize`); "unfinished" means no final box.
- **H (low observed success):** c <= 4 of 64 (c = 0 reported separately).
- **W (concentrated wrong answers):** c <= 4, at most 31 unfinished samples, and the most frequent normalized wrong
  answer appears in at least 16 of 64 samples (ties: lexicographically smallest normalized answer).
- **R (random):** uniform without replacement from all training problems with at least one surviving teacher
  solution (seed 3030).
- **Matching:** every selection uses the same number of unique problems, N = |W| (problems with a surviving teacher
  solution); R and H draw N problems stratified by MATH subject to W's subject mix. H \ W is matched to W on subject
  and on the success-count bins {0}, {1-2}, {3-4} (of 64), so W can be compared with rarity alone. Quotas: a
  (subject, bin) cell that cannot be filled is filled from the same bin in any subject, then from the nearest bin;
  the shortfall is reported. If fewer than 100 W problems have a surviving solution, W and H \ W are dropped and H is
  the only targeted arm (reported). Arms remain composite interventions (difficulty,
  diversity, repetition, length); response-length and subject distributions are reported.

## Teacher pool

Qwen2.5-Math-72B-Instruct solutions from Pre-registration 29a (4 samples per training problem, 2,048 tokens),
verifier-correct (reasoning not audited; a random 50 are audited by Codex and reported), retokenized for the 1.5B and
kept if at most 1,023 tokens; all distinct surviving solutions of a problem are kept. Survival by selection is reported.
The RL training universe is unchanged (all 3,904 training problems).

## Stage 1: starts and frozen forecasts (no RL)

`sft_start.py` (fixed SFT seed 3030): problems visited uniformly in shuffled passes, rotating through each problem's
solutions; mean token NLL, AdamW 1e-5, batches of 8, clip 1; starting models saved at 0.5M and 2M presented response
tokens along one trajectory (retained tokens, unique solutions seen and repetitions reported; checkpoints hashed).
Starts: R, H, W, H \ W at 0.5M and 2M; TailSFT on R at 2M (Algorithm 1: margin of current minus initial mean NLL,
the most-improved fraction dropped, rising linearly 0 -> 0.5; loss over retained tokens; adaptation: selection batch 8
instead of 16).
Each start, and the base itself (so that changes in pass@64 have a 64-sample base reference), is evaluated on the 255
validation and 511 test problems with 64 samples (T = 1, 1,024 tokens); the law uses the first 32 samples (its
registered input depth); pass@64 uses all 64.
**Frozen forecasts, written to a file before any Stage-2 run:** for each start X and the base B, the law's complete
conditional curves f_X(u) and f_B(u) (pass@16 at pass@1 u; law.py, tau = -0.20, strict reachability), absolute and
differenced, on validation and test; also uniform and affine comparators and a zero-shift predictor. Features for rank
comparison only: the change in pass@1, pass@32, pass@64, mean NLL on the validation problems' human solutions (one
frozen corpus: our prompt plus the human solution text truncated to 1,023 tokens, response tokens only, EOS included,
token-averaged over the corpus; the appended token is the chat stop token <|im_end|>, as in sft_start.py), and a frozen power-map (BOPTR-style) forecast from
each start's 64-sample pass@N curve (calibrated on Pre-registration 21's Dr. GRPO checkpoints; unscorable cases
reported; not full BOPTR).

**Advancing to Stage 2 (validation forecasts only):** always R-2M (the dose-matched random-SFT control) and, if run in
Stage 1, TailSFT-2M; plus the single start with the largest absolute validation-forecast shift at the 2M dose among
H, W, H \ W (ties: W, then H), provided its forecast shift is at least 2 points on the validation grid; otherwise it
advances as a weak test. **Validation grid:** pass@1 values 0.40, 0.42, ..., 0.56, kept if inside B1k8's validation
trajectory range (Pre-registration 23's B1k8 validation evaluation) and inside both laws' reachable ranges with an
explicit margin (at least 0.005 above each law's initial pass@1 and below its final recorded value); equal weights;
an empty grid means "no common support" (the start cannot advance as the targeted start).

## Stage 2: RL from the saved starts

Dr. GRPO, Pre-registration 21's recipe (1.5B, 1,024 tokens, 16 x 8, AdamW 1e-6 with a fresh optimizer, 150 steps),
seeds 8, 9, 10, from each saved start (the same start for all seeds); checkpoints 5, 10, 25, 50, 100, 150 on the 511
test problems (32 samples, T = 1), step 150 also at T = 1.15 and 1.3. Reference: B1k seeds 8-10 (existing).

## Readings fixed in advance

Matched quantities: for each checkpoint of start X with observed pass@1 u, the observed gap is pass@16 minus B1k's
same-seed T = 1 trajectory at u, and the forecast gap is f_X(u) - f_B(u); both use Pre-registration 21's eligibility
(u inside B1k's support), checkpoint averaging within seed and equal seed weights. The residual is observed minus
forecast gap; its interval comes from a joint problem bootstrap (2,000 draws) that recomputes the prior estimation,
the law's initial state and the reference interpolation with tau frozen. At least two eligible checkpoints per seed,
otherwise "insufficient overlap". Intervals condition on the trained seeds.
- **Registered mean forecast agrees:** the residual's 90% interval lies within +/-2.
- **Forecast fails (bias detected):** |residual| >= 2 and the two-sided test of residual = 0 has Holm-adjusted
  p <= 0.05 over the advanced starts (an operational rule: it detects bias of at least the tolerance's size in the
  estimate, not error confidently beyond +/-2; the one-sided test of |residual| > 2 is reported alongside).
- Bootstrap draws in which a checkpoint leaves the reference's or a law's support drop that checkpoint for that draw;
  if more than 10% of draws are affected for a seed, that seed's verdict is "not interpreted".
- Otherwise inconclusive. Per-seed residuals and per-checkpoint bands are reported; a verdict is reported for every
  advanced start (no overall success from whichever start agrees).
- **Curve raised / lowered (reported for each start):** the observed gap (and, secondarily, the gap against B1k's
  temperature envelope) >= +2 with a 95% interval above 0, positive in all seeds and Holm-adjusted p <= 0.05 (two-sided
  test of zero gap; one Holm family over the advanced starts), or the mirror rule (lowered).
- **Targeting claim:** only if the targeted start's observed gap exceeds R-2M's by >= 2 with a 95% interval above 0.
- **Predictor comparison:** rank agreement of each predictor with the observed gaps (descriptive with three starts).
Reported: absolute forecasts and observations for every start; finishing, boxed rate, truncation and length of every
start and checkpoint; pass@16 by stratum of the base's test-set counts and by each start's own zero-count stratum;
final accuracy at equal RL-generated tokens and total teacher, SFT and RL compute; each predictor's rank agreement
with step-150 pass@1 and pass@16 (descriptive). Truncation sensitivity: the decisive pair (the targeted start and R-2M,
seed 8) at checkpoints 25, 50, 100, 150 and B1k8 at steps 25-150, all at 2,048 tokens, giving conditional gaps at that
budget (descriptive, one seed).

## Cost (incremental estimate, assuming the P29a pool and good packing)

Stage 1: five SFT trajectories (R, H, W, H \ W to 2M with the 0.5M checkpoint saved on the way; TailSFT-R to 2M),
nine starts plus the base, about 10M presented tokens, initial TailSFT and human-NLL scoring, and twenty 64-sample
evaluations: about 13 GPU-hours, about 2 hours of one 8 x A100 machine (about $65). Stage 2: nine RL runs, their 9 x 8
evaluations, and the 2,048-token check (eight evaluations): about 65 GPU-hours, about 8 hours of one machine (about
$260). Total about $330, plus setup and idle tails; timing is validated on the first SFT trajectory.

## Amendment 1 (October 28, before any teacher sample of this pool was produced)

The 72B teacher pool could not be produced (Pre-registration 29, fourth addendum). The teacher becomes
Qwen2.5-Math-7B-Instruct (revision ef9926d75ab1d54532f6a30dd5e760355eb9aa4d), single-GPU vLLM, 4 samples per training
problem, T = 1, 2,048 tokens, seed 2929, our prompt; each lane first runs a 48-problem gate that stops it if pass@1 is
below 0.6. Filtering, retokenization, the 1,023-token cap and the selections are unchanged. The teacher is weaker than
the 72B model and was itself trained with RL, so W and H \ W may keep fewer problems (survival by selection is
reported, and the 100-problem rule above still applies). The generation shares the machine of Pre-registration 32.
Amendment 1 addendum (Codex review): the two generation lanes download the teacher into separate directories (no
shared-directory race); a gate result is re-checked when a saved evaluation is reused; the two 48-problem gate outputs
overlap the generation shards and are excluded from pool assembly.

## Amendment 2 (October 29, from the Stage 0 table, before any teacher output was inspected)

Stage 0 (p30_stage0.py -> data/p30_stage0.json; base pass@1 on the 3,920 training problems 30.6% at 64 samples):
H has 456 problems (c = 0: 110; c = 1-2: 145; c = 3-4: 201); W has 14. Since W has fewer than 100 problems even
before teacher filtering, W and H \ W are dropped by the registered rule and H is the only targeted arm. With W gone,
N is redefined as the number of H problems with a surviving teacher solution; R draws N problems from all problems
with a surviving solution, stratified to H's subject mix (seed 3030). Stage 1 starts: R and H at 0.5M and 2M
presented tokens, TailSFT on R at 2M, and the base (three SFT trajectories, 6M presented tokens in total;
five starts plus the base; twelve 64-sample evaluations). Stage 2 advances R-2M, TailSFT-2M and H-2M. H-2M is the
targeted start unless its validation grid is empty ("no common support": it then cannot advance as the targeted
start); its forecast shift is the equal-weight mean over the validation grid of f_H(u) - f_B(u), and if the absolute
value of that mean is below 2 points it advances as a weak test. Everything else is unchanged.

Amendment 2 addendum (teacher pool, after generation): 9,821 of 15,680 teacher samples survive (5,793 not verified,
62 over 1,023 tokens, 4 duplicates); 3,290 problems keep at least one solution. About a third of all samples
(5,131) degenerate into random multilingual tokens without a final box, more often on later (harder) shards
(truncation 25-46% at 2,048 tokens): the model's own generation config is greedy, and T = 1 with top-p 1 lets it
fall into its low-probability tail (a hypothesis; no controlled decoding comparison was run). The registered audit
of 50 archived solutions (uniform, Random(3030), by Codex) found 14 with stray tokens or damaged notation and at
least one with flawed reasoning behind a verified answer; a non-ASCII count (51 of 9,887 verified responses above 1%)
misses ASCII corruption, so the pool is noisy, not clean. H keeps 210 of 456 problems
(c = 0: 12 of 110; 1-2: 65 of 145; 3-4: 133 of 201), so N = 210; R has 210 problems (16 also in H). Selections:
p30_select.py (data/p30_select_report.json, with SHA-256 of the archive and both selections). The validation
human-NLL corpus (data/val_frontier_256_human.json) takes the MATH test solutions from EleutherAI/hendrycks_math (all
255 matched).

## Amendment 3 (October 29, project lead's decision after the audit; before any Stage 1 run)

Because the registered audit found damaged text in 14 of 50 archived T = 1 solutions, and Stage 1 presents each H
solution about 21 times, the teacher pool is regenerated with nucleus sampling: Qwen2.5-Math-7B-Instruct (same
revision), 4 samples per training problem, T = 0.7, top-p 0.8, 2,048 tokens, seed 3131, eight 490-problem shards
(single-GPU vLLM; a shard fails if its pass@1 is below 0.4). Qwen recommends greedy decoding for chain-of-thought,
which would leave one distinct solution per problem; T = 0.7/top-p 0.8 keeps several. The survival rule, H's
definition, N, R's stratified draw (seed 3030), Stage 1 and Stage 2 are unchanged; p30_select.py is rerun on the new
pool, and a new random-50 audit is reported before Stage 1. The T = 1 pool and its selections (N = 210) are kept as a
recorded alternative and not used.

Amendment 3 result (pool regenerated in job worth-p30-teach3-s0-20261029-060407, all eight shards exit 0): pass@1
by shard 67.7-95.9% (T = 1 pool: 46.4-72.4%), truncation 0.6-3.8% (25-46%); 169 of 15,680 samples degenerate
(no box and more than 5% stray non-ASCII; T = 1 pool: 5,131). This supports, without controlling for each factor
separately, the hypothesis that T = 1 sampling caused the earlier degeneration. Survival: 13,035 samples (2,015
not verified, 357 over 1,023 tokens, 273 duplicates); 3,550 problems keep a solution. H keeps 262 of 456 (c = 0: 26 of
110; 1-2: 79 of 145; 3-4: 157 of 201), so N = 262 (809 solutions, 3.09 per problem, mean 473 tokens); R has 262
problems (950 solutions; 18 also in H). Hashes in data/p30_select_report.json; the random-50 audit follows.
Audit (Codex, random 50 of the 13,035 archived responses, Random(3030); codex_p30_audit_amendment3.md): 47 clean,
1 damaged notation (an ASCII number-list typo), 2 flawed explanations behind verified answers, 0 language switches;
over the whole archive, 9 responses contain short Chinese insertions and none contains a run of 5 or more non-ASCII
characters. The residual defects are documented as teacher noise; Stage 1 proceeds on this pool.

## Amendment 4 (October 29, after inspecting Stage 1's validation and test results; before any Stage 2 run)

Stage 1's starts all begin above B1k's support (test pass@1 61.6-69.1 against B1k's 21.6-58.1; validation 60.0-67.1),
so the registered validation grid (0.40-0.56) is empty for every start and no Stage 2 checkpoint could be compared
with RL from the base at matched pass@1. Two more long Dr. GRPO seeds would not fix this (P27's LH1 reaches 67.2 after
1,000 steps; RL from the 2M starts would likely exceed it). The design is therefore narrowed (Codex's recommendation,
codex_p30_stage2_design.md):
- **Stage 1b (screening, no RL):** sft_start.py with the same selections, settings and SFT seed 3030, saving starts at
  25K, 50K and 150K presented tokens for R and for H (a new trajectory: GPU nondeterminism makes it a replicate of
  Stage 1's, not a copy). Each start is evaluated on the 255 validation and 511 test problems (64 samples, T = 1,
  1,024 tokens).
- **Dose choice (validation only):** the largest of 25K, 50K, 150K at which both R's and H's validation pass@1 lie in
  [0.35, 0.45] and at least four points of the registered grid (0.40, 0.42, ..., 0.56) survive its filters for both
  starts; if none qualifies, the dose whose R and H validation pass@1 are both closest to 0.40 (sum of absolute
  differences), reported as a deviation.
- **Stage 2:** R and H at the chosen dose advance (seeds 8, 9, 10; Pre-registration 21's recipe); R-2M and
  TailSFT-2M no longer advance automatically (their Stage 1 results are reported; high-dose RL is a separate,
  unregistered extension). Checkpoints 1, 2, 5, 10, 25, 50, 100, 150 on the 511 test problems (32 samples, T = 1),
  step 150 also at T = 1.15 and 1.3. Reference: B1k seeds 8-10 (primary), with P27's LH1 (seed 8) reported as a
  single-seed extension beyond B1k's support; LH3 (lr 3e-6) only as a sensitivity comparison.
- **Readings:** as registered (same-seed reference, equal seed weights, joint bootstrap, at least two eligible
  checkpoints per seed, the 10% support rule, the +2 threshold, frozen forecasts before RL). Step 0 (the start itself)
  is reported separately from the post-RL estimate, and per-checkpoint gaps are shown: if only early checkpoints are
  eligible, the claim is about early RL, not persistence through step 150. The targeting claim (H vs R) is read at the
  chosen dose. The 2,048-token sensitivity check is kept.

Amendment 4 dose choice (validation only; job worth-p30-stage1b-s0-20261029-164830): validation pass@1 R 0.462 /
0.561 / 0.585 and H 0.372 / 0.566 / 0.586 at 25K / 50K / 150K presented tokens (truncation 36-47%). No dose has both
in [0.35, 0.45] (R at 25K is 0.462), so the fallback applies (a deviation from the primary rule): 25K has the
smallest summed distance from 0.40 (0.090, against 0.327 and 0.371). D = 25K presented tokens (actual: see the
starts' start_meta.json); R-25K and H-25K advance, and P34's OPD start uses the same nominal dose.

## Amendment 5 (October 30, after Stage 2's registered analysis; recommended by the three-way review)

Stage 2's checkpoints 100 and 150 lie beyond B1k's pass@1 support, and R25's contrasts were not interpreted for lack
of support. To extend the empirical reference, Dr. GRPO from the base is rerun with B1k's exact recipe and hardware
(train_grpo.py, one L40S, seeds 8, 9, 10) for 700 steps, saving and evaluating steps 25, 50, 100, 150, 200, 300,
400, 500, 600, 700 (511 test problems, 32 samples, T = 1, 1,024 tokens). These long runs (B1kL) are replicates of B1k,
not copies: the steps up to 150 are compared with B1k's for reproducibility, and an extended analysis (registered
before their results are inspected) reads R25 and H25's checkpoints 1-150 against each seed's B1kL trajectory, with
Stage 2's rules. The primary Stage 2 verdicts above are unchanged by this extension.
