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
unfinished count, and final boxed answers grouped by the verifier's normalization (`verifier.normalize`).
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
token-averaged over the corpus), and a frozen power-map (BOPTR-style) forecast from
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
