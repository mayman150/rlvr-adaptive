# Pre-registration 38 (DRAFT, not frozen): does a start's coverage gain last only when it carries new information?

Drafted October 31, 2026, after P36 job 1's interim results and before P35 phase 2's results (TEACHF versus SELF) or
any run of this design. To be frozen, with the arm choice below, after reading E' (TEACHF versus SELF) and after
Codex's review.

## Why

Interim P36 (job 1, descriptive): self-SFT starts (CWS, UNI, TAIL) raise matched pass@16 by +2 to +4 points at RL
steps 1-10, but the gain falls to about 0 to +1 by steps 50-150, and every self arm ends at Dr. GRPO's endpoint
(pass@1 about 57, pass@16 about 87). The teacher start H25 keeps about +2 through step 50 and reaches pass@1 65. An
exact decomposition by base-difficulty stratum (results_p26/p36_interim_strata.txt) shows that the self starts' early
gain sits on test problems the base solves sometimes (c = 1-8 of 64: +1.4 to +1.7 points of +1.8 to +2.0) and is
gone by steps 50-150, whereas H25's gain moves toward problems the base never solved (c = 0: +0.86 at steps 1-10,
+1.52 at step 50). Working hypothesis (H_info): RL recovers whatever a start reweights among outcomes the base can
already sample; a lasting coverage gain needs solutions the base cannot sample.

## Part A: persistence (700 RL steps)

Arms (existing 25K starts; Dr. GRPO, Pre-registration 21's recipe, 700 steps, saving and evaluating steps 1, 2, 5,
10, 25, 50, 100, 150, 200, 300, 400, 500, 600, 700; 511 test problems, 32 samples, T = 1, 1,024 tokens):
- H25L: the H25 start (Pre-registration 30), RL seeds 8-10.
- TEACHFL: the TEACHF start (Pre-registration 35), RL seeds 8-10.
- SELFL: the SELF start (Pre-registration 35), RL seeds 8-9 (two lanes), descriptive only. [Arm choice to be fixed
  after E': if SELF's phase-2 gap at steps 50-150 is already about 0, SELFL is dropped and the lanes go to Part B.]
Reference: B1kL seeds 8-10 (Pre-registration 30, Amendment 5; one L40S each; the arms run on A100s, as in Stage 2).
Readings (analyze_p36.py's estimator and rules A and B, fixed-checkpoint support rule):
- A1: H25L and TEACHFL against B1kL_s over checkpoints 200-700 (late) and 1-150 (early), each its own Holm family
  of two; "persists" = rule A's "curve raised" criterion with +1 instead of +2 (pooled >= +1, 95% interval above 0,
  every seed positive, Holm p <= 0.05) on the late window. [Threshold to be agreed in review.]
- A2: TEACHFL against SELFL's trajectory (descriptive; two seeds).
- Stratum decomposition by base count (c = 0, 1-8, 9-32, > 32) at every checkpoint (descriptive).

## Part B: new versus familiar teacher information (150 RL steps)

Two teacher starts at the 25K dose (sft_start.py defaults; teacher archive data/p30_archive.json, verified
solutions; SFT seed 3030), differing only in which problems supply the solutions:
- NEW: the 59 training problems with c <= 1 of 64 base successes and an archived teacher solution (154 solutions).
- FAM: 59 problems drawn (seed 38) from the 681 with c >= 33 and a teacher solution, matched on subject counts to NEW.
RL seeds 8-10, 150 steps, checkpoints as in P36.
Readings: B1 NEW against FAM's trajectory (rule B, family of one); B2 each against B1k_s (rule A, Holm over two);
B3 (descriptive) the c = 0 test-stratum contribution for NEW versus FAM. H_info predicts B1 > 0 at late checkpoints
and a larger c = 0 contribution for NEW.

## Cost (estimate)

Part A: 8 lanes x 700 steps (about 21 hours of RL plus 3.5 hours of evaluation) on one 8 x A100 machine, cap 28 hours:
about $900. Part B: 6 runs on a second machine (about 9.5 hours): about $300.
