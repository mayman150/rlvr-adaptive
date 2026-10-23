# Interim numbers used in docs/report.pdf (October 23, 2026)

These were computed from the evaluation files in `results/` with the scripts named below; they are
interim (not every registered evaluation is complete) and are not registered verdicts.

## Pre-registration 14, main arms (three seeds, 511 problems)
Registered procedure via `analyze_v12`/`analyze_v14` functions (TAR and G32 contrasts pending):
- D32 against B's trajectory-and-temperature envelope: +1.02, 90% [+0.35, +1.56], 95% [+0.23, +1.68];
  per seed +0.71, +1.23, +1.12 (average agreement). Step-150 pass@1, D32 minus B: -5.35, 90% [-5.99, -4.74].
- D32 against U32's trajectory -0.39; against C05's +0.92; against S32's: insufficient overlap.
- Reported: U32 +1.37 [+0.62, +2.11]; C05 +0.33; S32 insufficient overlap (beyond B's pass@1 range).

## Pre-registration 16 pilot (seed 12, 255 validation problems; `analyze_v16.py`)
- AR: +1.30, 95% [-0.44, +3.02]; step-150 pass@1 minus B -6.6 (not eligible: pass@1 criterion).
- MXGC: +1.12, 95% [-0.75, +2.69]; step-150 pass@1 minus B -6.5 (not eligible).
- S32: step-150 pass@1 minus B +8.5 (insufficient overlap). GC, PG, SD, MXR: pending.

## Groups of 32 (G32) and the warm start (S32), seed means over available seeds
- G32 seed 9 step-150 pass@1 52.2 (B 44.0). Pooled over available seeds: step 150 pass@1 52.5.
- S32 step 150: pass@1 52.5, pass@16 78.7 (three seeds).
- Residuals against the law's base-start trajectory (tau -0.20), observed minus predicted pass@16:
  S32 steps 0/50/100/150: +3.34/+3.25/+3.48/+3.56 (three seeds); G32 steps 50/100/150: -1.72/+1.91/+3.87
  (two seeds); B steps 50/100/150: -1.19/-0.81/+0.50.

## Check 16 (1,024-token budget; partial)
- B seed 10 step 150 at 1,024: pass@1 57.7, pass@16 88.4, truncated 15.5% (at 512: 44.6, 75.2, 48%).
- Base model at 1,024: pass@1 21.6, pass@16 79.3, truncated 36.2% (at 512: 14.1, 62.6).
- S32 seed 9 step 150 at 1,024: pass@1 63.8, pass@16 89.7, truncated 4.9% (at 512: 53.3, 79.8).

## Check 17 (transfer probe; partial: B seed 8, S32 seed 9)
Mean per-problem margin m_x (nats per token), rare / medium problems:
- B seed 8: base 0.5262 / 0.6444; step 50 0.5394 / 0.6587; step 100 0.5462 / 0.6653; step 150 0.5509 / 0.6697.
- S32 seed 9: step 0 0.5648 / 0.6973; step 50 0.5693 / 0.7019; step 100 0.5728 / 0.7057; step 150 0.5752 / 0.7075.
