# Check 20: Dr. GRPO checkpoints at steps 10-40 (reference for slow arms)

Declared October 24, 2026. Several arms (BBG, W-REINFORCE, MaxRL, AR, MXGC) sit at pass@1 between the
base model (14.1) and Dr. GRPO's step 50 (about 37), where the registered reference is a straight chord;
for a concave trajectory this understates the reference and overstates their gains. This check trains
Dr. GRPO (Pre-registration 12's settings) for 50 steps, seeds 8, 9, 10, saving steps 10, 20, 30, 40, 50,
on Nibi H100s, and evaluates them on the 511 problems (32 samples, T = 1). Reported: each slow arm's gain
at matched pass@1 with and without these points, as a sensitivity analysis next to the registered
verdicts (which are unchanged). Step 50 is compared with Pre-registration 12's step 50 (L40S) to check
hardware agreement.

**Amendment (October 25, 2026, before any run).** Moved from Nibi to AWS SageMaker L40S (the hardware of
Pre-registration 12), so the step-50 hardware comparison becomes a same-hardware reproducibility check.
