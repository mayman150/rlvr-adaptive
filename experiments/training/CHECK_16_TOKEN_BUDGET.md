# Check 16: does the 512-token answer budget drive the results?

Declared October 22, 2026, before any evaluation at the new budget. Requested by the independent
reviewer after Pre-registration 14's interim results.

**Why.** At 512 tokens, 48% of Dr. GRPO's (B) sampled answers at step 150 are truncated, 33% of the
warm start's (S32) and 59% of seeded memory's. Hard problems need longer answers, so the warm
start's gain (concentrated on problems the base never solves) may partly be "finishes within the
budget", and every curve in the project was measured at 512 tokens.

**What.** Existing checkpoints re-evaluated with answers up to 1,024 tokens (no new training): the
base model; B at steps 50, 100, 150; S32 at steps 0, 50, 100, 150; seeds 9 and 10; same 511
problems, 32 samples, T = 1.

**Reported.** Truncation rates at both budgets; B's curve at 1,024 against 512; S32 against B's
trajectory at matched pass@1 at 1,024 where their pass@1 ranges overlap (Pre-registration 12's
interpolation, shared bootstrap draws), and S32's step-0 gain at both budgets; pass@16 by base
difficulty.

**Reading fixed in advance.** If S32's step-0 matched-accuracy gain at 1,024 is below half its gain
at 512, or its 90% interval includes 0, the warm-start advantage is reported as largely a budget
effect. If B's truncation at 1,024 stays above 20%, a larger budget is needed before conclusions
about the budget. Training at a larger budget is a separate decision.

Cost: about $40 to $60 (8 single-GPU evaluation lanes).
