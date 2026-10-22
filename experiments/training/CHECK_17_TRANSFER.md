# Check 17: do training updates transfer to rarely solved questions?

Declared October 22, 2026, before any measurement. Evaluation only (no training).

**Premise under test.** For a question the policy almost never solves, its own term in the update
is missing (no sampled success), so it improves only through transfer from the questions trained on:
Delta p_x ~ eta sum_x' c_x' <grad p_x, grad p_x'>. If this is what limits diversity at matched
accuracy, Dr. GRPO (B) should transfer little to rarely solved held-out questions, and the warm
start (S32), whose gains sit on such questions, should transfer more.

**Probe.** 395 held-out evaluation problems (never trained on) that the base model solves 1-4 of
32 times ("rare", 149) or 5-16 times ("medium", 246) at a 1,024-token budget; per problem up to 4
correct and 4 wrong base answers, all with a parseable final box (`data/transfer_traces.json`,
2,913 traces, fixed seed). For each checkpoint: per-token mean log-likelihood of every trace.
Per problem, the **margin** = mean over its correct traces minus mean over its wrong ones (style
shared by both cancels); **transfer** = the margin's change from the base model.
Checkpoints: B at steps 50, 100, 150 and S32 at 0 (after fine-tuning), 50, 100, 150; seeds 8-10.

**Readings fixed in advance** (pooled over seeds, 90% bootstrap intervals over problems):
1. "B transfers little to rare questions" holds if B's step-150 transfer on rare problems is at most
   half its transfer on medium problems, or its interval includes 0.
2. "The warm start transfers more" holds if S32's step-150 transfer on rare problems minus B's has
   an interval above 0.
3. Reported: per-problem rank correlation between transfer and the later pass@16 change; the same
   with correct-trace likelihood alone (style-confounded, secondary); a length-matched version
   (correct and wrong traces paired by length).
If 1 and 2 both hold, the transfer explanation is pursued as the method's basis; otherwise it is
dropped.

Cost: about $15-25 (6 single-GPU lanes, forward passes only).
