Review before launch, quickly: experiments/training/PREREGISTRATION_31.md and the new centralize_grads / sam_step code
and its call site in experiments/training/train_grpo.py (just before gradient clipping), against the Flat-RLVR paper
(arXiv 2610.29281: Algorithm 1, Appendix B.1-B.3, D, Table 11). Answer numbered: (1) deviations from the paper's
algorithm that matter (sign of the ascent for our loss convention: our loss is -(A log pi)/N, so .grad is the gradient
of the loss to minimize; is theta + rho g/||g|| the correct ascent on the loss?); GC definition; focal weight; annealing
index; (2) bugs (memory, no_grad/enable_grad, model mode, interaction with clipping, the 1-indexed t); (3) whether the
design is a fair test of the paper's claim on our curve and whether pass@32 at 32 samples is adequate; (4) wording.
