# Related work posted in 2025–2026 (search of October 21, 2026)

Two literature searches on arXiv (abstract pages opened for every entry; "verified" = checked again
by hand, including the full text where noted). Venue only as stated on arXiv.

## Overlap with the diagnosis

| arXiv | Paper | Venue | Claim | Relation |
|---|---|---|---|---|
| 2510.13651 (verified) | Davis, Recht: What is the objective of reasoning with RL? | – | Binary-reward RL is ascent on a monotone transform of per-prompt success (rejection sampling: log; GRPO: arcsin sqrt p) | Prior form of our "same direction, per-question weight"; credit it |
| 2510.23049 | Thrampoulidis et al.: Advantage shaping as surrogate reward maximization | TMLR | pass@k advantage shaping = surrogate reward | Prior form of our pass@k-weight analysis |
| 2511.16231 | Yu et al.: Pass@k is a diagnostic, not an objective | – | pass@k gradient = reweighted pass@1 gradient | Same as our pass@4 result's explanation |
| 2602.11128 | Heckel et al.: Asymmetric prompt weighting for RLVR | – | Prompt weights set the time to reach a target accuracy | Supports "weights set the speed" |
| 2602.21189 (verified) | Barakat et al.: Prompt interference; why pass@k optimization can degrade pass@1 | NeurIPS 2026 | Upweighting low-success prompts can rotate the update away from pass@1 through shared parameters | Must answer: per-question weights can change the parameter-space direction |
| 2606.15455 (verified, full text) | Yuan et al.: Diversity collapse via the lens of overtraining (BBG) | – | Collapse = overtraining saturated problems; gate updates by E[k(1-p)^(k-1)], k = 16 | Closest framing and nearly our coverage weight; pass@1 lower than GRPO on every benchmark |
| 2609.22547 | Yang et al.: RLVR is a kernel, not a function | – | pass@k crossovers need confidence bands; per-prompt effects | Supports our per-question identity |
| 2610.07405 | Rath et al.: What pass@k cannot measure | NeurIPS 2026 workshop | GRPO vs RFT; pass@k misses diversity | Evaluation critique; cite |
| 2512.21625 | Tang et al.: Rethinking sample polarity (A3PO) | – | Negative samples drive exploration | Counter-claim to "NSR's diversity is speed" |
| 2605.07137 | Ingle et al.: Adaptive negative reinforcement (A-NSR) | – | Scheduled / confidence-weighted NSR, Qwen2.5-Math-1.5B | Counter-claim on our exact model |
| 2602.05548 | Yu et al.: Implicit advantage symmetry (A-GRAE) | NeurIPS 2026 | Down-weighting correct trajectories helps exploration | Counter-claim |
| 2607.01490 | Decugis et al.: FADE | – | Advantage decomposition; best accuracy–diversity trade-off | Counter-claim |
| 2601.15609 | Fan et al.: When sharpening becomes collapse | – | Finite-batch sampling bias changes the outcome | Tension with "noise sets only speed" |
| 2602.02710 (verified) | Tajwar et al.: Maximum likelihood RL (MaxRL) | ICML 2026 | Interpolates expected reward and likelihood of correct rollouts | Per-question reweighting; popular baseline |

Not found published: noise sets the speed along one curve; NSR on PSR's curve on two families; the
random-sign noise causal test; the matched-pass@1 protocol; temperature does not beat the training
curve; pass@k at equal pass@1 = spread of success across questions. No method paper above evaluates
at matched pass@1 (PBA has one early-stopped control).

## Overlap with the method

| arXiv | Paper | Method | Relation |
|---|---|---|---|
| 2610.24236 (verified, full text) | Li, Galhotra, Cardie: TAR, adaptive replay | 128 base samples per MATH problem, up to 3 diverse correct ones kept, replayed with an NLL loss gated on the policy falling below the base; Qwen2.5-0.5B/1.5B-Instruct, GRPO, group 8 | Scoops base seeding; 1.5B pass@1 41.2 vs 37.9, pass@8 61.1 vs 60.2 (not matched) |
| 2610.22290 | EliteReplay | Quality-diversity archive of base solutions, NLL replay; text-to-SQL | Seeding outside math; pass@1 below baselines |
| 2610.11064 | Re:Max | One stored rollout per solution mode, uniform replay | Close to our v1 |
| 2606.15455 | BBG (above) | Posterior pass@16 weight as a gate on updates | Our weight, used on-policy instead of for replay |
| 2607.20543 | PBA | Base samples gate prompts; KL anchor on unsolved | Baseline |
| 2602.06717 | F-GRPO | Focal down-weighting of easy groups | Cheap baseline |
| 2605.07114 | HORA | Rollout allocation by posterior hit probability | Stronger version of our G32 control |
| 2607.27787 | LSPO | LoRA scaffold for all-fail prompts | Hard-question rescue |
| 2607.05339 | TREK | Short KL distillation on verified solutions, then GRPO | Like our S32 control |
| 2606.01281 | POPO | Prioritized replay groups replace zero-variance groups | Replay baseline |
| 2603.16157 | DyJR | Recent-trajectory buffer with a Jensen–Shannon regularizer | NeurIPS 2026; its baselines show reviewer expectations |
| 2606.22317 | Curriculum RL beyond the base model | pass@k-located boundary plus teacher guidance | Boundary-expansion claim |
| 2509.25808 | AR3PO | Adaptive rollouts plus reuse of earlier correct responses | Precursor of v1 |

Not published as one method: replay questions chosen by d pass@k / d log p, base traces retired at
the first own success, the log-sum archive objective, matched-accuracy evaluation.
