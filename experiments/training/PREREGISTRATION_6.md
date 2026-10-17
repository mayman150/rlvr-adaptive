# Pre-registration 6: do implementation choices create the PSR / NSR separation?

Declared October 17, 2026, before any run of this design (see this file's commit time).

## Why

Strictly on-policy with sequence-sum losses, PSR and NSR have the same expected update, and
in seed 1 of Pre-registration 4 they reached the same pass@64 (83.2 and 83.0). Zhu et al.
(2025) report that NSR preserves pass@k where PSR does not, with verl, which differs from our
setting in three ways that the two-moment theorem does not cover:

1. **Reuse of samples:** one rollout batch of 1,024 prompts feeds updates on mini-batches of
   256, so three of four updates use stale samples, with PPO clipping at 0.2.
2. **Length normalization:** each answer's loss is averaged over its tokens.
3. **Normalization by the selected samples:** PSR and NSR average over the answers they train
   on, so the normalizer depends on the rewards.

## Runs

PSR and NSR, seeds 1 and 2, "verl-like": each rollout batch of 64 prompts by 8 answers feeds
4 consecutive updates of 16 prompts (clipped surrogate, ratio 0.2), with per-answer token
averaging and normalization by the number of trained answers. Generated answers (19,200) and
updates (150) match the on-policy runs. Otherwise as Pre-registration 5 (pass@32 on 128
problems at steps 25, 50, 75, 100, 150; greedy accuracy; entropy probe every 5 steps).

## Prediction and decision

**Separation recreated** if, in both seeds, NSR minus PSR pass@32 at step 150 is at least 3
points, or NSR's probe entropy over steps 100 to 150 is at least 1.2 times PSR's. Then the
separation is driven by implementation, and single-factor ablations follow. **Not
recreated** otherwise: the separation then needs conditions outside this scale (larger
models, longer training, longer answers).
