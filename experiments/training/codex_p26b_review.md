1. **Bugs to fix before launch.** Small CPU checks reproduced these:

   - **RTD can retain the gradient it claims to remove.** At [train_grpo.py:901](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/train_grpo.py:901), a zero projected residual leaves the original `G` in `.grad`, while reporting `removed_share=1`. Zero/drop negligible residuals using a relative tolerance. An entirely zero teacher gradient also incorrectly reports removal share 1.
   - **Diagnostic model retention remains.** The last `lp` at [direction_diag.py:202](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/direction_diag.py:202) retains parameters through its autograd graph after `del runner`. This also occurred with checkpointed backward in the CPU check. Delete `lp` after backward; otherwise the next checkpoint load can OOM.
   - **Zero gradients falsely count as rotation.** [OD:932](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/train_grpo.py:932) and [PW:994](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/train_grpo.py:994) return cosine 0 for undefined zero-vector comparisons. PW’s cancelling example produced uniform `alpha=(1,1)` but cosine 0. Use the registered inactive cosine 1.
   - **Diagnostic zero denominators remain.** Zero checkpoint displacement crashes [line 113](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/direction_diag.py:113); an exactly rank-one update makes `perp` retention divide by zero at [148](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/direction_diag.py:148). Zero sketches produce NaNs at [168](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/direction_diag.py:168) and [217](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/direction_diag.py:217). Guard finite, positive denominators.
   - **Cross-spectrum “shares” are misleading.** [Line 65](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/direction_diag.py:65) divides signed eigenvalues by positive eigenvalue mass: eigenvalues `(-1,1)` become shares `(1,-1)`. The top-share gate must handle indefinite/noise-dominated estimates explicitly.

2. **The ten earlier points:**

   | Earlier point | Status |
   |---|---|
   | 1. Setup guards | **Resolved** for the registered combinations; snapshots/tables follow guards. |
   | 2. Degeneracy | **Partial:** original OD/PW failures fixed; remaining cases above. |
   | 3. Diagnostic memory | **Partial:** training mode/microbatching fixed; `lp` retention remains. |
   | 4. Power iteration | **Partial:** exact W0 and consistent final pair fixed; OD logs residual, RTD discards it; convergence unverified. |
   | 5. OD interpretation | **Mostly:** rescaling explanation corrected; rank-one exception needs qualification below. |
   | 6. PW backward | **Confirmed sound:** probe clearing, constant denominator, weighted pass, joint clipping. |
   | 7. Sketch scope | **Resolved:** diagnostic now uses the same selected matrices and sketch implementation. |
   | 8. GPU fit | **Still unverified.** |
   | 9. Weight geometry/control | **Mostly resolved:** lazy loading, separate trajectory Gram, `other`, rounding fidelity added. Rank notions are correctly distinguished.  |
   | 10. Selection/mechanism | **Partial:** rotation and seed references added; zero-step readings and `perp` matching remain incomplete. |

3. **Teacher arm:** the accumulation order is correct at [line 1621](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/train_grpo.py:1621): clear gradients → teacher backward → project teacher alone → accumulate RL → clip jointly → AdamW. `lambda=1` with `N=128×1024` implements one unit-advantage answer per retained solution. Steps 0–9 are unprojected. Given the promised all-problem archive, [load_discovery](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/train_grpo.py:507) is appropriate: missing responses are accepted; retained responses undergo length, verification and box checks. Backward accumulation itself is supported by PyTorch’s semantics. 

   After the fixes, TD/RTD are sound **screening arms for residualizing teacher gradients against the tracked joint-training direction**. The stronger “outside the shared RL direction” claim is not isolated: the tracker includes teacher learning from step 0; bilateral projection removes whole row/column subspaces; other parameters remain unprojected; Adam momentum and coordinate scaling affect the realized displacement.  The performance/TD-contrast gates are reasonable for screening; explicitly require RTD itself to satisfy both B1k performance conditions and the inherited overlap guards.

4. **Amendment wording to correct:**

   - [Line 55](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_26.md:55): zero `D` skips projection; a lost vector restarts and normally projects **that same step**.
   - [Line 58](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_26.md:58): unchanged applies to updates **parallel to tracked `uvᵀ`**, not every rank-one update.
   - [Line 74](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_26.md:74): the expected gradients are conditional on mixed groups. The clipped cross-spectrum ratio does not directly establish dominance of an unconditional RL direction.
   - [Line 70](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_26.md:70) promises per-matrix retention; output is aggregate. The 80% recovery rule needs positive `B1k−other` gain, and `perp`’s matched-pass@1 reference still needs specifying.