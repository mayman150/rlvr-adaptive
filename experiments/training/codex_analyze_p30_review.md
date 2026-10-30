Several fixes are needed in [analyze_p30.py](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/analyze_p30.py) before use. No files changed; checks used synthetic inputs only, without reading Stage 2 data.

The primary gap formulas, interpolation/tie ordering, full-data eligibility, and forecast agreement/failure thresholds are correct. Priors, initial states, laws and references are recomputed using shared problem draws with tau frozen. **LawCache’s latest-key design is valid:** L254 is draw-major, with seeds inside; every call for a key uses the same indices. Recomputing the base law for subsequent arms is redundant but correct.

1. **Law-support failures do not drop observed gaps — L134.** Only forecast/residual summaries lose those checkpoints, violating the registered joint support rule. Insert before assigning `g["fc"]`:

   ```python
   if key != "full" and (fx is None or fb is None):
       out[st] = None
       continue
   ```

   If a full-data eligible checkpoint lacks either forecast, report forecast/residual as unscorable; do not silently average a smaller full-data checkpoint set.

2. **Bootstrap pooling changes seed weights; empty support can crash — L163–168.** `nanmean` averages whichever seeds survive. Replace:

   ```python
   pooled = np.mean(np.vstack([ps[s] for s in full]), axis=0)
   ok = pooled[np.isfinite(pooled)]
   # Return an explicit "not interpreted (support)" result if ok.size == 0.
   ```

   Also replace the estimate’s `np.nanmean` with `np.mean`. Preserve affected-draw counts before filtering; print each seed exceeding 10% as “not interpreted.” Synthetic checks reproduced both unequal weighting and the empty-percentile crash.

3. **Missing verdicts and shrinking Holm families — L258, L278, L308–329.** Insufficient-overlap starts disappear from the verdict output; P34 primary disappears when R is unavailable, and secondary is skipped when primary lacks overlap. Fix:
   - L258: store `"elig": el`, retaining even a single eligible checkpoint.
   - Keep fixed families: `{R25,H25}` separately for observed gap and residual; `{P34 primary,P34 secondary}` for P34. Use `p=1` for an unavailable/uninterpretable member rather than removing it.
   - Compute/report P34 primary and secondary independently, printing their missing/overlap statuses. Its **secondary formula is otherwise correct**: interpolate same-seed R, including its start, at each OPD checkpoint; do not replace it with a difference of pooled B1k gaps.

4. **Unregistered P30 equivalence verdict — L180–190.** P34 explicitly registers observed-gap equivalence; P30 does not. Change the signature to:

   ```python
   def curve_verdict(t, q, equivalence=False):
   ```

   Prefix L189’s condition with `equivalence and`; pass `equivalence=True` only for P34. Raised/lowered thresholds, 95% intervals and unanimous seed signs otherwise match.

5. **Envelope reading is incomplete — L248, L292–294.** Eligibility always uses trajectory support, rather than the trajectory/temperature union used by P21’s envelope. For envelope eligibility, use:

   ```python
   ranges = [curve(p, full_idx)[0][[0, -1]] for p in (traj, temps)]
   lo, hi = min(r[0] for r in ranges), max(r[1] for r in ranges)
   ```

   Filter checkpoints against `lo, hi`. For P30, report the secondary envelope’s 95% interval, seed signs, support status and registered raised/lowered rule with its R/H Holm family; the current pooled 90% interval alone is insufficient.

6. **Targeting can claim support after a failed support guard — L301–306.** Before interpreting H−R, require both contrasts to have sufficient overlap, finite estimates and all seed affected-draw fractions ≤0.10. Otherwise print “targeting: not interpreted.” The paired difference and ≥2/95%-above-zero rule are correct; no additional Holm test is registered.

7. **Required uncertainty/reporting is missing — L285–300.** Add the residual tolerance test alongside the zero-bias test:

   ```python
   z = t["draws"][np.isfinite(t["draws"])]
   p_beyond_2 = np.mean(z <= 2) if t["est"] >= 0 else np.mean(z >= -2)
   ```

   Report its direction explicitly. Before discarding `per_draw`, calculate checkpoint bands, for example:

   ```python
   np.nanpercentile(
       [np.nan if d[st] is None else d[st]["obs"] for d in per_draw[s]],
       [5, 95],
   )
   ```

   Label these pointwise 90% intervals; report corresponding forecast/residual bands and P34-secondary checkpoint gaps. Print eligible steps and unsupported checkpoints, and scope each claim to the observed overlap.

8. **LH1 violates the single-seed extension — L339–343.** Replace:

   ```python
   for s in (8,):
       ...
       if u > curve(refs[s][0], full_idx)[0][-1]:
   ```

   Currently seeds 9/10 are compared with seed-8 LH1, and “beyond support” incorrectly assumes step 150 has the largest reference pass@1. Keep LH3 separately labeled as sensitivity if included.

9. **Step 0 and remaining registered reports are incomplete.** Move start loading/reporting before L235’s missing-checkpoint `continue`; keep step 0 excluded from post-RL averaging. Add these exact output quantities:

   | Missing output | Required calculation/source |
   |---|---|
   | Absolute observations/forecasts | Store/report checkpoint `100*u`, `100*v`, `100*r`, `100*fx`, `100*fb`; currently only differences appear. |
   | P34 pre-RL comparison | `100*pk(*sc, k, full_idx)` for OPD, R and base, `k=(1,16,32,64)`. |
   | P34 endpoint/training reports | Step-150 pass@1 against same-seed R/B1k; `opd_log.json`’s `reverse_kl_per_token` and `batch_reward`; actual dose from `start_meta.json`. |
   | Evaluation diagnostics | Retain evaluation JSON; report `1-truncated_rate`, `boxed_rate`, `truncated_rate`, and length fields for every start/checkpoint, including temperature evaluations. |
   | Strata/ranks/compute | Report stratum pass@16 via `pk(c,n,16,np.flatnonzero(mask))`; predictor rank agreement with gaps and step-150 pass@1/pass@16; equal-RL-token accuracy and teacher/SFT/RL compute. |
   | Amendment 4 reporting | State the registered 25K fallback deviation; retain R-2M/TailSFT Stage-1 results; run the descriptive 2,048-token comparison for H/R seed 8 and B1k8 at steps 25/50/100/150. |

10. **Input handling can select arbitrary runs or crash — L60, L88, L230–232.** Reject multiple matches in `one`; preflight missing references, 511-problem alignment and sample depths. Select one explicit OPD seed-10 root instead of repeatedly replacing a complete run with any nonempty candidate. For confirmatory execution, replace the argument declaration with `choices=(2000,)`, or explicitly label other draw counts exploratory.