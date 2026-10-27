# Pre-registration 21: registered result (October 27, 2026)

1.5B, trained and evaluated at 1,024 tokens, seeds 8-10, vLLM evaluation (Amendment 2). Full output with simultaneous
bands: RESULTS_21_23_output.txt.

| Contrast | Estimate | 95% interval | Verdict |
|---|---|---|---|
| NSR1k vs PSR1k trajectory | -0.25 | [-0.72, +0.20] | average equivalence |
| PSR1k vs B1k trajectory | +0.11 | [-0.26, +0.49] | average equivalence |
| NSR1k vs B1k trajectory | -0.15 | [-0.62, +0.31] | average equivalence |
| AC1k vs B1k envelope | -0.12 | [-0.60, +0.29] | average equivalence (replicates Pre-registration 18's null) |

Joint statement: all four contrasts reach average equivalence -> **the core result replicates at 1,024 tokens**.
Holm-adjusted p = 1.000 for every test. Step-150 truncation 15.2-17.0% in every arm (below the 20% qualifier).
Sensitivity without the step-25 reference knots: PSR1k +0.14, NSR1k +0.44 (95% [-0.07, +0.97]), NSR1k vs PSR1k
+0.33; the step-25 knots matter for NSR's early segment. Greedy test accuracy at step 150: 63.6-64.6 in every arm.
