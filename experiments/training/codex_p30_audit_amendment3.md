**47 of 50 responses were clean.** I flattened all **13,035** archived responses in training-problem order, then response order, and used `random.Random(3030).sample(flattened, 50)`. I read every sampled response in full against its question and reference answer. The [archive](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/data/p30_archive.json) matches the SHA-256 recorded in the [selection report](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/data/p30_select_report.json). Files were unchanged.

| Classification | Count |
|---|---:|
| Clean | 47 |
| Stray tokens or damaged notation | 1 |
| Flawed reasoning behind a verified answer | 2 |
| Language switch (non-English) | 0 |
| Other | 0 |

Every non-clean case is listed below. **Training indices and response numbers are zero-based; response numbers refer to positions in the archived `responses` list.**

| Training index | Response | Classification and defect |
|---:|---:|---|
| 1521 | 0 | **Damaged notation:** one enumerated sequence is `21, 22, 24, 25, 26`, followed by “23 is prime.” The intended consecutive sequence is `21, 22, 23, 24, 25`. The final answer, **28**, is correct. |
| 1621 | 3 | **Flawed reasoning:** fixes the fifth chair as empty and calculates \(4!=24\), then abruptly concludes **120**. It omits the five choices for the empty chair; the correct count is \(5\times4!=120\). |
| 3171 | 0 | **Flawed reasoning:** incorrectly says modular division requires “4 and 12” to be coprime to modulus 12. The displayed reduction \(8x\equiv4\pmod{12}\Rightarrow2x\equiv1\pmod3\) is valid because coefficient, right-hand side, and modulus are divided by 4. Subsequent algebra and the final answer, **5**, are correct. |

Compared with the previous **T = 1** audit, stray tokens/damaged notation fell from **14/50 (28%) to 1/50 (2%)**, a **26 percentage-point reduction**. The current defect is a localized ASCII number-list typo; no random-token degeneration appeared in the sample. Flawed explanations remain: **2/50** here, versus **at least one** previously, which does not establish an improvement in their frequency.

Across the whole archive:

| Cheap automatic indicator | Responses flagged |
|---|---:|
| Any CJK character, scanning Han/kana/Hangul ranges | **9/13,035 (0.069%)** |
| Any run of ≥5 consecutive non-ASCII characters outside common math symbols | **0/13,035** |

The nine CJK flags contain short Chinese insertions. There were also **zero** runs of five consecutive non-ASCII characters before applying any math-symbol exemption; neither indicator detects the three sampled defects above.

**The regenerated pool is fit to proceed with the registered SFT starts, with the residual defects documented as teacher noise.** Text quality has improved substantially, while the two flawed explanations show that final-answer verification still does not certify sound reasoning.