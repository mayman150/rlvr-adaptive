(1) At **$4/A100-hour**, Qwen costs $0.267/step and Smol $0.533/step at 2,048 tokens. Qwen confirmation requires 5×3×700 + 3×500 = 12,000 steps: **$3,200**. Smol requires 3×3×700 + 3×500 = 7,800 steps: **$4,160**. Reference extensions are included.

Use Fable’s provisional $7 per 200-problem, 256-sample Qwen reading: $14 at 512 samples. Provisionally allow $28 for Smol; inference throughput needs separate calibration.

| Stage | Training | Evaluation/acquisition | Total |
|---|---:|---:|---:|
| 0: acquisition, gates, calibration | $40 | $560 | $600 |
| 1: five-arm Qwen pilot, exploration audit, Smol smoke runs | $300 | $100 | $400 |
| 2: Qwen confirmation | $3,400 | $1,600 | $5,000 |
| 3: Smol selected-delivery confirmation | $4,400 | $2,400 | $6,800 |
| 4: four Qwen controls, three seeds, 700 steps | $2,400 | $800 | $3,200 |
| **Total** | **$10,540** | **$5,460** | **$16,000** |

Training allowances include SFT and startup beyond the calculated RL costs. Evaluation/acquisition allowances cover generation, teacher acquisition, and judge APIs; researcher labor is excluded. These are planning caps requiring Stage 0 calibration.

Fable’s $5.5–7K buys shorter runs, fewer arms/seeds, and lighter evaluation. The agreed confirmation training alone costs $7,360; the discrepancy principally reflects scope.

(2) **Minimal credible version: approximately $12K**—$8.5K training and $3.5K acquisition/evaluation. Preserve Qwen’s factorial, Smol’s selected-delivery replication, three paired seeds, 700-step persistence, 1,200-step references, and registered late 512-sample readings.

Use answer-free teacher banks throughout and retain a three-seed answer-only control. Cut separate answer-free variants, visible-prefix mechanism arms, retention variants, clustering, and routine early large-k evaluations. Preserve exposure accounting, shared-problem and acquisition-pipeline comparisons, and calibrated reasoning-validity audits. Describe delivery effects at the protocol level; omit scheduling-specific causal claims.

(3) **Remaining disagreement:** whether delivery effects must replicate across Smol’s full factorial; selected-delivery replication supports a narrower cross-family conclusion.