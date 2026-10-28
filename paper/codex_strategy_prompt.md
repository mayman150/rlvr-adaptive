Strategy critique (be candid, compact; search the web for prior work; link claims). The user wants the paper to be
"very interesting and good for the community" and asks whether to change something in pre-training or post-training to
solve the issue that RLVR follows a fixed frontier. State: docs/full_story.tex "Where we stand"; INVESTIGATION_27.md;
first P28 results: SimpleRL-Zoo-7B (pass@1 70.3, pass@16 93.6) and Oat-Zero-7B (86.3, 95.0) lie within about 1 point
BELOW the frozen law's prediction from the 7B base counts (FORECAST_20.json: 94.5 and 96.1); our B7 is -0.1.

My proposal: (1) "change the starting model, predict the RL outcome": targeted SFT/mid-training before RL (e.g. teacher
solutions on the base's hardest / confidently-wrong problems vs random problems at equal tokens), measure the new
starting model's per-problem counts, predict its post-RL curve with the frozen law, then run RL and verify; framing:
RL methods do not move the frontier, changing the start moves it as the law predicts, and the law becomes a cheap
predictor of a base/mid-training choice's RL outcome. (2) evaluate the strongest counter-claims (ProRL, BroRL,
DeepScaleR from R1-distilled bases, long CoT) on our curve at long budgets, evaluation only. (3) later: RL on several
bases, law predicts each frontier.

Answer, numbered: 1. Is this the most interesting/impactful direction for ICML and the community? What would make it
excellent vs incremental? 2. Prior work that already does (1)-(3) or predicts RL outcomes from base models (e.g.
mid-training papers like OctoThinker, interplay of pre/mid/RL, base-model "RL-readiness" predictors, BOPTR) - novelty
risk? 3. Design of (1): what exactly to fine-tune on, doses, controls, the decisive test, failure modes (e.g. SFT changes
format/length so the law's inputs shift), cost at 1.5B. 4. Anything better I am missing (pre-training or post-training).
