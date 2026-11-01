# Schema debate, round 2 (November 1, 2026)
Read r1_claude.md, r1_codex.md, r1_fable.md and r2_claude.md (paper/debate_2026_11_01/). Goal: converge on ONE schema
for the user. In at most 700 words:
1. Critique the other two schemas (strongest objection to each; is anything wrong in the facts?).
2. HARVEST (Fable) vs rare-success constraints (Codex) vs the hybrid in r2_claude.md: which should be the paper's
   method, and why? Check HARVEST's novelty against STaR (2203.14465), RFT/rejection sampling, ReST/Expert Iteration,
   DART-Math (difficulty-aware rejection tuning), BroRL (2510.01180), and say what the defensible claim is. Check the
   key fact behind HARVEST (deep sampling solves 53% of never-solved training problems; source full_story.tex in the
   repo or wherever it is) and whether harvested solutions could be lucky guesses.
3. Equal-compute accounting: how to count harvest generation against RL rollouts so the comparison is fair (e.g., a
   Dr. GRPO arm given the same extra rollout budget, or BroRL-style larger groups at equal tokens).
4. The agreed schema: thesis; method; arms and controls; setting and metrics; Stage 0-3 with costs and gates; kill
   criteria; fallback. Mark any remaining disagreement in one line.
No praise.

## CORRECTION from the coordinator (read before answering; it changes the premises of r1)
The r1 brief wrongly called several RL-side levers untested. docs/full_story.tex (repo) records earlier
pre-registrations (P12-P18) that tested, at 1.5B (512 and 1,024 tokens) and partly at 7B: advantage rules including
pass@k objectives, MaxRL and BBG; entropy terms; replay and group completion; larger groups (G32: groups of 32, 4x
rollouts; exploratory, no frontier gain established, beyond-range checkpoints); rollout allocation (P18: AC/AV/HR,
4 pilot + 64 extra rollouts per step, at most 32 per problem; AC raised success-bearing pilot-zero groups to 50-54%
vs 18-21% for AV, yet stayed on the curve: "the starvation explanation is not supported in this form"); a self-SFT
warm start on base successes from 32 samples at 2.46M tokens (S32; step-0 gain +4.77 at 512 tokens falls to +2.15
at 1,024); human-solution warm starts on 487 never-solved problems (P19; not interpretable, trajectories do not
overlap). See full_story.tex sections around lines 560-700 and the summary at line 806. Deep sampling (256 extra
samples on 232 of 309 c<=4 problems) solves 53% of never-solved problems at least once, mean success rate 0.93%
(line 658). Not tested to our knowledge: BroRL-scale groups (hundreds of rollouts), training temperature above 1,
and harvesting at 512-2,048 samples per problem on c = 0 problems. Re-evaluate HARVEST and the rare-success constraint
in light of P18 (successes on rare problems during RL did not move the curve) and S32/D32.
