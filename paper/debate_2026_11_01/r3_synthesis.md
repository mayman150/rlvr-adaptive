# Round 3: coordinator's synthesis for sign-off (November 1, 2026)

Converged so far (r2_codex.md, r2_fable.md): HARVEST first; rare-success constraints only if fresh samples show an
absolute probability decline; harder held-out set + MATH-511; 2,048 tokens; 256 samples; pass@1/16/64/128; equal
total compute (charging all harvest generation) as primary, matched-accuracy frontier as the coverage diagnostic;
Stage 0 cheap and decisive. Codex's correction accepted: the 53% yield was on evaluation problems (P17 cohort), so
training-harvest yield is unverified until Stage 0.

New element (coordinator): a PEER arm to separate "stronger teacher" from "different model". SFT on verified
solutions from SmolLM3-3B-Base (a different family with similar coverage: pass@32 84.3 vs Qwen base 85.5, higher
pass@1 44.3 vs 21.6; its 64-sample census of the MATH training problems is queued), matched to TEACHF/HARVEST on
tokens, updates, prompts and lengths. If PEER moves the frontier like the teacher, "reasoning from a different model
at equal coverage" is the carrier (a cheap, teacher-free-in-strength method); if not, strength matters.

Proposed final schema
- Thesis: RLVR coverage is set by what enters the policy's training distribution, not by the RL algorithm; verified
  rare successes outside the policy's sampled reach (from deep self-sampling or a different model), distilled before
  RL, raise held-out pass@64/128 at equal total compute and lift the matched-accuracy frontier, whereas the same
  information as in-RL gradient (P18) and the same compute as longer RL do not.
- Stage 0 (week 1, <= $200): (a) training harvest pilot: 0/64 training problems (110), up to 1,024 samples each,
  stop after 2 verified solutions; blind audit of 50 derivations; (b) SmolLM3 census (queued) -> PEER pool;
  (c) 256-sample, 2,048-token re-evaluation of existing step-150 checkpoints (B1k, SELF, TEACHF, S32, 1,000-step) on
  MATH-511 hard strata and the new hard set; (d) fresh-sample retention probes on rare training problems at SELF and
  B1k checkpoints (suppression vs catch-up). Gates: audited usable yield >= 40% of the pilot cohort; hard-set
  validation pass@128 between 10% and 80% for the base.
- Stage 1 (weeks 2-3, <= $600, one 8 x A100 job): HARVEST x2, PEER x2, extended Dr. GRPO at equal compute x2,
  uniform shallow RFT (SELF) reused; 150 steps. Gate (screen, two seeds): HARVEST or PEER >= +1 vs SELF and >= +2 vs
  Dr. GRPO at matched pass@1 where supported, AND >= +2 hard-set pass@128 at equal compute with <= 1 pp pass@1 loss.
- Stage 2 (weeks 4-6, <= $1,500): pre-registered confirmatory test of the winner, 3 paired seeds, against the
  strongest control (Codex's list: extended Dr. GRPO, BroRL-style large groups if affordable, DART-style RFT,
  token-matched teacher, corrupted-label control), Holm family fixed in advance.
- Stage 3 (weeks 7-9, <= $1,000): SmolLM3-3B replication; 500-step persistence; 4,096-token sensitivity.
- Kill: Stage 0 yield gate fails; Stage 1 gains vanish at 2,048 or on held-out problems; equivalence within +/-1.
  Fallback: the mechanism paper (channel: gradient null vs SFT positive vs teacher positive; strength vs difference).

Sign-off questions (max 350 words): (1) accept or amend the schema (concrete edits only); (2) the Stage 1 gate:
Codex proposed >= 3 pp pass@128, Fable >= 1 pp vs SELF / >= 2 vs B1k; settle on one with a reason; (3) does the PEER
arm belong, and what confound does it carry (SmolLM3 pass@1 is higher; style; length)? (4) one sentence: the
probability this schema yields an ICML method paper vs the fallback.
