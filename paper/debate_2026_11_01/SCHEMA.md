# Research schema (agreed by Claude, Codex and Fable; November 1, 2026)

Sources: r1-r3 in this folder. Remaining disagreement: the Stage 1 gate (resolved below toward Fable's, with reason).

## Question
Is RLVR's coverage limit set by what the policy happens to sample, so that more sampling fixes it, or by what the
model can produce at all, so that a different source is needed? And through which channel must the extra
information enter: SFT before RL or gradient during RL?

## Thesis (scoped to the tested settings)
RLVR coverage is set by what enters the policy's training distribution, not by the RL algorithm. Verified solutions
to problems that a frozen 64-sample census misses, obtained by deep self-sampling (HARVEST) or from a different
model (PEER), and distilled before RL, raise held-out pass@128 at equal total compute. Over ten RL-side levers
(advantage rules including pass@k objectives, entropy terms, larger groups, rollout allocation, replay, prefixes)
and shallow self-SFT starts, none does.

## Method
- HARVEST: on training problems with 0 of 64 base successes, sample up to 1,024 attempts per problem in capped
  rounds, stopping after 2 verified solutions. Run a blind audit of the derivations, not only the answers. Distill a
  25K-token SFT dose, then run unchanged Dr. GRPO.
- PEER (exploratory source ablation): the same pipeline using SmolLM3-3B-Base's verified solutions on the same frozen
  prompts, with length bins, tokenized dose, updates and audited validity matched. A positive result means
  "alternate-source distillation"; a null result does not prove strength is needed.
- Rare-success constraint (Codex): only if fresh-sample probes show an absolute decline in rare-problem success
  during RL.

## Setting and metrics
- Model and data: Qwen2.5-Math-1.5B on MATH train. Tests: MATH-511 for continuity, plus 400-500 deduplicated,
  open-ended OlympiadBench-style problems. Validation and test sets are frozen separately before Stage 0, and
  screening uses validation only.
- Budget and sampling: 2,048 tokens (4,096-token sensitivity check); 256 samples per problem; report pass@1, 16, 64
  and 128.
- Primary reading: pass@128 at equal total compute, with a 1-point pass@1 non-inferiority margin. Every census
  sample, rejected generation, verification, SFT step and RL step is charged.
- Compute control: extended Dr. GRPO, which spends the harvest's compute on more RL.
- Diagnostic only: the matched-pass@1 frontier, where it has support. Reasoning-strategy variety is audited
  separately.

## Stages
- Stage 0 (week 1, <= $200; generation and evaluation only):
  - (a) Training harvest pilot with audit.
  - (b) SmolLM3 census (job already queued), giving the PEER pool.
  - (c) 256-sample, 2,048-token re-evaluation of existing step-150 checkpoints (B1k, SELF, TEACHF, S32, 1,000-step),
    asking whether earlier nulls hid a large-k effect.
  - (d) Fresh-sample retention probes on rare training problems (suppression or reference catch-up?).
  - (e) Throughput calibration.
  - Gates: audited usable yield on at least 40% of the pilot cohort (otherwise drop HARVEST; PEER can still
    proceed); base validation pass@128 between 10% and 80%.
- Stage 1 (weeks 2-3, <= $600; one 8 x A100 job):
  - Arms: HARVEST x2, PEER x2 and extended Dr. GRPO x2; SELF reused.
  - Gate (screen, on validation): at least +2 pp pass@128 over extended Dr. GRPO at equal compute, and at least +1
    pp over SELF, with both seeds positive and a pass@1 loss of 1 pp or less.
  - Why not Codex's 3 pp: the largest verified content effect so far is +1.5 to +2.1, and a 3 pp screen on two
    seeds has little power against a true +2.
- Stage 2 (weeks 4-6, <= $1,500): pre-registered confirmatory test of the winner, 3 paired seeds.
  - Controls: extended Dr. GRPO, DART-style rejection fine-tuning with the same generator, token-matched teacher, and
    a corrupted-label control. BroRL-style large groups only if the budget remains.
  - Rule: pooled >= +2, Holm-adjusted 95% interval above 0, every seed positive.
- Stage 3 (weeks 7-9, <= $1,000): SmolLM3-3B replication, 500-step persistence and 4,096-token sensitivity. Weeks
  10-11 are for writing.

## Kill criteria and fallback
- Kill: yield gate fails; gains vanish at 2,048 or 4,096 tokens or on held-out problems; equivalence within +/-1
  (an interval wholly inside). A failure establishes the channel result only, not a model-support ceiling.
- Fallback: a mechanism paper separating discovery, retention, across-problem coverage and strategy variety. It
  covers RL-side levers versus SFT starts versus teacher and peer sources, with large-k and hard-set evidence.

## Odds (subjective)
| Source | ICML method paper | Mechanism fallback | Neither |
|---|---|---|---|
| Codex | 30% | 50% | 20% |
| Fable | 30% | 55% | 15% |

Both branches use the same runs.
