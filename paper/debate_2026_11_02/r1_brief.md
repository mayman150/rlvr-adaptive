# Direction debate, November 2, 2026 (Claude, Codex, Fable): what is the next research direction?

The user finds the external reviewer's direction interesting. Read: reviewer_directions.md (this folder);
paper/lit_2026_11_01/LIT_SYNTHESIS.md (verified literature, with codex_verify.md and fable_lit.md);
paper/icml/paper_v6.tex (current draft); paper/debate_2026_11_01b/PLAN.md (the headroom plan agreed yesterday);
experiments/training/PREREGISTRATION_39.md (Stage 0: hard sets, pass@128 re-evaluation, deep-sampling harvest pilot at
1,024 samples on 109 zero-observed-success training problems, SmolLM3 census; queued, not started).

Assets: Dr. GRPO trainer (train_grpo.py; interleaved SFT already implemented), SFT starts (sft_start.py), teacher
archive (Qwen2.5-Math-7B-Instruct solutions on 3,550 training problems), TEACHF/SELF/CORR starts and their RL runs,
700-step references, matched-accuracy estimator (analyze_p36.py), frozen hard sets (OlympiadBench + AIME 200/400),
SmolLM3-3B-Base imported. Constraints: about 11 weeks to the ICML deadline (late January 2027); 8 x A100 p4de at about
$32/hour with capacity delays of hours; 1.5B Dr. GRPO about 2 min/step/A100 at 1,024 tokens (3B and 4,096 tokens
unmeasured); budget should stay around $3-5K; one person coordinating.

Coordinator's opening position (Claude): combine the reviewer's directions 1 and 2 into one 2 x 3 study and run 4 as a
small audit. Information source {deep-sampled self (harvest), teacher} x delivery {SFT then RL; guidance withdrawn
during RL (MFC-style backward curriculum); SFT then RL with a retention term (e.g., the perplexity-novelty bonus of
2605.29190 or KL to the SFT start)}, plus plain Dr. GRPO, on problems with headroom (hard sets), read at matched
pass@1 (diagnostic of frontier shift) and equal compute (practical), with persistence to about 500 steps, a
reasoning-valid coverage check on a hard subset, and a second family for the winner. The paper's claim would be
whichever cell durably lifts the frontier, and why.

Questions (at most 900 words each):
1. Is the reviewer's direction the right one for an ICML method or science paper in 11 weeks? What is the single
   research question and the thesis that could come out either way?
2. Which of the reviewer's five directions are essential, which optional, which to drop? Critique the coordinator's
   2 x 3 design (too big? confounded? missing a control such as answer-free teacher traces or partial traces?).
3. Novelty against MFC (2609.13997), Curriculum RL (2606.22317), AdaBack (2506.18110), PrefixRL (2601.18795), DATPO,
   2605.29190, Quagmires (2510.01624), No Free Lunch (2610.39295): what exactly would be new?
4. A concrete staged plan (setting, models, arms, seeds, steps, budgets, metrics, gates, costs) that fits the
   constraints, reusing Stage 0 and existing assets; what Stage 0 should change before it starts.
5. Biggest risks (implementation of MFC-style curriculum in our trainer, compute, support at matched accuracy, judge
   validity for reasoning-valid coverage) and how to de-risk them in week 1.
No praise.

## UPDATE from the user (supersedes the deadline constraint above)
"Forget about ICML if we cannot catch it; we have NeurIPS. I need a good paper." Target: a strong paper, NeurIPS 2027
(deadline about mid-May 2027, about 6.5 months), ICML only if it is genuinely ready. Optimize for scientific quality
and a clear, defensible contribution, not speed. Budget can be planned in stages (state what each stage costs and buys);
more seeds, longer training, a second model family and mechanism analysis are now affordable in time.
