# The external reviewer's recommended directions (condensed from the review the user pasted, November 1, 2026)

Bottom line: build the paper around the finding that self-generated and teacher-generated reasoning have different
effects on subsequent RL at matched prompts and lengths. Central question: "What information does the teacher introduce
that the student cannot efficiently recover through its own sampling, and what training mechanism allows the student to
retain that information after the teacher disappears?" Proposed title: "What Moves the RLVR Coverage Frontier? A
Pre-Registered Study of Optimization, Search, and Guidance." Proposed hypothesis: standard RLVR optimization
redistributes probability mass across existing behavior, whereas structured exploration and targeted external reasoning
guidance can change the effective problem-level coverage frontier.

Directions (reviewer's priority):
1. Information-boundary hypothesis / deep sampling vs teacher (ESSENTIAL): on the same hard training problems, SFT on
   verified solutions harvested from the base at 64 / 256 / 1,024 samples vs strong-teacher solutions vs teacher with
   final answer removed vs teacher partial reasoning; identical RL after. Outcomes: deep self matches teacher (sampling
   efficiency), teacher stays better (information gap), or teacher-guided RL beats teacher SFT then RL (delivery matters).
2. Persistence (VERY HIGH): teacher SFT -> RL vs teacher guidance during RL with progressive withdrawal (MFC-style) vs
   interleaved teacher SFT and RL vs the same curriculum with deep-sampled self guidance; equal tokens and compute;
   evaluate at steps 0-700; matched-pass@1 pass@16 plus pass@128/256; track whether the traces that produced the gain
   are forgotten or RL concentrates mass.
3. Reasoning-valid coverage (VERY HIGH): CoT-Pass@k-style C_k^valid on a hard subset (judge or manual), compared with
   answer pass@k at matched answer pass@1.
4. Structured exploration at matched accuracy (ESSENTIAL): DATPO / tree rollouts, MFC, rollout allocation, UCB-Con
   evaluated under the matched-pass@1 protocol.
5. Reasoning-distribution mechanism: strategy-cluster entropy and valid-strategy probabilities before SFT, after SFT,
   after RL; likelihood of post-RL successful traces under the base model ("effective sampling support").
Also: a second independent model family; hierarchical seed/problem uncertainty; 2,048-3,072 tokens primary for warm
starts; answer-free teacher traces (remove all explicit answers, not only the final box); "zero-observed-success"
wording; the law's tau is fitted on RL runs. Reviewer score of draft v6: 5/10; 7/10 plausible with the DATPO/MFC
matched-accuracy tests, reasoning-valid coverage, deep sampling vs teacher, and persistent teacher-guided curriculum.
