Task: a careful literature review for an ICML 2027 paper (deadline late January 2027). Our draft:
/Users/mmayman/Desktop/adaptive-rlvr/paper/icml/paper_v6.tex (thesis: RLVR algorithms move along a matched-accuracy
coverage frontier; small teacher-solution SFT starts beat self-generated starts on identical prompts but their
advantage decays during RL; large SFT doses land back on the frontier). An external review
(/Users/mmayman/Desktop/adaptive-rlvr/paper/lit_2026_11_01/external_review.md) says recent work moves the frontier with
structured exploration and teacher-guided curricula. Codex is separately verifying those nine items; do NOT duplicate
that. Your job is the BROADER search, 2025 to October 2026, on: (1) methods that claim to expand the reasoning boundary
or pass@k at large k beyond the base model: teacher/hint-guided RL on zero-success problems, backward or partial-trace
curricula, prefix/hint scaffolding with withdrawal, tree or branched rollouts, rollout scaling, self-distillation with
privileged information (e.g., answer-conditioned), deep/rejection sampling (RFT, DART-Math, STaR variants); (2) evidence
on persistence: do teacher-injected gains survive later RL; (3) evaluation: reasoning-valid pass@k (CoT-Pass@k),
strategy-level diversity metrics, matched-accuracy or compute-matched comparisons, support/likelihood analyses of
whether RL-found solutions were already likely under the base model. For each paper give: arXiv ID / venue, models and
data, the main quantitative claim, whether it compares at fixed step/budget or matched accuracy, whether results are on
Qwen-Math only or other families, and relevance to us. Verify at least the abstract of every paper you list; mark
unverified ones. End with: (a) the 5 papers we must engage with most, (b) open gaps no paper has filled that our
matched-accuracy, pre-registered instrument could fill. Max 1,500 words. Write to
/Users/mmayman/Desktop/adaptive-rlvr/paper/lit_2026_11_01/fable_lit.md and return it. No file changes otherwise, no
compute. Economical tool use; if rate-limited, wait and retry.
