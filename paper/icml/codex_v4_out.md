```latex
%%% SNIPPET 1: after "The answer budget as a confound"
% Replace that section's final sentence containing the P21 placeholder with:
% Re-evaluation changes the measurement protocol; \cref{sec:replication} reports replication with training and evaluation at $1{,}024$ tokens.

\section{Replication at a larger budget and a larger model}\label{sec:replication}

Pre-registration~21 trains and evaluates Qwen2.5-Math-1.5B at $1{,}024$ tokens in seeds~$8$--$10$. Each arm uses $150$ steps and $128$ rollouts per step, with problem order paired to its $512$-token counterpart. Both replications change the answer cap and loss normalizer together, so neither isolates the causal effect of the cap. With B1k denoting Dr.\ GRPO, matched-pass@1 differences in pass@16 are NSR1k--PSR1k $-0.25$, 95\% $[-0.72,+0.20]$; PSR1k--B1k $+0.11$, 95\% $[-0.26,+0.49]$; NSR1k--B1k $-0.15$, 95\% $[-0.62,+0.31]$; and AC1k against the B1k envelope $-0.12$, 95\% $[-0.60,+0.29]$. All four meet average equivalence, supporting the registered joint statement that the core result replicates at $1{,}024$ tokens. Equivalence requires pooled and per-seed 90\% intervals within $\pm2$ points; bootstrap intervals resample problems conditional on the trained seeds.

Step-$150$ truncation is $15.2$--$17.0\%$, below the $20\%$ reporting qualifier. Comparison points remain steps $50$, $100$, and $150$; step~$25$ supplies reference knots only. Omitting these knots gives PSR1k--B1k $+0.14$, 95\% $[-0.24,+0.53]$; NSR1k--B1k $+0.44$, 95\% $[-0.07,+0.97]$; and NSR1k--PSR1k $+0.33$, 95\% $[-0.22,+0.91]$. The early NSR comparison therefore depends on reference resolution.

At $7$B and $2{,}048$ tokens, Pre-registration~20 uses seeds~$8$ and~$9$. PSR7--B7 is $-0.09$, 90\% $[-0.48,+0.31]$, and NSR7--B7 is $-0.09$, 90\% $[-0.45,+0.27]$; both reach average equivalence. AC7 against the B7 envelope gives $-0.15$, 90\% $[-0.57,+0.16]$, 95\% $[-0.64,+0.23]$, replicating the allocation null. NSR7--PSR7 gives $+0.16$, 90\% $[-0.29,+0.57]$, but $13.7\%$ of bootstrap points in seed~$9$ are clamped, exceeding the $10\%$ guard. This contrast is not interpreted: three of four contrasts replicate, leaving the full joint claim unsupported. Both replications retain frozen AC settings; their null results concern that allocation rule.

Late test-set pass@16 is compressed near $94$--$95$, limiting sensitivity. The separately reported hard cohort contains $253$ problems selected by independent base screening for zero to two successes in eight samples, with labels audited before evaluation. Screening samples are not reused for evaluation, and label disagreements are removed without replacement. The trained B7 checkpoints there span pass@16 $80.8$--$84.5$. Only NSR7--B7 reaches average equivalence; PSR7--B7 is inconclusive, and the other two contrasts fail the clamping guard. The frozen frontier model meets rule~3 in aggregate, with RMSE $0.23$ over eight B7 seed-checkpoints. Its observed-minus-predicted pass@16 errors offset: $+7.0$ to $+22.9$ points for zero successes in $32$ base samples and $-6.8$ to $-2.7$ for one to four.

The evaluations use vLLM following registered amendments. The equivalence check gave base-7B pass@1/pass@16 of $30.5/88.0$ and $30.1/86.8$ in independent runs, versus $29.9/87.5$ with Hugging Face. For B1k seed~$9$, step~$50$ gave $52.2/87.5$ and $51.9/86.4$ versus $52.5/86.7$; step~$150$ gave $58.1/86.7$ twice versus $58.0/86.8$. Truncation and lengths also agreed. Every replication comparison uses one engine, including re-evaluated base models and references; the frozen rule~3 forecast retains its original Hugging Face base counts.


%%% SNIPPET 2: after "A per-problem model of the frontier", including its figure

\section{Process credit from minimum hints}\label{sec:minimumhints}

Pre-registration~23 tests process credit on the policy's own failed prefixes. During the first $100$ training steps, it probes up to four failed answers on each of up to four problems with at most one success in eight rollouts. Prefixes contain at most $256$ tokens and end before the answer box. Each receives zero, $32$, or $64$ tokens from a verified human solution, cut before its answer, with two continuations per hint level under a total $1{,}024$-token budget. Minimum-help credit (MH) assigns values $1$, $2/3$, or $1/3$ for the smallest tested hint level yielding a verified success, and $0$ if none succeeds. Values are centred within each problem, and credit updates only policy-generated prefix tokens. Its coefficient $c_t=2(1-t/100)$ decreases linearly to zero; Dr.\ GRPO training continues to step~$150$. The averaged-value control (AVH) uses identical hints, queries, and schedule but scores each prefix by mean success across its six continuations.

On $255$ validation problems in seed~$8$, MH against the B1k envelope gives $-0.01$ pass@16 points, 95\% $[-1.24,+0.89]$; AVH gives $-0.05$, 95\% $[-1.32,+0.91]$. MH against the AVH trajectory gives $+0.05$, 95\% $[-0.67,+0.84]$. Both mechanism gates are met: MH credits $520$ prefixes, exceeding the $150$ floor, and values differ in $61\%$ of probed problem-steps, above the $30\%$ gate. AVH credits $563$ prefixes with contrasts in $64\%$. Neither reaches the registered $+1.5$-point gain gate, so selection drops the method without a three-seed test. These intervals remain conditional on one training seed.


%%% SNIPPET 3: replace the abstract

\begin{abstract}
Comparisons of reinforcement learning with verifiable rewards (RLVR) often mix learning speed with repeated-sampling success. We compare pass@16 at matched pass@1 using interpolated references and registered tests with bootstrap intervals over problems. Across the tested correctness-only rules, budgets, and Qwen model sizes, the evidence supports changes in the rate of progress along a shared pass@16-versus-pass@1 curve within each setting, rather than changes in the curve. We derive expected per-problem updates and exact second moments under independent on-policy sampling. A $1.5$B replication trained and evaluated at $1{,}024$ tokens meets all four average equivalence tests. A $7$B replication at $2{,}048$ tokens meets three; the direct contrast between negative- and positive-sample reinforcement is not interpreted under the clamping guard. Fixed-budget rollout allocation meets no registered frontier-improvement criterion. A minimum-hint credit pilot gives $-0.01$ points, 95\% $[-1.24,+0.89]$, and is dropped by its selection rule. The answer budget can create apparent frontier gains: re-evaluation reduces the supervised warm-start step-$0$ gain from $+4.77$ points, 90\% $[+3.45,+6.07]$, at $512$ tokens to $+2.15$, 90\% $[+1.10,+3.20]$, at $1{,}024$. Matched-accuracy conclusions remain conditional on answer budget, reference support, and observed training seeds.
\end{abstract}


%%% SNIPPET 4: replace the introduction paragraph beginning "We derive the mean"

We derive the mean and exact second moment of correctness-only updates under independent on-policy sampling and bounded reward-dependent stopping, and test whether changes in advantage rules or successful-sample availability alter pass@16 at matched pass@1. The registered comparisons support changes in progress rate along a shared curve within each tested setting: all four contrasts replicate at $1{,}024$ tokens on $1.5$B, and three replicate at $2{,}048$ tokens on $7$B; the direct $7$B NSR--PSR contrast is not interpreted under the clamping guard (\cref{sec:replication}). Fixed-budget allocation supplies more successes to pilot-zero problems but fails the registered improvement rules and its model forecast (\cref{sec:allocation}). Answer-budget checks reduce the supervised warm-start step-$0$ gain from $+4.77$ points at $512$ tokens, 90\% $[+3.45,+6.07]$, to $+2.15$ at $1{,}024$ tokens, 90\% $[+1.10,+3.20]$, qualifying the apparent gain as largely a budget effect (\cref{sec:budget}). \pending{Check 21: remaining human-solution seeds and the pooled budget-effect verdict.} Minimum-help process credit meets its mechanism gates but fails pilot selection (\cref{sec:minimumhints}). \pending{P22: results of external versus self prefix guidance.} Together, these comparisons distinguish faster accuracy progress from a supported change in the pass@16-versus-pass@1 relation.


%%% SNIPPET 5: replace appendix table rows 20 through 23; retain the pending rows

20 & 7B replication (2,048 tokens) & PSR7--B7 and NSR7--B7 reach average equivalence; AC7 replicates the allocation null ($-0.15$, 95\% $[-0.64,+0.23]$). NSR7--PSR7 is not interpreted: $13.7\%$ clamping exceeds the $10\%$ guard. Three of four contrasts replicate. Rule~3 RMSE is $0.23$, with large offsetting stratum errors.\\
21 & 1.5B replication at 1,024 tokens & All four contrasts reach average equivalence: the core result replicates. AC1k against B1k's envelope gives $-0.12$, 95\% $[-0.60,+0.29]$. Truncation is $15.2$--$17.0\%$; step-$25$ reference knots affect the early NSR comparison.\\
21 (check) & Human-solution arms at 1,024 tokens & \pending{pending}\\
22 & External versus self prefix guidance & \pending{pending}\\
23 & Minimum-help credit (pilot) & Against B1k's envelope, MH gives $-0.01$, 95\% $[-1.24,+0.89]$, and AVH $-0.05$, 95\% $[-1.32,+0.91]$. MH against AVH's trajectory gives $+0.05$, 95\% $[-0.67,+0.84]$. Mechanism gates met: $520$ MH prefixes, contrasts in $61\%$. Neither arm reaches the $+1.5$ gain gate; no candidate, method dropped.\\


%%% SNIPPET 6: replace the Discussion section

\section{Discussion}\label{sec:moves}

The tested correctness-only interventions mainly change how quickly training reaches a given accuracy. At matched pass@1, the registered comparisons support a shared pass@16 curve within each setting. The larger-budget replication reaches average equivalence in all four contrasts; the $7$B replication does so in three, while its direct NSR--PSR comparison is excluded by the clamping guard. This is agreement within the registered tolerance over supported checkpoints. It permits smaller differences and does not establish equivalence at every checkpoint. The small replay gain also remains below the registered frontier-improvement threshold.

Similar pass@16 at matched pass@1 means similar repeated-sampling coverage at a fixed mean success probability. It does not identify the full distribution of per-problem success or the properties of generated solutions. Nor does it establish invariance for other task families, longer training, or non-correctness rewards. Minimum-help credit supplies external process information, but its failed selection in one validation seed cannot settle that broader question. Answer budgets remain part of the estimand: differential truncation can create apparent gains, while pass@16 compression at $7$B reduces sensitivity. \pending{Check 21: pooled human-solution budget-effect verdict.} \pending{P22: external versus self prefix guidance.}

The update identities do not imply the observed invariance. Per-problem coefficients can rotate the aggregate gradient through shared parameters, and optimizer state and noise can alter the realized update. Nor does aggregate prediction identify a learning mechanism: the $7$B model fit combines small aggregate error with large offsetting stratum errors.

The forthcoming causal update-direction test is therefore the natural next step (\pending{P25}). Perturbing direction at a common policy while controlling update magnitude and answer budget would ask whether updates follow the same local pass@1--pass@16 relation. Measuring both quantities after the intervention would connect direction to frontier displacement directly. This would test whether the shared curve reflects a common effective update direction, beyond the descriptive agreement of completed training trajectories.

% Draft-v4 consistency edits in Limitations:
% Replace its first two sentences with:
% Our experiments use 1B, 1.5B, and 7B models, with main runs of 150 steps on Qwen and 300 on OLMo. Registered conclusions for longer training remain incomplete.
% Replace ``The 1,024-token check remains partial.'' with:
% The pooled human-solution budget check remains incomplete (\pending{Check 21}).
```