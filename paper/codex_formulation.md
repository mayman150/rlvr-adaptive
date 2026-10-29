Yes. The collection helps formulate a theory, especially by separating **observable success profiles, paths through those profiles, variational objectives, and discovery times**. The substantive corrections are in B and C.

**A — Correct, with qualifications about finite changes and “dispersion.”** Assume independent samples conditional on the problem and a fixed evaluation protocol. This is the sampling model already used in [paper.tex, lines 64–67](/Users/mmayman/Desktop/adaptive-rlvr/paper/icml/paper.tex:64). 

A precise proposition is:

> Let \(p_x\) and \(q_x\) denote reference and arm success probabilities, with \(\mathbb E p=\mathbb E q\), and set \(\delta_x=q_x-p_x\). For integer \(k\ge2\),
> \[
> \Delta_k
> =\mathbb E[(1-p)^k-(1-q)^k]
> =\operatorname{Cov}(w_k(p),\delta)+R_k,
> \qquad w_k(p)=k(1-p)^{k-1},
> \]
> where
> \[
> -\frac{k(k-1)}2\mathbb E[\delta^2]\le R_k\le0.
> \]
> Alternatively, the exact covariance representation is
> \[
> \Delta_k=\operatorname{Cov}(\bar w_{k,x},\delta_x),
> \qquad
> \bar w_{k,x}=\int_0^1w_k(p_x+t\delta_x)\,dt.
> \]

These follow from Taylor’s theorem and integration along the segment from \(p\) to \(q\). Thus your first-order covariance is correct; positive gain requires positive covariance with reference hardness, but positive covariance alone is insufficient for a finite change.

The ordering statement is also correct: if the arm success distribution is smaller in **convex order**, equivalently a mean-preserving contraction of the reference distribution, every \(P_k\), \(k>1\), increases weakly. For equally weighted finite vectors, this is Schur-concavity under majorization. 

However, **higher pass@16 alone does not establish a mean-preserving contraction or lower variance**. For example,
\[
p=(.10,.55,.55),\qquad q=(.20,.20,.80)
\]
both have mean \(0.4\). Pass@16 increases from \(0.9382\) to \(0.9812\), while variance increases from \(0.045\) to \(0.080\). The precise interpretation is that pass@\(k\) measures a particular concave functional of the success distribution; majorization provides a sufficient ordering across all such functionals.

At fixed every \(p_x\), redistributing probability among correct approaches or wrong answers leaves every \(P_k\) unchanged. Within-problem diversity can still affect subsequent learning and thereby change \(p_x\).

Existing counts can test:

- The matched changes in \(P_2,P_4,P_8,P_{16},P_{32}\). At exactly matched mean,
  \[
  \Delta_2=-\Delta\operatorname{Var}_x(p_x).
  \]
  Hence pass@2 supplies a direct variance diagnostic.
- Whether gains occur on low-reference-success problems and losses on high-reference-success problems.
- The covariance decomposition, accounting for binomial uncertainty. Naively using the same noisy reference counts inside both \(w_k(\hat p)\) and \(\hat\delta\) creates bias.

One practical qualification matters: [the registered reference interpolates pass@\(k\) values](/Users/mmayman/Desktop/adaptive-rlvr/paper/icml/paper.tex:74). If its endpoints have profiles \(p^0,p^1\), then
\[
P_k((1-t)p^0+tp^1)\ge(1-t)P_k(p^0)+tP_k(p^1).
\]
Consequently, gain against that interpolated reference equals gain against the interpolated **profile**, plus a nonnegative Jensen gap. Account for that gap when interpreting the registered gain through per-problem redistribution.

**B — The two examples are correct; the general invariance and necessity claims are incorrect.**

For an unrestricted policy class, binary reward, common \(\beta>0\), and forward KL \(D_{\mathrm{KL}}(\pi\Vert\pi_0)\),
\[
\pi^*(a\mid x)=
\frac{\pi_0(a\mid x)e^{r(a)/\beta}}{Z_x},
\qquad
\operatorname{logit}p_x^*
=\operatorname{logit}p_{0,x}+\frac1\beta.
\]
This gives the uniform-shift family on problems with \(0<p_{0,x}<1\); unsupported outcomes remain unsupported. It is the standard Gibbs variational solution. 

Your frozen law also defines a curve parametrized uniquely by mean accuracy. Write \(s_x=p_x(1-p_x)\). Then
\[
\dot m
=\mathbb E[s^2]+\tau(\mathbb E s)^2
=\operatorname{Var}(s)+(1+\tau)(\mathbb E s)^2.
\]
For the frozen \(\tau=-0.20\), this is strictly positive whenever an interior success probability remains. Therefore the initial profile and frozen \(\tau\) determine a unique \(P_k\)-versus-\(m\) graph on the attained range. See [law.py, lines 40–69](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/law.py:40). The code’s `uniform_shift` argument named `beta` is an affine slope, distinct from the KL weight above.

The useful invariant proposition is narrower:

> **Invariance under a change of clock.** Suppose a locally Lipschitz vector field \(F\) determines a unique flow, and methods start from the same state with
> \[
> \dot\theta^{(j)}=a_j(t,\theta^{(j)})F(\theta^{(j)}),
> \qquad a_j>0.
> \]
> They traverse the same state orbit. Wherever mean success is strictly monotone along that orbit, they share the same matched-accuracy coverage curve.

A different common function of \(p_x\) generally changes the orbit. For example, from \(p=(.1,.5)\), the rules \(\dot\theta_x=1\) and \(\dot\theta_x=p_x(1-p_x)\) have initial \(dP_{16}/dP_1\) approximately \(0.8724\) and \(0.3784\). Both use only current success probability, yet their curves differ.

Thus **leaving a particular curve does not require information beyond \(p_x\)**. Such information can provide another mechanism for leaving it. Common scalar dynamics with a common additive forcing do preserve initial ties and problem ordering under uniqueness; that is a defensible symmetry statement.

Test the frozen law, uniform shift, and affine alternative at the same accuracy using both aggregate \(P_k\) and per-problem profiles, as [law.py already supports](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/law.py:108). Your [correctness-only expected-update theorem](/Users/mmayman/Desktop/adaptive-rlvr/paper/icml/paper.tex:199) does not imply a closed success-only flow through shared neural parameters.

**C — The stationarity equation is correct for the written objective; the population coefficient needs care, and compression is conditional.**

Use \(\lambda\) for the collision coefficient:
\[
\mathcal C_x(\pi)=\sum_a\pi(a\mid x)^2,
\qquad
J_x(\pi)=\mathbb E_\pi r-\beta D_{\mathrm{KL}}(\pi\Vert\pi_0)-\lambda\mathcal C_x(\pi).
\]
For finite outcome support, positive base probabilities, \(\beta>0\), and \(\lambda\ge0\), this objective is strictly concave and has a unique optimum satisfying
\[
\pi^*(a)\propto
\pi_0(a)\exp\!\left(\frac{r(a)-2\lambda\pi^*(a)}{\beta}\right).
\]
This is an **implicit** equation. Collision probability is \(e^{-H_2(\pi)}\); penalizing collision favors larger Rényi-2 entropy but is not a linear Rényi-entropy bonus.

There is a factor-of-two distinction between differentiating the whole expected population reward and a score update that treats the bonus as fixed. For P32’s Batch bonus, with full sequence scores \(S_i=\nabla\log\pi(y_i)\),
\[
\mathbb E\!\left[\frac1G\sum_i(A_i+cB_i)S_i\right]
=\frac{G-1}{G}\nabla\!\left(p-\frac c2\mathcal C\right).
\]
Therefore a population score update with bonus \(-c\pi(a)\) corresponds to \(\lambda=c/2\). Your \(2c\pi(a)\) stationarity term is correct when explicitly defining the objective with \(\lambda=c\). The Batch rule itself uses repetitions of other samples in the group. 

Actual P32 additionally masks answer tokens and has no KL term: [P32, lines 18–31 and 47](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_32.md:18), implemented in [train_grpo.py, lines 983–1028](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/train_grpo.py:983). Its masked update is generally not the full gradient of the collision objective. The Gibbs formulation should therefore be presented as a controlled model.

For the compression question, P32 merges all accepted answers into one correct outcome class. Define
\[
H_x=\sum_{a\ {\rm wrong}}\pi(a\mid x,\mathrm{wrong})^2,
\qquad
\mathcal C_x=p_x^2+(1-p_x)^2H_x.
\]
At \(\lambda=0\), increasing the collision coefficient at fixed reward and fixed \(\beta\) gives
\[
\left.\frac{\partial p_x}{\partial\lambda}\right|_0
=-\frac2\beta p_x(1-p_x)
\underbrace{[p_x-(1-p_x)H_x]}_{h_x}.
\]
Thus correctness decreases when \(p_x>H_x/(1+H_x)\), and increases below that threshold. The effect also vanishes near saturation. “It lowers \(p_x\) most where \(p_x\) is highest” is consequently not correct.

A usable conditional proposition is:

> **Local compression under homogeneous wrong-answer collision.** Introduce a common correctness reward multiplier \(\eta(\lambda)\) and choose it to keep mean success fixed. At \(\lambda=0\), let \(s_x=p_x(1-p_x)\) and let \(\mathbb E_s\) denote expectation weighted by \(s_x\). Then
> \[
> \left.\frac{\partial p_x}{\partial\lambda}\right|_{\rm matched\ mean}
> =-\frac2\beta s_x(h_x-\mathbb E_s h),
> \]
> and
> \[
> \left.\frac{\partial P_k}{\partial\lambda}\right|_{\rm matched\ mean}
> =-\frac{2\mathbb E s}{\beta}\operatorname{Cov}_s(w_k(p),h).
> \]
> If \(H_x=H\) across problems, \(h=(1+H)p-H\) is increasing in \(p\). Every \(P_k\), \(k>1\), then increases to first order, and every convex functional of the success distribution decreases to first order.

This is a local result under a specified mean adjustment. Its susceptibility scales as \(\lambda/\beta\) holding the baseline profile fixed; changing \(\beta\) also changes that profile.

Heterogeneous wrong-answer concentration can reverse the prediction. With
\[
p=(.2,.4),\qquad H=(.1,1),\qquad\beta=1,
\]
the matched-mean derivative is \((-0.06144,+0.06144)\): the harder problem loses success and the easier one gains it. This increases spread and lowers every \(P_k\), \(k>1\).

P32 can test the conditional compression model through low-/high-reference-success changes, matched pass@2, higher-\(k\) gains, and the interaction with \(H_x\). Reconstruct the **full** outcome collision, including the “no parsed final box” class: P32’s registered conditional collision excludes that class ([lines 64–74](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/PREREGISTRATION_32.md:64)). Distinct-wrong-answer counts measure the manipulation; they do not identify compression. Wrong-answer diversity may cause changes in success probabilities, so “gain comes through \(p_x\)” does not establish that diversity was causally irrelevant.

**D — Needs an explicit condition; a discovery-time bound is sound.**

Define \(T_x\) as the first on-policy sample on problem \(x\) receiving verified reward one.

> **Rare-success discovery bound.** Suppose, conditional on every history before the first success, sample \(t\) has success probability at most a deterministic \(\epsilon_t\). Then
> \[
> \Pr(T_x\le M)
> \le1-\prod_{t=1}^M(1-\epsilon_t)
> \le\sum_{t=1}^M\epsilon_t.
> \]
> For a uniform cap \(\epsilon\), this is at most \(1-(1-\epsilon)^M\le M\epsilon\).

The proof iterates conditional failure probabilities. For frozen independent sampling with probability \(p_{0,x}\), equality holds in the geometric expression and \(\mathbb E T_x=1/p_{0,x}\). Under frozen sampling, a Dr. GRPO group of size \(G\) supplies nonzero correctness advantages with probability
\[
1-p_{0,x}^{G}-(1-p_{0,x})^{G}.
\]

This bounds acquisition of **direct verified signal**, not learning through transfer from other problems. Shared-parameter updates, teacher data, or P32 bonuses on repeated wrong outcomes can change the pre-success probability; applying the bound requires controlling that change.

The influence interpretation is sound: each draw in the \(k\)-sample OR event is pivotal when the other \(k-1\) draws fail, so total influence is exactly \(k(1-p)^{k-1}=w_k(p)\). This is the elementary Margulis–Russo viewpoint underlying A’s weights. 

Existing base counts and P32’s logged group counts can test mixed-group frequencies and their evolution ([training log fields](/Users/mmayman/Desktop/adaptive-rlvr/experiments/training/train_grpo.py:1924)). They cannot establish a hazard cap from initial counts alone. In particular, zero successes in 32 draws does not establish \(p_0=0\): its individual one-sided 95% binomial upper bound is approximately \(0.0894\).

The family connections are:

| Family and supplied lines | Useful contribution to formulation |
|---|---|
| **113**, [2795–2799](/tmp/oaimath/CONTENTS.md:2795); **238**, [5847–5856](/tmp/oaimath/CONTENTS.md:5847) | **A/C:** motivate distinguishing laws with fixed observables from their remaining entropy or full distribution. Useful analogies; matching and shuffle bounds do not supply the coverage proposition. |
| **119**, [2940–2949](/tmp/oaimath/CONTENTS.md:2940) | Binary coarse-graining is a useful modeling perspective. Its specific contraction inequalities require a uniform cube and a specified independent-noise channel. |
| **186**, [4320–4329](/tmp/oaimath/CONTENTS.md:4320) | **A/D:** influence and pivotality language is useful. The exact OR calculation already gives the needed weight; graph/hypergraph threshold bounds add no neural-learning guarantee. |
| **221**, [5229–5238](/tmp/oaimath/CONTENTS.md:5229); **234**, [5648–5657](/tmp/oaimath/CONTENTS.md:5648) | **B/C:** encourage a variational formulation with explicit base measure, reward, and regularizer. The generic Gibbs identity does the work here; specialized spin-glass formulas are unnecessary. |
| **217**, [5093–5102](/tmp/oaimath/CONTENTS.md:5093) | Its free-energy fluctuation results supply no identified tool for A–D. |
| **227**, [5471–5473](/tmp/oaimath/CONTENTS.md:5471) | **D:** useful distinction between equilibrium optima and finite-time accessibility. Importing SK barriers would require justified dynamics or a comparison theorem. Rare-event sampling alone does not establish metastability. |
| **093**, [2304–2313](/tmp/oaimath/CONTENTS.md:2304); **139**, [3343–3352](/tmp/oaimath/CONTENTS.md:3343) | Potential tools for a future log-concave diffusion or oracle-sampling model satisfying their premises. Those premises have not been established for these runs. |
| **360**, [8731–8749](/tmp/oaimath/CONTENTS.md:8731); **374**, [9018–9027](/tmp/oaimath/CONTENTS.md:9018) | **A:** suggest comparing success-distribution measures. The elementary bound \(\lvert P_k(\mu)-P_k(\nu)\rvert\le kW_1(\mu,\nu)\) is usable; their geometric transport-map regularity results are not needed. |

The collection is useful for formulating objects and assumptions: observable fibers, curve orbits, variational objectives, and first-hit times. I would put A and B’s monotonicity/clock propositions into the paper, and present C as a conditional mechanism hypothesis for P32. D is defensible as a bound on direct discovery under an explicit pre-success probability cap. The specialized collection theorems do not establish a universal base-model frontier for these experiments.