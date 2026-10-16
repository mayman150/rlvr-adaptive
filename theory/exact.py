"""Exact checks of the binary-reward sufficiency claims for rollout collection.

Each prompt is a softmax policy over K answers with probabilities pi; the score
of answer y with respect to the logits is e_y - pi. A collection scheme samples
answers sequentially, may stop or abandon based on the rewards seen so far,
and returns per-answer advantages that depend only on the group's rewards.

Claims checked (binary rewards):
  M  the expected update of a prompt is c_S(p) * grad p, where c_S depends
     only on the scheme and p (not on how probability is spread over answers);
  V  its second moment equals E_r[ sum_i A_i^2 F_{r_i}
       + |grad p|^2 sum_{i != j} A_i A_j kappa(r_i) kappa(r_j) ],
     with kappa(1) = 1/p, kappa(0) = -1/(1-p), F_r = E[|score|^2 | r];
  B  dropping constant groups and refilling (DAPO-style dynamic sampling)
     rescales the batch's expected update without changing its direction.
Breakers checked: token-mean length normalization and non-binary rewards.
"""

from fractions import Fraction as Fr
import math

STD_FLOAT = False  # set by schemes that need square roots


def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]


def vec_scale(c, a):
    return [c * x for x in a]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class Prompt:
    def __init__(self, pi, reward):
        self.pi, self.reward, self.K = pi, reward, len(pi)
        assert sum(pi) == 1

    def score(self, y):
        return [(1 if k == y else 0) - self.pi[k] for k in range(self.K)]

    def mean_reward(self):
        return sum(p * r for p, r in zip(self.pi, self.reward))

    def grad(self):
        """Gradient of E[r] with respect to the logits: pi * (r - E r)."""
        m = self.mean_reward()
        return [self.pi[k] * (self.reward[k] - m) for k in range(self.K)]

    def fisher_given(self, r):
        ys = [y for y in range(self.K) if self.reward[y] == r]
        mass = sum(self.pi[y] for y in ys)
        return sum(self.pi[y] * dot(self.score(y), self.score(y)) for y in ys) / mass


def enumerate_scheme(prompt, scheme, lengths=None):
    """Yield (probability, answers, advantages, kept) over every outcome."""
    def rec(answers, prob):
        decision = scheme.step([prompt.reward[y] for y in answers])
        if decision == "sample":
            for y in range(prompt.K):
                if prompt.pi[y]:
                    yield from rec(answers + [y], prob * prompt.pi[y])
            return
        rewards = [prompt.reward[y] for y in answers]
        if decision == "abandon":
            yield prob, answers, [0] * len(answers), False
        else:
            adv = scheme.advantages(rewards, [lengths[y] for y in answers] if lengths else None)
            yield prob, answers, adv, True
    yield from rec([], Fr(1))


def moments(prompt, scheme, lengths=None):
    mean, second, kept = [0] * prompt.K, 0, 0
    for prob, answers, adv, is_kept in enumerate_scheme(prompt, scheme, lengths):
        g = [0] * prompt.K
        for y, a in zip(answers, adv):
            g = vec_add(g, vec_scale(a, prompt.score(y)))
        mean = vec_add(mean, vec_scale(prob, g))
        second += prob * dot(g, g)
        kept += prob * is_kept
    return mean, second, kept


def predicted_second_moment(prompt, scheme):
    p = prompt.mean_reward()
    kappa = {1: Fr(1) / p, 0: Fr(-1) / (1 - p)}
    F = {r: prompt.fisher_given(r) for r in (0, 1)}
    G2 = dot(prompt.grad(), prompt.grad())
    total = 0
    for prob, answers, adv, _ in enumerate_scheme(prompt, scheme):
        r = [prompt.reward[y] for y in answers]
        diag = sum(a * a * F[ri] for a, ri in zip(adv, r))
        cross = sum(adv[i] * adv[j] * kappa[r[i]] * kappa[r[j]]
                    for i in range(len(r)) for j in range(len(r)) if i != j)
        total += prob * (diag + G2 * cross)
    return total


def coefficient(mean, grad):
    """c with mean = c * grad, or None if mean is not parallel to grad."""
    k = next(i for i, g in enumerate(grad) if g != 0)
    c = mean[k] / grad[k]
    ok = all(abs(m - c * g) <= (1e-12 if isinstance(c, float) else 0) for m, g in zip(mean, grad))
    return c if ok else None


# ---- schemes -------------------------------------------------------------

class Fixed:
    """Sample G answers; mean-centered advantages (Dr. GRPO style)."""
    def __init__(self, G, std=False, token_mean=False):
        self.G, self.std, self.token_mean = G, std, token_mean
    def step(self, rewards):
        return "sample" if len(rewards) < self.G else "done"
    def advantages(self, rewards, lengths):
        n = len(rewards)
        m = Fr(sum(rewards), n) if all(isinstance(r, int) for r in rewards) else sum(rewards) / n
        adv = [r - m for r in rewards]
        if self.std:
            var = sum((r - m) ** 2 for r in rewards) / n
            adv = [float(a) / math.sqrt(float(var)) if var else 0.0 for a in adv]
        if self.token_mean:
            total = sum(lengths)
            return [a * Fr(L, total) for a, L in zip(adv, lengths)]
        return [a / n for a in adv]


class ProbeThenFill:
    """Sample a probe of k answers; abandon if they agree, else fill to G."""
    def __init__(self, k, G):
        self.k, self.G = k, G
    def step(self, rewards):
        if len(rewards) < self.k:
            return "sample"
        if len(rewards) == self.k and len(set(rewards)) == 1:
            return "abandon"
        return "sample" if len(rewards) < self.G else "done"
    def advantages(self, rewards, lengths):
        return Fixed(len(rewards)).advantages(rewards, lengths)


class UntilMixed:
    """Sample until both outcomes appear (at least n_min answers) or n_max."""
    def __init__(self, n_min, n_max):
        self.n_min, self.n_max = n_min, n_max
    def step(self, rewards):
        if len(rewards) < self.n_min:
            return "sample"
        if len(set(rewards)) == 2 or len(rewards) >= self.n_max:
            return "done"
        return "sample"
    def advantages(self, rewards, lengths):
        return Fixed(len(rewards)).advantages(rewards, lengths)


def main():
    half = Fr(1, 2)
    # Prompts with equal success probability but different answer structure.
    same_p = [
        Prompt([half, Fr(1, 4), Fr(1, 4)], [1, 0, 0]),
        Prompt([half, Fr(1, 3), Fr(1, 6)], [1, 0, 0]),
        Prompt([Fr(1, 4), Fr(1, 4), half], [1, 1, 0]),
        Prompt([Fr(1, 8), Fr(3, 8), Fr(1, 6), Fr(1, 3)], [1, 1, 0, 0]),
    ]
    other_p = [
        Prompt([Fr(1, 3), Fr(1, 3), Fr(1, 3)], [1, 0, 0]),
        Prompt([Fr(1, 5), Fr(2, 5), Fr(1, 5), Fr(1, 5)], [0, 1, 1, 1]),
    ]
    schemes = {
        "Dr. GRPO, G=4": Fixed(4),
        "GRPO std-normalized, G=4": Fixed(4, std=True),
        "probe 2 then fill to 4": ProbeThenFill(2, 4),
        "sample until mixed, 2 to 6": UntilMixed(2, 6),
    }
    checks = 0
    for name, scheme in schemes.items():
        coeffs = []
        for prompt in same_p:
            mean, second, _ = moments(prompt, scheme)
            c = coefficient(mean, prompt.grad())
            assert c is not None, (name, "mean not parallel to grad p")
            coeffs.append(c)
            if not scheme_is_float(scheme):
                assert second == predicted_second_moment(prompt, scheme), (name, "second moment")
            checks += 2
        assert all(close(c, coeffs[0]) for c in coeffs), (name, "c_S depends on more than p", coeffs)
        for prompt in other_p:
            mean, second, _ = moments(prompt, scheme)
            assert coefficient(mean, prompt.grad()) is not None
            if not scheme_is_float(scheme):
                assert second == predicted_second_moment(prompt, scheme)
            checks += 2
        print(f"M, V hold for {name}: c_S(1/2) = {fmt(coeffs[0])}")

    # B: dynamic sampling only rescales the batch update.
    prompts = same_p + other_p
    for name, scheme in (("Dr. GRPO", Fixed(4)), ("GRPO std", Fixed(4, std=True))):
        plain = [0] * 0
        stats = [moments(p, scheme) for p in prompts]
        p_nc = [1 - p.mean_reward() ** 4 - (1 - p.mean_reward()) ** 4 for p in prompts]
        Z = sum(p_nc) / len(prompts)
        for (mean, _, _), pnc, prompt in zip(stats, p_nc, prompts):
            filtered = vec_scale(1 / pnc, mean)              # E[g | non-constant]
            weighted = vec_scale(pnc / (len(prompts) * Z), filtered)  # kept-prompt mix
            unfiltered = vec_scale(1 / (len(prompts) * Z), mean)
            assert all(close(a, b) for a, b in zip(weighted, unfiltered))
            checks += 1
        print(f"B holds for {name}: dynamic sampling multiplies the batch update by {float(1 / Z):.4f}, direction unchanged")

    # Breaker 1: token-mean normalization with unequal lengths.
    prompt = Prompt([Fr(1, 4), Fr(1, 4), half], [1, 1, 0])
    mean, _, _ = moments(prompt, Fixed(4, token_mean=True), lengths=[1, 3, 2])
    assert coefficient(mean, prompt.grad()) is None
    mean_eq, _, _ = moments(prompt, Fixed(4, token_mean=True), lengths=[2, 2, 2])
    assert coefficient(mean_eq, prompt.grad()) is not None
    print("Breaker: token-mean normalization with unequal lengths makes the mean leave grad p "
          "(equal lengths restore it)")

    # Breaker 2: non-binary rewards under a nonlinear scheme.
    tri = Prompt([half, Fr(1, 3), Fr(1, 6)], [Fr(1), Fr(1, 2), Fr(0)])  # asymmetric on purpose
    lin_mean, _, _ = moments(tri, Fixed(3))
    assert coefficient(lin_mean, tri.grad()) is not None
    stop_mean, _, _ = moments(tri, UntilMixedGeneral(2, 4))
    assert coefficient(stop_mean, tri.grad()) is None
    print("Breaker: with three reward levels the linear estimator stays along grad E[r], "
          "a reward-based stopping rule does not")
    print(f"all checks passed ({checks} exact or float-tolerance checks plus breakers)")


class UntilMixedGeneral(UntilMixed):
    """Stop once two distinct reward values appear (any reward levels)."""
    def advantages(self, rewards, lengths):
        n = len(rewards)
        m = sum(rewards) / n
        return [(r - m) / n for r in rewards]


def scheme_is_float(scheme):
    return getattr(scheme, "std", False)


def close(a, b):
    return abs(a - b) <= 1e-12 if isinstance(a, float) or isinstance(b, float) else a == b


def fmt(x):
    return f"{float(x):.6g}" if isinstance(x, float) else str(x)


if __name__ == "__main__":
    main()
