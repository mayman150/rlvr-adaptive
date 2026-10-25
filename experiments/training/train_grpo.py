"""On-policy GRPO: mean-centered advantages versus failures scaled by w = F1 / F0.

Each step samples P prompts and G answers per prompt, verifies them, computes
advantages, and takes one gradient step (no importance ratios, no clipping of
ratios), with the loss
    -(1 / (P G L)) sum_i A_i sum_t log pi(y_it)
and a constant normalizer L = max_new_tokens (sequence sums, no length
normalization), as the theory assumes.

Arms:
  baseline  A_i = r_i - mean(r)                               (Dr. GRPO)
  scaled    A_i = (r_i - mean(r)) * (w if r_i = 0 else 1)
where w = F1 / F0 is the ratio of mean squared full-parameter gradient norms of
correct and wrong answers, measured once on the starting model before training.
Both arms generate exactly the same number of answers.
"""

import argparse
import gzip
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verifier import last_boxed, reward  # noqa: E402
import policy_table  # noqa: E402

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
STOP_TOKENS = ("<|endoftext|>", "<|im_end|>")


def build_prompt(tok, question):
    if tok.chat_template is None:
        return f"{SYSTEM}\n\n{question}\n\n"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def stop_token_ids(tok):
    """End-of-answer token ids: the model's eos plus Qwen-style chat terminators when the vocabulary has them."""
    ids = [tok.eos_token_id] if tok.eos_token_id is not None else []
    for t in STOP_TOKENS:
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id and i not in ids:
            ids.append(i)
    return ids


class Runner:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.tok = AutoTokenizer.from_pretrained(args.model)
        self.tok.padding_side = "left"
        self.stop_ids = stop_token_ids(self.tok)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.stop_ids[0]
        self.model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(self.device)
        self.model.gradient_checkpointing_enable()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda"
                         else nullcontext())

    def trim(self, row):
        for k, t in enumerate(row.tolist()):
            if t in self.stop_ids:
                return row[: k + 1]
        return row

    @torch.no_grad()
    def generate(self, texts, n_each, max_new, seed, greedy=False):
        self.model.eval()
        items = [t for t in texts for _ in range(n_each)]
        outs = []
        torch.manual_seed(seed)
        budget = self.args.gen_batch * self.args.prefill_tokens
        s = 0
        while s < len(items):
            b = self.args.gen_batch
            enc = self.tok(items[s:s + b], return_tensors="pt", padding=True)
            while b > 1 and enc.input_ids.shape[1] * b > budget:
                b = max(1, budget // enc.input_ids.shape[1])
                enc = self.tok(items[s:s + b], return_tensors="pt", padding=True)
            enc = enc.to(self.device)
            kwargs = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
            try:
                with self.autocast:
                    gen = self.model.generate(**enc, max_new_tokens=max_new, eos_token_id=self.stop_ids,
                                              pad_token_id=self.pad_id, **kwargs)
            except torch.OutOfMemoryError:
                if b == 1:
                    raise
                del enc
                torch.cuda.empty_cache()
                budget = max(1, budget // 2)
                print(f"generation out of memory at batch {b}; retrying with a smaller batch", flush=True)
                continue
            s += b
            prompt_len = enc.input_ids.shape[1]
            for row, ids, mask in zip(gen, enc.input_ids, enc.attention_mask):
                outs.append((ids[mask.bool()], self.trim(row[prompt_len:])))
        return outs

    def logprob_sum(self, pairs, entropy=None):
        """Sum of response log-probabilities for each (prompt_ids, response_ids) pair.

        If `entropy` is a list, the summed token entropies of each response are appended to it."""
        seqs = [torch.cat([p, r]) for p, r in pairs]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, ((p, r), s) in enumerate(zip(pairs, seqs)):
            ids[i, : len(s)] = s
            mask[i, : len(s)] = 1
        with self.autocast:
            logits = self.model(input_ids=ids, attention_mask=mask).logits
        sums = []
        for i, (p, r) in enumerate(pairs):
            rows = logits[i, len(p) - 1: len(p) - 1 + len(r)].float()
            lse = torch.logsumexp(rows, dim=-1)
            sums.append((rows.gather(1, r[:, None]).squeeze(1) - lse).sum())
            if entropy is not None:
                with torch.no_grad():
                    z = rows.detach()
                    entropy.append(float((lse.detach() - (torch.softmax(z, dim=-1) * z).sum(-1)).sum()))
        return torch.stack(sums)

    def token_logprobs(self, pairs):
        """Differentiable per-token response log-probabilities for each (prompt_ids, response_ids) pair."""
        seqs = [torch.cat([p, r]) for p, r in pairs]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, ((p, r), s) in enumerate(zip(pairs, seqs)):
            ids[i, : len(s)] = s
            mask[i, : len(s)] = 1
        with self.autocast:
            logits = self.model(input_ids=ids, attention_mask=mask).logits
        out = []
        for i, (p, r) in enumerate(pairs):
            rows = logits[i, len(p) - 1: len(p) - 1 + len(r)].float()
            out.append(rows.gather(1, r[:, None]).squeeze(1) - torch.logsumexp(rows, dim=-1))
        return out

    def logprob_and_entropy(self, pairs):
        """Response log-probability sums and differentiable token-entropy sums for each pair."""
        seqs = [torch.cat([p, r]) for p, r in pairs]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, ((p, r), s) in enumerate(zip(pairs, seqs)):
            ids[i, : len(s)] = s
            mask[i, : len(s)] = 1
        with self.autocast:
            logits = self.model(input_ids=ids, attention_mask=mask).logits
        sums, ents = [], []
        for i, (p, r) in enumerate(pairs):
            rows = logits[i, len(p) - 1: len(p) - 1 + len(r)].float()
            logp = torch.log_softmax(rows, dim=-1)
            sums.append(logp.gather(1, r[:, None]).squeeze(1).sum())
            ents.append(-(logp.exp() * logp).sum())
        return torch.stack(sums), torch.stack(ents)

    def logprob_tokens(self, pairs):
        """Per-token response log-probabilities for each (prompt_ids, response_ids) pair."""
        seqs = [torch.cat([p, r]) for p, r in pairs]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, ((p, r), s) in enumerate(zip(pairs, seqs)):
            ids[i, : len(s)] = s
            mask[i, : len(s)] = 1
        with self.autocast:
            logits = self.model(input_ids=ids, attention_mask=mask).logits
        out = []
        for i, (p, r) in enumerate(pairs):
            rows = logits[i, len(p) - 1: len(p) - 1 + len(r)].float()
            out.append(rows.gather(1, r[:, None]).squeeze(1) - torch.logsumexp(rows, dim=-1))
        return out

    def grad_snapshot(self):
        return [p.grad.detach().to("cpu", copy=True) if p.grad is not None else torch.zeros(p.shape)
                for p in self.params]

    def grad_sq_norm(self):
        return float(sum((p.grad.float() ** 2).sum() for p in self.params if p.grad is not None))

    def measure_ratio(self, problems, n_prompts, group, seed):
        """Mean squared full-parameter score norm of correct and wrong answers."""
        self.model.train()
        texts = [build_prompt(self.tok, q["question"]) for q in problems[:n_prompts]]
        answers = self.generate(texts, group, self.args.max_new, seed)
        norms = {0: [], 1: []}
        owners = {0: [], 1: []}
        counts = [0] * n_prompts
        for k, (p, r) in enumerate(answers):
            q = problems[k // group]
            rew = reward(self.tok.decode(r), q["answer"])
            self.model.zero_grad(set_to_none=True)
            self.logprob_sum([(p, r)])[0].backward()
            norms[rew].append(self.grad_sq_norm())
            owners[rew].append(k // group)
            counts[k // group] += rew
        self.model.zero_grad(set_to_none=True)
        F1 = sum(norms[1]) / max(len(norms[1]), 1)
        F0 = sum(norms[0]) / max(len(norms[0]), 1)
        return {"F1": F1, "F0": F0, "n_correct": len(norms[1]), "n_wrong": len(norms[0]),
                "norms_correct": norms[1], "norms_wrong": norms[0],
                "questions_correct": owners[1], "questions_wrong": owners[0],
                "groups": [[c, group] for c in counts]}

    @torch.no_grad()
    def evaluate(self, eval_set, seed, k=1, greedy=True, samples=None):
        """Accuracy over k answers per problem, and the number correct for each problem.

        If `samples` is a list, (problem index, response text, reward) is appended for every answer."""
        texts = [build_prompt(self.tok, q["question"]) for q in eval_set]
        outs = self.generate(texts, k, self.args.max_new, seed, greedy=greedy)
        decoded = [self.tok.decode(r, skip_special_tokens=True) for _, r in outs]
        correct = [reward(self.tok.decode(r), eval_set[i // k]["answer"]) for i, (_, r) in enumerate(outs)]
        if samples is not None:
            samples.extend((i // k, t, c) for i, (t, c) in enumerate(zip(decoded, correct)))
        per_problem = [sum(correct[i * k:(i + 1) * k]) for i in range(len(eval_set))]
        return sum(correct) / len(correct), per_problem


def adaptive_collect(runner, args, arm, batch, decide, seed):
    """Sample `block` answers per active prompt per round until each prompt stops.

    decide(n0, n1) -> None to continue, or (a, b): advantages for every success
    and failure of that prompt (a = b = 0 abandons). Returns answers, advantages,
    rewards, per-prompt (successes, answers), rounds, and the number abandoned."""
    texts = [build_prompt(runner.tok, q["question"]) for q in batch]
    answers = [[] for _ in batch]
    rewards = [[] for _ in batch]
    final = [None] * len(batch)
    rnd = 0
    while any(f is None for f in final):
        active = [j for j, f in enumerate(final) if f is None]
        outs = runner.generate([texts[j] for j in active], args.block, args.max_new, seed + 7919 * rnd)
        for k, pair in enumerate(outs):
            j = active[k // args.block]
            answers[j].append(pair)
            rewards[j].append(reward(runner.tok.decode(pair[1]), batch[j]["answer"]))
        for j in active:
            n1 = sum(rewards[j])
            n0 = len(rewards[j]) - n1
            d = decide(n0, n1)
            if d is None and len(rewards[j]) >= args.cap:
                raise RuntimeError("rule did not stop at the cap")
            final[j] = d
        rnd += 1
    flat_answers, flat_adv, flat_rew = [], [], []
    for j in range(len(batch)):
        a, b = final[j]
        for pair, r in zip(answers[j], rewards[j]):
            flat_answers.append(pair)
            flat_rew.append(r)
            flat_adv.append(a if r == 1 else b)
    groups = [[sum(rs), len(rs)] for rs in rewards]
    abandoned = sum(1 for f in final if f == (0.0, 0.0))
    return flat_answers, flat_adv, flat_rew, groups, rnd, abandoned


def probe_rule(args):
    """SARA-style: abandon after the first block if all answers agree, else fill to the cap."""
    def decide(n0, n1):
        n = n0 + n1
        if n == args.block and (n0 == 0 or n1 == 0):
            return 0.0, 0.0
        if n >= args.cap:
            m = n1 / n
            return 1 - m, -m
        return None
    return decide


def table_rule(table, scale):
    def decide(n0, n1):
        d, a, b = table[(n0, n1)]
        return None if d == "continue" else (a * scale, b * scale)
    return decide


def table_scale(table):
    biggest = max(max(abs(a), abs(b)) for d, a, b in table.values() if d == "stop")
    return 1.0 / biggest if biggest > 0 else 1.0


FIXED_ARMS = ("baseline", "scaled", "psr", "nsr", "wreinforce", "shift", "passk", "psrnoise", "maxrl")


def fixed_advantage(arm, r, m, w, pos_weight, shift=0.0, n=8, k=4):
    """Advantage of one answer with reward r in a fixed group with mean reward m.

    psr rewards only correct answers, nsr penalizes only wrong ones, and wreinforce
    weights correct answers by pos_weight (Zhu et al. 2025). On-policy, psr and nsr have
    exactly the same expected update, n grad p per prompt."""
    if arm in ("psr", "psrnoise"):
        return float(r)
    if arm == "nsr":
        return float(r) - 1.0
    if arm == "wreinforce":
        return pos_weight if r else -1.0
    if arm == "shift":
        return float(r) - shift
    if arm == "maxrl":
        # Tajwar et al. (2026), Algorithm 1: (r - mean) / mean, no gradient for groups without a success.
        return (r - m) / m if m > 0 else 0.0
    if arm == "passk":
        # Unbiased pass@k weight with a leave-one-out baseline (independent of this answer):
        # expected update 2 k (1-p)^(k-1) grad p for n = 2k, in general (n/k) k (1-p)^(k-1) grad p.
        c_other = round(m * n) - int(r)
        weight = 1.0 if r else 1.0 - math.comb(n - 1 - c_other, k - 1) / math.comb(n - 1, k - 1)
        baseline = 1.0 - math.comb(n - 1 - c_other, k) / math.comb(n - 1, k)
        return weight - baseline
    a = r - m
    return a * w if (arm == "scaled" and r == 0) else a


def add_sign_noise(adv, groups, calib, args, step):
    """PSR plus c * sigma_i on every answer, sigma_i independent random signs.

    The added term has mean zero whatever c is, so the expected update stays n grad p. c is
    set each step so that the theorem's predicted second moment, sum over answers of
    E[A^2] F_r, equals NSR's on this step's reward counts: PSR gives c_q F1 per question,
    NSR (G - c_q) F0, and the sign term adds c^2 (c_q F1 + (G - c_q) F0)."""
    assert calib and calib.get("F0") and calib.get("F1"), "psrnoise needs the calibration F0 and F1"
    F0, F1 = calib["F0"], calib["F1"]
    extra = sum((n - c) * F0 - c * F1 for c, n in groups)
    base = sum(c * F1 + (n - c) * F0 for c, n in groups)
    scale = args.noise_scale * math.sqrt(max(extra, 0.0) / base) if base > 0 else 0.0
    rng = random.Random(args.seed * 7919 + step)
    return [a + scale * (1.0 if rng.random() < 0.5 else -1.0) for a in adv], scale


def pass_at_k(n, c, k):
    """Unbiased pass@k from c correct out of n samples."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


@torch.no_grad()
def probe_entropy(runner, args, answers, rewards, owner):
    """Token entropy and mean token log-probability of the first ref_block answers of
    every prompt (an on-policy sample whatever the arm), split by correctness."""
    seen, idx = {}, []
    for k, j in enumerate(owner):
        seen[j] = seen.get(j, 0) + 1
        if seen[j] <= args.ref_block:
            idx.append(k)
    out = {1: [0.0, 0.0, 0], 0: [0.0, 0.0, 0]}
    for s in range(0, len(idx), args.train_batch):
        chunk = idx[s:s + args.train_batch]
        ent = []
        lps = runner.logprob_sum([answers[k] for k in chunk], ent)
        for k, e, lp in zip(chunk, ent, lps.tolist()):
            o = out[int(rewards[k])]
            o[0] += e
            o[1] += lp
            o[2] += len(answers[k][1])
    tot_tokens = out[1][2] + out[0][2]
    res = {"entropy": (out[1][0] + out[0][0]) / tot_tokens if tot_tokens else None}
    for r, name in ((1, "correct"), (0, "wrong")):
        if out[r][2]:
            res[f"entropy_{name}"] = out[r][0] / out[r][2]
            res[f"logprob_{name}"] = out[r][1] / out[r][2]
    return res


def save_bf16(runner, path):
    bf16 = {k: v.detach().to(torch.bfloat16) for k, v in runner.model.state_dict().items()}
    runner.model.save_pretrained(path, state_dict=bf16, safe_serialization=True)
    runner.tok.save_pretrained(path)
    del bf16
    if runner.device == "cuda":
        torch.cuda.empty_cache()


def backward_weighted(runner, args, answers, idx, weights, norm, entropy=None, ent_coef=0.0, ent_stats=None):
    """Accumulate the gradient of -sum_k weights[k] log pi(answer_k) / norm into .grad.

    With ent_coef > 0 the loss also has -ent_coef * sum of token entropies / norm over the given
    answers (an entropy bonus in the same forward pass); ent_stats collects [entropy sum, tokens]."""
    for s in range(0, len(idx), args.train_batch):
        chunk = idx[s:s + args.train_batch]
        if ent_coef > 0 or ent_stats is not None:
            lp, ent = runner.logprob_and_entropy([answers[k] for k in chunk])
            if ent_stats is not None:
                ent_stats[0] += float(ent.detach().sum())
                ent_stats[1] += sum(len(answers[k][1]) for k in chunk)
        else:
            lp = runner.logprob_sum([answers[k] for k in chunk], entropy)
            ent = None
        wt = torch.tensor([weights[k] for k in chunk], device=runner.device) / norm
        loss = -(wt * lp).sum()
        if ent is not None:
            loss = loss - ent_coef * ent.sum() / norm
        loss.backward()


def bank_tensors(runner, path, size):
    """Frozen prompt-response sequences (token ids on the device) for the entropy floor."""
    p = Path(path) if Path(path).exists() else HERE / "data" / path
    items = json.loads(p.read_text())[:size]
    out = []
    for it in items:
        pi = runner.tok(build_prompt(runner.tok, it["question"]), add_special_tokens=False, return_tensors="pt").input_ids[0]
        ri = runner.tok(it["response"], add_special_tokens=False, return_tensors="pt").input_ids[0][:512]
        out.append((pi.to(runner.device), ri.to(runner.device)))
    return out


def bank_entropy(runner, bank, batch=4, grad=False):
    """Mean token entropy of the current policy on the frozen bank; with grad=True, also backward
    into .grad (the caller zeroes .grad around it)."""
    total, count = 0.0, sum(len(r) for _, r in bank)
    for s in range(0, len(bank), batch):
        chunk = bank[s:s + batch]
        if grad:
            _, ent = runner.logprob_and_entropy(chunk)
            (ent.sum() / count).backward()
            total += float(ent.detach().sum())
        else:
            with torch.no_grad():
                _, ent = runner.logprob_and_entropy(chunk)
            total += float(ent.sum())
    return total / count


@torch.no_grad()
def apply_entropy_floor(runner, opt, bank, h, h_before, floor, args):
    """After opt.step(): if the first-order prediction H + <h, u> of the bank entropy falls below the
    floor (u = Adam's actual parameter change, -lr m_hat / (sqrt(v_hat) + eps), no weight decay),
    move along h by the missing amount over |h|^2, then check the measured entropy and double the
    correction up to three times. Returns statistics."""
    group = opt.param_groups[0]
    hu = hh = 0.0
    for p, hp in zip(runner.params, h):
        st = opt.state[p]
        t = float(st["step"])
        u = -group["lr"] * (st["exp_avg"] / (1 - group["betas"][0] ** t)) / (
            (st["exp_avg_sq"] / (1 - group["betas"][1] ** t)).sqrt() + group["eps"])
        hf = hp.float()
        hu += float((hf * u).sum())
        hh += float((hf * hf).sum())
    slack = floor - h_before - hu
    tol = args.floor_tol * abs(floor)
    after = bank_entropy(runner, bank)
    stats = {"floor": floor, "h_before": h_before, "predicted": h_before + hu, "h_step": after, "c": 0.0, "tries": 0,
             "correction_norm": 0.0, "violated": False}
    if after >= floor - tol:
        return stats
    if hh <= 0:
        stats.update({"h_after": after, "violated": True})
        return stats
    # The step left the bank entropy below the floor: move along h, starting from the first-order
    # amount (at least the measured shortfall), doubling until the measured entropy is restored.
    c = max(slack, floor - after) / hh
    applied = 0.0
    for tries in range(8):
        for p, hp in zip(runner.params, h):
            p.add_(hp.to(p.dtype), alpha=c - applied)
        applied = c
        after = bank_entropy(runner, bank)
        stats.update({"c": c, "tries": tries + 1, "h_after": after, "correction_norm": c * hh ** 0.5})
        if after >= floor - tol:
            break
        c *= 2
    stats["violated"] = after < floor - tol
    return stats


def sequence_logprobs(runner, args, answers):
    with torch.no_grad():
        out = []
        for s in range(0, len(answers), args.train_batch):
            out.extend(float(x) for x in runner.logprob_sum(answers[s:s + args.train_batch]))
    return out


def unlikeliness_rewards(rewards, seq_lp, G, beta):
    """He et al. (2025): within each group with mixed rewards, rank all G samples by sequence
    probability (rank 0 = most probable) and give a correct sample 1 - beta (G - rank) / G;
    wrong samples keep 0; all-correct and all-wrong groups are unchanged."""
    out = list(map(float, rewards))
    for j in range(len(rewards) // G):
        g = rewards[j * G:(j + 1) * G]
        if 0 < sum(g) < G:
            order = sorted(range(G), key=lambda i: -seq_lp[j * G + i])
            rank = {i: r for r, i in enumerate(order)}
            for i in range(G):
                if g[i]:
                    out[j * G + i] = 1.0 - beta * (G - rank[i]) / G
    return out


def load_discovery(runner, path, problems):
    """Discovery results (list of {question, answer, c, n, response}) mapped to indices of `problems`.
    Canonicalization (fixed by Pre-registration 14, amendment 1): the retained response text is
    retokenized without special tokens and the stop token appended; a trace is dropped (its count c is
    kept) if the retokenized answer exceeds max_new - 1 tokens, or if its decoding no longer verifies or
    has no parseable final box. Every training problem must appear exactly once."""
    p = Path(path) if Path(path).exists() else HERE / "data" / path
    records = json.loads(p.read_text())
    found = {d["question"]: d for d in records}
    if len(found) != len(records):
        raise ValueError(f"{p}: {len(records) - len(found)} duplicate discovery records")
    absent = [q["question"] for q in problems if q["question"] not in found]
    if absent:
        raise ValueError(f"{p}: {len(absent)} training problems have no discovery record")
    tok = runner.tok
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    stop = im_end if isinstance(im_end, int) and im_end in runner.stop_ids else runner.stop_ids[0]
    out, dropped = {}, {"too_long": 0, "unverified": 0}
    for i, q in enumerate(problems):
        d = found[q["question"]]
        seq = None
        if d.get("response"):
            ids = tok(d["response"], add_special_tokens=False).input_ids
            text = tok.decode(ids, skip_special_tokens=True)
            if len(ids) > runner.args.max_new - 1:
                dropped["too_long"] += 1
            elif not (reward(text, q["answer"]) and last_boxed(text) is not None):
                dropped["unverified"] += 1
            else:
                seq = (torch.tensor(tok(build_prompt(tok, q["question"])).input_ids), torch.tensor(ids + [stop]))
        out[i] = {"c": int(d["c"]), "n": int(d["n"]), "seq": seq}
    kept = sum(d["seq"] is not None for d in out.values())
    print(f"discovery: {len(out)} problems, {kept} retained traces, dropped {dropped}", flush=True)
    return out


def sft_phase(runner, args, discovery, arm):
    """Supervised fine-tuning on the discovered successes (mean token log-likelihood), AdamW at
    args.sft_lr, until args.sft_tokens response tokens are processed (cycling, shuffled by seed)."""
    seqs = [d["seq"] for d in discovery.values() if d["seq"] is not None]
    if not seqs:
        raise ValueError("SFT archive has no retained successes")
    rng = random.Random(args.seed * 31337)
    opt = torch.optim.AdamW(runner.params, lr=args.sft_lr, betas=(0.9, 0.99), weight_decay=0.0)
    runner.model.train()
    seen, steps, order = 0, 0, []
    while seen < args.sft_tokens:
        if not order:
            order = list(range(len(seqs)))
            rng.shuffle(order)
        batch = [seqs[order.pop()] for _ in range(min(args.sft_batch, len(order)))]
        n_tok = sum(len(r) for _, r in batch)
        opt.zero_grad(set_to_none=True)
        for s in range(0, len(batch), args.train_batch):
            chunk = [(a.to(runner.device), b.to(runner.device)) for a, b in batch[s:s + args.train_batch]]
            (-runner.logprob_sum(chunk).sum() / n_tok).backward()
        torch.nn.utils.clip_grad_norm_(runner.params, args.clip)
        opt.step()
        seen += n_tok
        steps += 1
        if steps % 50 == 0:
            print(f"[{arm}] SFT step {steps}: {seen} response tokens", flush=True)
    opt.zero_grad(set_to_none=True)
    del opt
    if runner.device == "cuda":
        torch.cuda.empty_cache()
    print(f"[{arm}] SFT done: {steps} steps, {seen} response tokens, {len(seqs)} distinct successes", flush=True)
    return {"steps": steps, "tokens": seen, "sequences": len(seqs)}


def surrogate_flags(args):
    return args.reuse > 1 or args.length_norm or args.norm_selected


def allocated_rollouts(runner, args, step, batch, texts):
    """Two-stage allocation at a fixed total of prompts_per_step x group rollouts: alloc_first per problem,
    then the rest distributed in proportion to allocation_value (k = coverage_k) of each problem's first-stage
    counts, at most alloc_cap per problem. Returns answers and rewards grouped by problem, the group sizes,
    and statistics."""
    P, n0 = len(batch), args.alloc_first
    total = args.prompts_per_step * args.group
    first = runner.generate(texts, n0, args.max_new, args.seed * 100003 + step)
    r1 = [reward(runner.tok.decode(r), batch[k // n0]["answer"]) for k, (_, r) in enumerate(first)]
    c1 = [sum(r1[j * n0:(j + 1) * n0]) for j in range(P)]
    if args.alloc_value == "hora":
        extra = hora_allocation(c1, n0, total - P * n0, args.alloc_cap - n0)
    else:
        value = {"coverage": lambda c: allocation_value(c, n0, args.coverage_k),
                 "law": lambda c: law_value(c, n0, args.coverage_k, args.alloc_lambda),
                 "variance": lambda c: (c + 0.5) * (n0 - c + 0.5) / ((n0 + 1.0) * (n0 + 2.0))}[args.alloc_value]
        extra = allocate_extra([value(c) for c in c1], total - P * n0, args.alloc_cap - n0)
    assert sum(extra) == total - P * n0, "allocation must use the whole budget"
    items = [texts[j] for j in range(P) for _ in range(extra[j])]
    second = runner.generate(items, 1, args.max_new, args.seed * 100003 + step + 104729) if items else []
    r2 = [reward(runner.tok.decode(r), batch[j]["answer"]) for j, (_, r) in zip([j for j in range(P) for _ in range(extra[j])], second)]
    answers, rewards, sizes, k2 = [], [], [], 0
    for j in range(P):
        answers += first[j * n0:(j + 1) * n0] + second[k2:k2 + extra[j]]
        rewards += r1[j * n0:(j + 1) * n0] + r2[k2:k2 + extra[j]]
        sizes.append(n0 + extra[j])
        k2 += extra[j]
    assert len(answers) == len(rewards) == total == sum(sizes)
    stats = {"sizes": sizes, "first_counts": c1, "max": max(sizes), "min": min(sizes)}
    return answers, rewards, sizes, stats


def prefix_rollouts(runner, args, step, batch, texts, answers, rewards, keys, rescue, disc_c):
    """Replace the last `prefix_rollouts` answers of each rare question's group (base successes c <= prefix_max_c of 32,
    or no success at its latest attempt, and a stored success) with rollouts that continue from the first
    f_t of a stored success's tokens, f_t = prefix_frac (1 - step / prefix_anneal_steps); the prefix is context
    (no gradient), the reward is the verifier's on prefix + continuation. Edits answers and rewards in place."""
    G, m = args.group, args.prefix_rollouts
    f = args.prefix_frac * max(0.0, 1.0 - step / args.prefix_anneal_steps)
    runner._prefixed = set()
    if f <= 0:
        return {"frac": 0.0, "questions": 0}
    items, slots, prefixes, cuts = [], [], [], []
    for j, key in enumerate(keys):
        bank = rescue.get(key)
        if not bank or disc_c.get(key, 0) > args.prefix_max_c:
            continue
        if (args.prefix_fixed or args.prefix_match_archive) and key not in runner._archive_keys:
            continue
        seq = bank[0] if args.prefix_fixed else bank[step % len(bank)]
        cut = int(f * (len(seq[1]) - 1))
        if args.prefix_before_box:
            cut = min(cut, box_start(runner, seq[1]))
        if cut < 1:
            continue
        pre = runner.tok.decode(seq[1][:cut])
        for i in range(m):
            items.append(texts[j] + pre)
            slots.append(j * G + G - 1 - i)
            prefixes.append(pre)
            cuts.append(cut)
    if not items:
        return {"frac": f, "questions": 0}
    # The prefix counts against the answer budget: each continuation is capped at max_new - its prefix length
    # (generated to the largest cap, then cut, which is the same as stopping at its own cap).
    outs = runner.generate(items, 1, args.max_new - min(cuts), args.seed * 100003 + step + 7919)
    solved, capped, tokens = 0, 0, 0
    prefix_only = sum(reward(pre, batch[slot // G]["answer"]) for slot, pre in zip(slots, prefixes))
    for (pi, ri), slot, pre, cut in zip(outs, slots, prefixes, cuts):
        cap = args.max_new - cut
        if len(ri) >= cap:
            ri = ri[:cap]
            capped += int(int(ri[-1]) not in runner.stop_ids)
        r = reward(pre + runner.tok.decode(ri), batch[slot // G]["answer"])
        answers[slot], rewards[slot] = (pi, ri), r
        runner._prefixed.add(slot)
        solved += r
        tokens += len(pi) + len(ri)
    return {"frac": f, "questions": len(items) // m, "rollouts": len(items), "solved": solved, "capped": capped, "tokens": tokens,
            "prefix_tokens": sum(cuts), "prefix_only_correct": prefix_only}


def box_start(runner, ids):
    """Number of leading response tokens before the first final-answer box (all of them if there is none), so that
    a prefix never supplies the boxed answer. Cached per response."""
    cache = runner.__dict__.setdefault("_box_start", {})
    key = tuple(ids.tolist())
    if key not in cache:
        lo, hi = 0, len(ids)  # largest k with no "\\boxed" in the decoding of ids[:k]
        if "\\boxed" not in runner.tok.decode(ids):
            cache[key] = len(ids)
        else:
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if "\\boxed" in runner.tok.decode(ids[:mid]):
                    hi = mid - 1
                else:
                    lo = mid
            cache[key] = lo
    return cache[key]


def bbg_utility(m, n, k):
    """Yuan et al. (2026), Bayesian boundary gating: E[k (1 - p)^(k-1)] under p ~ Beta(m + 1, n - m + 1)."""
    out = float(k)
    for j in range(1, k):
        out *= (n - m + j) / (n + 1 + j)
    return out


def bbg_coefficients(groups, k, tau):
    """Per-problem coefficients alpha(m) = u(m; n, k) if u >= tau (n + 1), else 0, rescaled to mean 1 over
    the step's problems (so the step size matches the ungated rule when every problem is active)."""
    alpha = []
    for c, n in groups:
        u = bbg_utility(c, n, k)
        alpha.append(u if u >= tau * (n + 1) else 0.0)
    total = sum(alpha)
    return [a * len(alpha) / total if total > 0 else 0.0 for a in alpha]


def tar_bank(runner, args, discovery):
    """TAR's memory bank (Li et al. 2026) from the discovery archive: the retained base successes with
    their per-token log-probabilities under the starting (base) model."""
    seqs = [d["seq"] for d in discovery.values() if d["seq"] is not None]
    if not seqs:
        raise ValueError("TAR bank has no retained successes")
    base = []
    with torch.no_grad():
        for s in range(0, len(seqs), args.train_batch):
            chunk = [(a.to(runner.device), b.to(runner.device)) for a, b in seqs[s:s + args.train_batch]]
            base.extend(t.detach().cpu() for t in runner.token_logprobs(chunk))
    print(f"TAR bank: {len(seqs)} base successes", flush=True)
    return seqs, base


def tar_backward(runner, args, bank, rng):
    """Gradient of lambda * L_rep (Li et al. 2026, Eq. 8): over a uniform replay batch from the bank,
    sg[g] * sum_t sg[w_t] (-log pi_t) / (eps + sum_t sg[w_t]), averaged over trajectories, with
    delta_t = log pi_base - log pi, g = phi(mean_t delta_t), w_t = phi(delta_t), phi(x) = max(1 - e^-x, 0)."""
    seqs, base = bank
    pick = rng.sample(range(len(seqs)), min(args.tar_batch, len(seqs)))
    gates, active, tokens = [], 0, 0
    for s in range(0, len(pick), args.train_batch):
        idx = pick[s:s + args.train_batch]
        chunk = [(seqs[i][0].to(runner.device), seqs[i][1].to(runner.device)) for i in idx]
        loss = 0.0
        for i, lp in zip(idx, runner.token_logprobs(chunk)):
            delta = base[i].to(runner.device) - lp.detach()
            g = max(1.0 - math.exp(-float(delta.mean())), 0.0)
            wt = torch.clamp(1.0 - torch.exp(-delta), min=0.0)
            gates.append(g)
            if g > 0:
                active += 1
                tokens += int((wt > 0).sum())
                loss = loss + g * (wt * (-lp)).sum() / (1e-8 + wt.sum())
        if torch.is_tensor(loss):
            (args.tar_lambda * loss / len(pick)).backward()
    return {"replayed": len(pick), "active": active, "mean_gate": sum(gates) / len(gates), "weighted_tokens": tokens}


def allocation_value(c, n, k):
    """Expected frontier gain per extra rollout on a problem with c successes of n:
    posterior mean of k p (1 - p)^k under p ~ Beta(c + 1/2, n - c + 1/2) (the pass@k slope
    k (1 - p)^(k-1) times the per-rollout signal p (1 - p))."""
    a, b = c + 0.5, n - c + 0.5
    lbeta = lambda x, y: math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)
    return k * math.exp(lbeta(a + 1, b + k) - lbeta(a, b))


def law_value(c, n, k=16, lam=0.37):
    """Frontier value of one extra rollout under the frontier law: E[p^2 (1-p)^2 (k (1-p)^(k-1) - lam)] with
    p ~ Beta(c + 1/2, n - c + 1/2) (per-rollout log-odds step p (1-p) times the matched-pass@1 frontier slope
    d pass@k / d theta - lam d pass@1 / d theta), clipped at 0; lam = Dr. GRPO's observed frontier slope."""
    a, b = c + 0.5, n - c + 0.5
    lb = lambda x, y: math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)
    v = k * math.exp(lb(a + 2, b + k + 1) - lb(a, b)) - lam * math.exp(lb(a + 2, b + 2) - lb(a, b))
    return max(v, 0.0)


def hora_allocation(counts, n0, extra, cap_each):
    """HORA (2605.07114): greedy on the marginal posterior hit gain M_i(l) = B(a_i + 1, b_i + l) / B(a_i, b_i),
    Beta(1, 1) prior, a_i = 1 + c_i, b_i = 1 + n0 - c_i; at most cap_each extras per problem."""
    a = [1.0 + c for c in counts]
    b = [1.0 + n0 - c for c in counts]
    m = [ai / (ai + bi) for ai, bi in zip(a, b)]
    out = [0] * len(counts)
    for _ in range(extra):
        i = max((j for j in range(len(counts)) if out[j] < cap_each), key=lambda j: m[j], default=None)
        if i is None:
            break
        out[i] += 1
        m[i] *= (b[i] + out[i] - 1) / (a[i] + b[i] + out[i])
    return out


def allocate_extra(values, extra, cap_each):
    """Integer allocation of `extra` rollouts proportional to `values`, at most cap_each per problem
    (largest remainders; overflow redistributed). Ties and all-zero values fall back to uniform."""
    m = len(values)
    out = [0] * m
    left, active = extra, list(range(m))
    while left > 0 and active:
        tot = sum(values[i] for i in active)
        share = {i: (values[i] / tot if tot > 0 else 1.0 / len(active)) * left for i in active}
        base = {i: min(int(share[i]), cap_each - out[i]) for i in active}
        for i in active:
            out[i] += base[i]
        left -= sum(base.values())
        order = sorted(active, key=lambda i: -(share[i] - int(share[i])))
        for i in order:
            if left == 0:
                break
            if out[i] < cap_each:
                out[i] += 1
                left -= 1
        active = [i for i in active if out[i] < cap_each]
    return out


def coverage_weight(c, n, k):
    """Posterior mean of d pass@k / d log p = k p (1 - p)^(k-1) under p ~ Beta(c + 1/2, n - c + 1/2)
    (c of n correct at the question's latest attempt): k B(a + 1, b + k - 1) / B(a, b)."""
    a, b = c + 0.5, n - c + 0.5
    lbeta = lambda x, y: math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)
    return k * math.exp(lbeta(a + 1, b + k - 1) - lbeta(a, b))


def memory_backward(runner, args, archive, rng, norm, phat=None):
    """Gradient of -lambda * sum over sampled archived questions of log sum_{y in S_q} pi(y | q), / norm.

    Questions are drawn uniformly, or (memory_weighting = coverage) with probability proportional
    to coverage_weight(p_hat_q, k), p_hat_q the question's smoothed success rate when it was
    attempted, without replacement."""
    keys = [k for k, v in archive.items() if v]
    if not keys:
        return {"questions": 0, "sequences": 0}
    m = min(args.memory_questions, len(keys))
    if args.memory_weighting == "coverage":
        w = [coverage_weight(*phat[k], args.coverage_k) for k in keys]
        # successive weighted sampling without replacement: exponential keys -log(u) / w, smallest first
        order = sorted(range(len(keys)), key=lambda i: -math.log(max(rng.random(), 1e-300)) / max(w[i], 1e-300))
        pick = [keys[i] for i in order[:m]]
    else:
        pick = rng.sample(keys, m)
    seqs, owner = [], []
    for j, k in enumerate(pick):
        for pr in archive[k]:
            seqs.append((pr[0].to(runner.device), pr[1].to(runner.device)))
            owner.append(j)
    values, tokens = [], 0
    for j in range(len(pick)):
        mine = [sq for sq, o in zip(seqs, owner) if o == j]
        lm = torch.logsumexp(runner.logprob_sum(mine), dim=0)  # one question at a time bounds memory
        (-args.memory_lambda * lm / norm).backward()
        values.append(float(lm.detach()))
        tokens += sum(len(p) + len(r) for p, r in mine)
    out = {"questions": len(pick), "sequences": len(seqs), "tokens": tokens, "mean_log_mass": sum(values) / len(values)}
    if phat is not None:
        out["mean_p_hat"] = sum((phat[k][0] + 0.5) / (phat[k][1] + 1) for k in pick) / len(pick)
    return out


def backward_surrogate(runner, args, answers, idx, adv, old_lp, norm):
    """PPO-style clipped surrogate over tokens; with old_lp from the current policy (ratio 1)
    and no length normalization this equals the sequence-sum REINFORCE loss."""
    for s in range(0, len(idx), args.train_batch):
        chunk = idx[s:s + args.train_batch]
        new = runner.logprob_tokens([answers[k] for k in chunk])
        total = 0.0
        for k, lp in zip(chunk, new):
            ratio = torch.exp(lp - old_lp[k])
            a = adv[k]
            surr = torch.minimum(ratio * a, torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio) * a)
            term = surr.mean() * args.max_new if args.length_norm else surr.sum()
            total = total + term
        (-total / norm).backward()


@torch.no_grad()
def old_logprobs(runner, args, answers):
    out = []
    for s in range(0, len(answers), args.train_batch):
        out.extend(t.detach() for t in runner.logprob_tokens(answers[s:s + args.train_batch]))
    return out


def adam_scale(opt, p):
    """1 / (sqrt(v_hat) + eps) from Adam's state before this step's update (None before the first)."""
    st = opt.state.get(p, {}) if opt is not None else {}
    if st.get("exp_avg_sq") is None:
        return None
    group = opt.param_groups[0]
    v_hat = st["exp_avg_sq"].flatten() / (1 - group["betas"][1] ** float(st["step"]))
    return 1.0 / (v_hat.sqrt() + group["eps"])


@torch.no_grad()
def store_update(runner, opt):
    """Adam's update direction m_hat / (sqrt(v_hat) + eps) after this step (the parameter change
    is -lr times it; no weight decay), kept in bf16 on the device for the next step's diagnostics."""
    runner.last_update = None  # free the previous buffer first (peak memory)
    group, out = opt.param_groups[0], []
    for p in runner.params:
        st = opt.state[p]
        t = float(st["step"])
        m_hat = st["exp_avg"].flatten() / (1 - group["betas"][0] ** t)
        v_hat = st["exp_avg_sq"].flatten() / (1 - group["betas"][1] ** t)
        out.append((m_hat / (v_hat.sqrt() + group["eps"])).to(torch.bfloat16))
    runner.last_update = out


def shadow_stats(runner, args, answers, adv, owner, half_a, norm, opt):
    """Split-half statistics of another estimator on the same answers at the same policy;
    leaves .grad empty."""
    runner.model.zero_grad(set_to_none=True)
    live_a = [k for k, a in enumerate(adv) if a != 0 and owner[k] in half_a]
    live_b = [k for k, a in enumerate(adv) if a != 0 and owner[k] not in half_a]
    backward_weighted(runner, args, answers, live_a, adv, norm)
    ga = runner.grad_snapshot()
    backward_weighted(runner, args, answers, live_b, adv, norm)
    keys = ("aa", "bb", "ab", "aa_pre", "bb_pre", "ab_pre")
    st = dict.fromkeys(keys, 0.0)
    pre = all(adam_scale(opt, p) is not None for p in runner.params[:1])
    with torch.no_grad():
        for p, a_cpu in zip(runner.params, ga):
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            a = a_cpu.to(runner.device).flatten()
            b = g.flatten() - a
            st["aa"] += float(torch.dot(a, a))
            st["bb"] += float(torch.dot(b, b))
            st["ab"] += float(torch.dot(a, b))
            d = adam_scale(opt, p) if pre else None
            if d is not None:
                ad, bd = a * d, b * d
                st["aa_pre"] += float(torch.dot(ad, ad))
                st["bb_pre"] += float(torch.dot(bd, bd))
                st["ab_pre"] += float(torch.dot(ad, bd))
    del ga
    runner.model.zero_grad(set_to_none=True)
    return st if pre else {k: st[k] for k in ("aa", "bb", "ab")}


def diagnostic_step(runner, args, answers, adv, rewards, owner, norm, step, entropy, opt=None):
    """Backward pass with split-half gradient statistics.

    Prompts are split at random into halves A and B. g_A and g_B are the arm's gradient
    from each half; ref_B is the Dr. GRPO estimator on the first `ref_block` answers of
    each prompt in B, an unbiased direction for the population gradient that is
    independent of g_A. Returns statistics; .grad holds g_A + g_B, as without diagnostics."""
    prompts = sorted(set(owner))
    perm = prompts[:]
    random.Random(args.seed * 1000003 + step).shuffle(perm)
    half_a = set(perm[: len(perm) // 2])
    first = {}
    for k, j in enumerate(owner):
        first.setdefault(j, [])
        if len(first[j]) < args.ref_block:
            first[j].append(k)
    ref_w = {}
    for j in prompts:
        if j in half_a:
            continue
        m = sum(rewards[k] for k in first[j]) / len(first[j])
        for k in first[j]:
            if rewards[k] != m:
                ref_w[k] = rewards[k] - m
    runner.model.zero_grad(set_to_none=True)
    backward_weighted(runner, args, answers, sorted(ref_w), ref_w, norm)
    ref = runner.grad_snapshot()
    ref2 = None
    if args.within_ref:
        # Second reference from answers ref_block+1 .. 2 ref_block of the same prompts, with its
        # own mean: independent of ref given the prompts, so <ref, ref2> measures the
        # per-prompt signal |grad p_x|^2 and not only the shared part.
        second = {}
        for k, j in enumerate(owner):
            if j not in half_a:
                second.setdefault(j, [])
                if k not in first[j] and len(second[j]) < args.ref_block:
                    second[j].append(k)
        ref2_w = {}
        for j, ks in second.items():
            if len(ks) < args.ref_block:
                continue
            m = sum(rewards[k] for k in ks) / len(ks)
            for k in ks:
                if rewards[k] != m:
                    ref2_w[k] = rewards[k] - m
        runner.model.zero_grad(set_to_none=True)
        backward_weighted(runner, args, answers, sorted(ref2_w), ref2_w, norm)
        ref2 = runner.grad_snapshot()
    shadow = None
    if args.shadow_arm and step < args.shadow_until:
        # The shadow estimator (e.g. NSR) on the same answers and halves at this policy.
        means = {}
        for k, j in enumerate(owner):
            means.setdefault(j, []).append(rewards[k])
        sh_adv = [fixed_advantage(args.shadow_arm, rewards[k], sum(means[j]) / len(means[j]), 1.0,
                                  args.pos_weight, args.shift, len(means[j]), args.train_k)
                  for k, j in enumerate(owner)]
        shadow = shadow_stats(runner, args, answers, sh_adv, owner, half_a, norm, opt)
    runner.model.zero_grad(set_to_none=True)
    live_a = [k for k, a in enumerate(adv) if a != 0 and owner[k] in half_a]
    live_b = [k for k, a in enumerate(adv) if a != 0 and owner[k] not in half_a]
    backward_weighted(runner, args, answers, live_a, adv, norm, entropy)
    ga = runner.grad_snapshot()
    backward_weighted(runner, args, answers, live_b, adv, norm, entropy)
    stats = dict.fromkeys(("aa", "bb", "ab", "ar", "br", "rr"), 0.0)
    if ref2 is not None:
        stats.update(dict.fromkeys(("r1r2", "r2r2", "ar2"), 0.0))
    # The gradient divided by Adam's per-coordinate scale from before this step (D_{t-1} g_t;
    # not Adam's update, which also averages past gradients): the same statistics.
    precond = opt is not None and adam_scale(opt, runner.params[0]) is not None
    if precond:
        stats.update(dict.fromkeys(("aa_pre", "bb_pre", "ab_pre"), 0.0))
    # Lagged alignment of the previous step's actual Adam update direction with the population
    # gradient: <u_{t-1}, ref_t>, ref_t from this step's questions (fresh given the history).
    last = getattr(runner, "last_update", None)
    if last is not None:
        stats.update(dict.fromkeys(("u_ref", "uu"), 0.0))
    with torch.no_grad():
        for i, (p, a_cpu, r_cpu) in enumerate(zip(runner.params, ga, ref)):
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            a = a_cpu.to(runner.device).flatten()
            r = r_cpu.to(runner.device).flatten()
            b = g.flatten() - a
            stats["aa"] += float(torch.dot(a, a))
            stats["bb"] += float(torch.dot(b, b))
            stats["ab"] += float(torch.dot(a, b))
            stats["ar"] += float(torch.dot(a, r))
            stats["br"] += float(torch.dot(b, r))
            stats["rr"] += float(torch.dot(r, r))
            if last is not None:
                u = last[i].float()
                stats["u_ref"] += float(torch.dot(u, r))
                stats["uu"] += float(torch.dot(u, u))
            if precond:
                d = adam_scale(opt, p)
                ad, bd = a * d, b * d
                stats["aa_pre"] += float(torch.dot(ad, ad))
                stats["bb_pre"] += float(torch.dot(bd, bd))
                stats["ab_pre"] += float(torch.dot(ad, bd))
            if ref2 is not None:
                r2 = ref2[i].to(runner.device).flatten()
                stats["r1r2"] += float(torch.dot(r, r2))
                stats["r2r2"] += float(torch.dot(r2, r2))
                stats["ar2"] += float(torch.dot(a, r2))
    stats.update({"prompts_a": len(half_a), "prompts_b": len(prompts) - len(half_a),
                  "answers_a": sum(1 for j in owner if j in half_a),
                  "answers_b": sum(1 for j in owner if j not in half_a),
                  "ref_answers": len(ref_w), "norm": norm})
    if shadow is not None:
        stats["shadow"] = shadow
    return stats, len(live_a) + len(live_b)


def train_arm(args, arm, w, problems, eval_set, out_dir, calib=None, final_set=None, calib_problems=None):
    runner = Runner(args)
    opt = torch.optim.AdamW(runner.params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    rng = random.Random(args.seed)
    order = list(range(len(problems)))
    rng.shuffle(order)
    log, evals, sampled = [], [], []
    norm = args.budget * args.max_new
    calib_groups = (calib or {}).get("groups") or [[1, 8]]
    ratio = (calib["F0"] / calib["F1"]) if calib and calib.get("F1") else 1.0 / w
    spent, prompts_done, pointer, mean_n, recent = 0, 0, 0, float(args.group), []
    eval_steps = ({int(s) for s in args.eval_steps.split(",")} if args.eval_steps
                  else set(range(0, args.steps + 1, args.eval_every)))
    sample_steps = {int(s) for s in args.sample_eval_steps.split(",") if s}
    passk_steps = {int(s) for s in args.passk_steps.split(",") if s}
    save_steps = {int(s) for s in args.save_steps.split(",") if s}
    remeasure_steps = {int(s) for s in args.remeasure_steps.split(",") if s}
    # Pre-registration 12 options (all off by default).
    bank = bank_tensors(runner, args.floor_bank, args.floor_bank_size) if args.floor_rho > 0 else None
    floor = args.floor_rho * bank_entropy(runner, bank) if bank is not None else None
    archive, phat, mem_rng = {}, {}, random.Random(args.seed * 7717)
    base_trace = set()  # archive keys that still hold a base-model trace (retired at the first own success)
    sft_stats = None
    rescue, disc_c = None, {}
    if args.rescue_archive:
        assert args.reuse == 1 and not surrogate_flags(args) and not args.diagnostics, "rescue options need the plain on-policy loss"
        rescue = {}
        for i, d in load_discovery(runner, args.rescue_archive, problems).items():
            disc_c[i] = d["c"]
            if d["seq"] is not None:
                rescue[i] = [d["seq"]]
        if args.prefix_match_archive:
            other = load_discovery(runner, args.prefix_match_archive, problems)
            rescue = {i: v for i, v in rescue.items() if other[i]["seq"] is not None}
            del other
        runner._archive_keys = set(rescue)  # the stored solutions (bank[0]); own successes are appended later
        print(f"[{arm}] rescue bank: {len(rescue)} questions with a stored success", flush=True)
    assert not ((args.complete_groups or args.prefix_rollouts > 0) and rescue is None), "group completion and prefix rollouts need --rescue-archive"
    assert args.prefix_rollouts <= args.group and args.prefix_anneal_steps > 0, "prefix rollouts must fit in a group"
    assert not (args.alloc_first > 0 and (args.reuse > 1 or surrogate_flags(args) or args.diagnostics or args.prefix_rollouts or args.complete_groups
                                          or args.unlikely_beta > 0 or args.bbg_k > 0 or arm == "psrnoise")), "allocation is implemented for the plain on-policy loss"
    assert args.alloc_first == 0 or args.alloc_first <= args.group <= args.alloc_cap, "allocation needs alloc_first <= group <= alloc_cap"
    tar = None
    if args.tar_archive:
        assert arm == "baseline" and args.memory_lambda == 0 and not (args.seed_archive or args.sft_archive), \
            "TAR replay is run on Dr. GRPO without other replay"
        tar = tar_bank(runner, args, load_discovery(runner, args.tar_archive, problems))
    if args.seed_archive or args.sft_archive:
        discovery = load_discovery(runner, args.seed_archive or args.sft_archive, problems)
        print(f"[{arm}] discovery: {len(discovery)} questions, {sum(d['seq'] is not None for d in discovery.values())} with a success", flush=True)
        if args.seed_archive:
            for i, d in discovery.items():
                phat[i] = (d["c"], d["n"])
                if d["seq"] is not None:
                    archive[i] = [d["seq"]]
                    base_trace.add(i)
        if args.sft_archive:
            sft_stats = sft_phase(runner, args, discovery, arm)
            opt = torch.optim.AdamW(runner.params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)  # reset for RL
        del discovery
    early_ref = []
    ent_coef, ent_target = args.entropy_coef, None
    adaptive_entropy = args.entropy_target_frac > 0
    if floor is not None:
        print(f"[{arm}] entropy floor {floor:.4f} (rho {args.floor_rho} of the starting bank entropy)", flush=True)
    assert not remeasure_steps or calib_problems, "re-measuring needs the calibration questions"
    remeasured = []
    surrogate = args.reuse > 1 or args.length_norm or args.norm_selected
    assert not (surrogate and args.diagnostics), "diagnostics need the plain on-policy loss"
    assert not surrogate or arm in FIXED_ARMS, "verl-like options are implemented for fixed groups only"
    interventions = (args.entropy_coef > 0 or args.entropy_target_frac > 0 or args.floor_rho > 0 or args.memory_lambda > 0
                     or args.unlikely_beta > 0 or args.early_floor_rho > 0 or bool(args.seed_archive) or bool(args.sft_archive)
                     or bool(args.tar_archive))
    assert not (args.bbg_k > 0 and (surrogate or args.diagnostics or arm not in FIXED_ARMS)), \
        "Bayesian boundary gating is implemented for the plain on-policy loss of fixed groups"
    assert not (args.floor_rho > 0 and args.early_floor_rho > 0), "choose one entropy floor"
    assert not (interventions and (surrogate or args.diagnostics)), "Pre-registration 12 options need the plain on-policy loss without diagnostics"
    assert not (interventions and arm != "baseline"), "Pre-registration 12 options are defined for Dr. GRPO (arm baseline)"
    passk = []
    for step in range(args.steps + 1):
        if step in remeasure_steps:
            # Same questions and sampling seed as the calibration before training (paired).
            m = runner.measure_ratio(calib_problems, len(calib_problems), args.group, args.seed + 7)
            remeasured.append({"step": step, **m})
            print(f"[{arm}] step {step} re-measured F1 {m['F1']:.4g}, F0 {m['F0']:.4g} "
                  f"({m['n_correct']} correct, {m['n_wrong']} wrong)", flush=True)
            (out_dir / f"{arm}_remeasure.json").write_text(json.dumps(remeasured))
        if step in eval_steps:
            acc, per = runner.evaluate(eval_set, args.seed)
            evals.append({"step": step, "accuracy": acc, "correct": per})
            print(f"[{arm}] step {step} eval accuracy {acc:.4f}", flush=True)
            (out_dir / f"{arm}_eval.json").write_text(json.dumps(evals))
        if step in sample_steps:
            acc, per = runner.evaluate(eval_set, args.seed + 17 * step, k=args.sample_k, greedy=False)
            sampled.append({"step": step, "accuracy": acc, "k": args.sample_k, "correct": per})
            print(f"[{arm}] step {step} sampled accuracy {acc:.4f} (k={args.sample_k})", flush=True)
            (out_dir / f"{arm}_sampled_eval.json").write_text(json.dumps(sampled))
        if step in passk_steps:
            samples = [] if args.save_passk_texts else None
            _, per = runner.evaluate(eval_set[: args.passk_limit], args.seed + 31 * step, k=args.passk_n,
                                     greedy=False, samples=samples)
            if samples is not None:
                subset = eval_set[: args.passk_limit]
                with gzip.open(out_dir / f"{arm}_passk_texts_step{step}.json.gz", "wt") as fh:
                    json.dump({"step": step, "n": args.passk_n,
                               "problems": [{"question": q["question"], "answer": q["answer"]} for q in subset],
                               "samples": [{"problem": i, "text": t, "reward": c} for i, t, c in samples]}, fh)
            curve = {k: sum(pass_at_k(args.passk_n, c, k) for c in per) / len(per)
                     for k in (1, 2, 4, 8, 16, 32, 64) if k <= args.passk_n}
            passk.append({"step": step, "n": args.passk_n, "counts": per, "pass_at_k": curve})
            print(f"[{arm}] step {step} pass@k " + " ".join(f"{k}:{v:.3f}" for k, v in curve.items()), flush=True)
            (out_dir / f"{arm}_passk.json").write_text(json.dumps(passk))
        if step in save_steps and step != args.steps:
            save_bf16(runner, out_dir / f"ckpt_step{step}")
            print(f"[{arm}] saved checkpoint at step {step}", flush=True)
        if step == args.steps:
            if args.save_steps and step in save_steps:
                save_bf16(runner, out_dir / f"ckpt_step{step}")
            if args.save_model:
                path = Path(args.save_model.replace("{arm}", arm))
                bf16 = {k: v.detach().to(torch.bfloat16) for k, v in runner.model.state_dict().items()}
                runner.model.save_pretrained(path, state_dict=bf16, safe_serialization=True)
                runner.tok.save_pretrained(path)
                print(f"[{arm}] saved model to {path}", flush=True)
                del bf16
                if runner.device == "cuda":
                    torch.cuda.empty_cache()
            if final_set:
                acc, per = runner.evaluate(final_set, args.seed)
                print(f"[{arm}] final held-out accuracy {acc:.4f} over {len(final_set)} problems", flush=True)
                (out_dir / f"{arm}_final_eval.json").write_text(
                    json.dumps({"step": step, "accuracy": acc, "n": len(final_set), "correct": per}))
            break
        if runner.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        extra = {}
        if arm in FIXED_ARMS:
            pps, G = args.prompts_per_step, args.group
            if step % args.reuse == 0:
                n_pool = pps * args.reuse
                batch = [problems[order[(step * pps + j) % len(order)]] for j in range(n_pool)]
                texts = [build_prompt(runner.tok, q["question"]) for q in batch]
                if args.alloc_first > 0:
                    pool_answers, pool_rewards, alloc_n, extra["alloc"] = allocated_rollouts(runner, args, step, batch, texts)
                else:
                    pool_answers = runner.generate(texts, G, args.max_new, args.seed * 100003 + step)
                    pool_rewards = [reward(runner.tok.decode(r), batch[k // G]["answer"]) for k, (_, r) in enumerate(pool_answers)]
                if args.prefix_rollouts > 0:
                    extra["prefix"] = prefix_rollouts(runner, args, step, batch, texts, pool_answers, pool_rewards,
                                                      [order[(step * pps + j) % len(order)] for j in range(n_pool)], rescue, disc_c)
                pool_old = old_logprobs(runner, args, pool_answers) if surrogate else None
            j0 = (step % args.reuse) * pps * G
            answers = pool_answers[j0:j0 + pps * G]
            rewards = pool_rewards[j0:j0 + pps * G]
            old_lp = pool_old[j0:j0 + pps * G] if surrogate else None
            adv = []
            if args.alloc_first > 0:
                # variable group sizes: Dr. GRPO advantages within each problem's own group
                groups, k0 = [], 0
                for n_j in alloc_n:
                    g = rewards[k0:k0 + n_j]
                    m = sum(g) / len(g)
                    scale = G / n_j if args.alloc_norm == "group" else 1.0  # HORA: each problem keeps equal weight
                    adv.extend(scale * fixed_advantage(arm, r, m, w, args.pos_weight, args.shift, n_j, args.train_k) for r in g)
                    groups.append([sum(g), n_j])
                    k0 += n_j
            else:
                for j in range(pps):
                    g = rewards[j * G:(j + 1) * G]
                    m = sum(g) / len(g)
                    for r in g:
                        adv.append(fixed_advantage(arm, r, m, w, args.pos_weight, args.shift, G, args.train_k))
                groups = [[sum(rewards[j * G:(j + 1) * G]), G] for j in range(pps)]
            extra["stale"] = step % args.reuse
            keys = [order[(step * pps + j) % len(order)] for j in range(pps)] if args.reuse == 1 else None
            if args.unlikely_beta > 0:
                assert arm == "baseline" and not surrogate, "the unlikeliness reward is implemented for on-policy Dr. GRPO"
                shaped = unlikeliness_rewards(rewards, sequence_logprobs(runner, args, answers), G, args.unlikely_beta)
                adv = []
                for j in range(pps):
                    g = shaped[j * G:(j + 1) * G]
                    m = sum(g) / len(g)
                    adv.extend(r - m for r in g)
            if arm == "psrnoise":
                adv, extra["noise_c"] = add_sign_noise(adv, groups, calib, args, step)
            if args.bbg_k > 0:
                coef = bbg_coefficients(groups, args.bbg_k, args.bbg_tau)
                adv = [a * coef[i // G] for i, a in enumerate(adv)]
                extra["bbg_active"] = sum(c > 0 for c in coef)
            injected = []
            if args.complete_groups:
                # Zero-success groups whose question has a stored success: append it as a correct answer
                # (G + 1 answers, mean 1 / (G + 1)) and recompute the group's advantages with the arm's rule.
                m = 1.0 / (G + 1)
                for j in range(pps):
                    if groups[j][0] == 0 and rescue.get(keys[j]):
                        seq = rescue[keys[j]][step % len(rescue[keys[j]])]
                        for i in range(j * G, (j + 1) * G):
                            adv[i] = fixed_advantage(arm, 0, m, w, args.pos_weight, args.shift, G + 1, args.train_k)
                        injected.append(((seq[0].to(runner.device), seq[1].to(runner.device)),
                                         fixed_advantage(arm, 1, m, w, args.pos_weight, args.shift, G + 1, args.train_k)))
                extra["completed_groups"] = len(injected)
                extra["completion_tokens"] = sum(len(a[0]) + len(a[1]) for a, _ in injected)
            if args.prefix_rollouts > 0:
                plain = [r for k, r in enumerate(rewards) if k not in getattr(runner, "_prefixed", set())]
                extra["mean_reward_unassisted"] = sum(plain) / max(len(plain), 1)
        else:
            if arm == "optimal" and step % args.resolve_every == 0:
                prior = policy_table.fit_beta(recent if recent else calib_groups)
                _, table = policy_table.solve_table(prior[0], prior[1], ratio, args.table_r, args.cap, args.block)
                scale = table_scale(table)
                decide = table_rule(table, scale)
                extra["prior"] = prior
            elif arm == "probe" and step == 0:
                decide = probe_rule(args)
            n_prompts = min(64, max(2, round((args.budget * (step + 1) - spent) / mean_n)))
            batch = [problems[order[(pointer + j) % len(order)]] for j in range(n_prompts)]
            pointer += n_prompts
            answers, adv, rewards, groups, rounds, abandoned = adaptive_collect(
                runner, args, arm, batch, decide, args.seed * 100003 + step)
            spent += len(answers)
            prompts_done += n_prompts
            mean_n = spent / prompts_done
            recent = (recent + groups)[-200:]
            extra.update({"prompts": n_prompts, "answers": len(answers), "rounds": rounds,
                          "cumulative_answers": spent, "abandoned": abandoned})
        runner.model.train()
        opt.zero_grad(set_to_none=True)
        h_grad = None
        owner = [j for j, (_, n) in enumerate(groups) for _ in range(n)]
        entropy = []
        if surrogate:
            live = [k for k, a in enumerate(adv) if a != 0]
            step_norm = (len(live) * args.max_new if args.norm_selected else norm)
            if live:
                backward_surrogate(runner, args, answers, live, adv, old_lp, step_norm)
            n_live = len(live)
        elif args.diagnostics:
            stats, n_live = diagnostic_step(runner, args, answers, adv, rewards, owner, norm, step, entropy, opt)
            extra["split"] = stats
        else:
            h_grad = None
            if args.early_floor_rho > 0:
                # Bank = the first K response tokens of this step's own answers (fixed within the step).
                bank = [(pi, ri[: args.early_floor_tokens]) for pi, ri in answers if len(ri)]
                if floor is None:
                    # Reference: mean early-token entropy over the first `early_floor_warmup` steps
                    # (no enforcement during them), since one step's questions make it noisy.
                    early_ref.append(bank_entropy(runner, bank))
                    extra["early_entropy"] = early_ref[-1]
                    if len(early_ref) == args.early_floor_warmup:
                        floor = args.early_floor_rho * sum(early_ref) / len(early_ref)
                        print(f"[{arm}] early-token entropy floor {floor:.4f} (rho {args.early_floor_rho})", flush=True)
                    bank = None
            if floor is not None and bank is not None:
                # Gradient of the bank entropy at the current parameters, before the reward gradient.
                h_before = bank_entropy(runner, bank, grad=True)
                h_grad = [(q.grad.detach().to(torch.bfloat16).clone() if q.grad is not None else torch.zeros_like(q, dtype=torch.bfloat16))
                          for q in runner.params]
                opt.zero_grad(set_to_none=True)
            live = [k for k, a in enumerate(adv) if a != 0]
            if ent_coef > 0 or adaptive_entropy:
                stats_e = [0.0, 0]
                backward_weighted(runner, args, answers, list(range(len(answers))), adv, norm, None, ent_coef, stats_e)
                extra["batch_entropy"] = stats_e[0] / max(stats_e[1], 1)
                extra["entropy_coef"] = ent_coef
                n_live = len(answers) if ent_coef > 0 else len(live)
            else:
                backward_weighted(runner, args, answers, live, adv, norm, entropy)
                n_live = len(live)
            if arm in FIXED_ARMS and injected:
                backward_weighted(runner, args, [a for a, _ in injected], list(range(len(injected))), [x for _, x in injected], norm)
                n_live = max(n_live, len(injected))
            if args.memory_lambda > 0 and arm in FIXED_ARMS:
                base_lambda = args.memory_lambda
                if args.memory_anneal_steps > 0:
                    args.memory_lambda = base_lambda * max(0.0, 1.0 - step / args.memory_anneal_steps)
                if args.memory_lambda > 0:
                    extra["memory"] = memory_backward(runner, args, archive, mem_rng, norm, phat)
                    extra["memory"]["lambda"] = args.memory_lambda
                    n_live = max(n_live, extra["memory"]["questions"])
                args.memory_lambda = base_lambda
            if tar is not None:
                extra["tar"] = tar_backward(runner, args, tar, mem_rng)
                n_live = max(n_live, extra["tar"]["active"])
        live_tokens = sum(len(answers[k][1]) for k, a in enumerate(adv) if a != 0)
        if args.entropy_every and step % args.entropy_every == 0:
            extra["probe"] = probe_entropy(runner, args, answers, rewards, owner)
        seen, first_block = {}, []
        for k, j in enumerate(owner):
            seen[j] = seen.get(j, 0) + 1
            if seen[j] <= args.ref_block:
                first_block.append(rewards[k])
        grad_norm = float(torch.nn.utils.clip_grad_norm_(runner.params, args.clip)) if n_live else 0.0
        if n_live and not math.isfinite(grad_norm):
            # A non-finite gradient would corrupt every parameter; skip the update and record it.
            print(f"[{arm}] step {step}: non-finite gradient norm, update skipped", flush=True)
            extra["nonfinite_grad"] = True
            n_live = 0
        if n_live:
            opt.step()
            if args.diagnostics:
                store_update(runner, opt)
            if floor is not None and not surrogate and not args.diagnostics and h_grad is not None:
                extra["floor"] = apply_entropy_floor(runner, opt, bank, h_grad, h_before, floor, args)
                extra["floor"]["bank_tokens"] = sum(len(a) + len(b) for a, b in bank) * (2 + extra["floor"]["tries"])
        elif args.diagnostics:
            runner.last_update = None
        opt.zero_grad(set_to_none=True)
        if adaptive_entropy and "batch_entropy" in extra:
            if ent_target is None:
                ent_target = args.entropy_target_frac * extra["batch_entropy"]
            ent_coef = max(0.0, ent_coef + args.entropy_lr * (ent_target - extra["batch_entropy"]))
            extra["entropy_target"] = ent_target
        if rescue is not None and arm in FIXED_ARMS and keys is not None:
            for k, ((pi, ri), rew) in enumerate(zip(answers, rewards)):
                if rew == 1 and len(ri) and int(ri[-1]) in runner.stop_ids and not getattr(runner, "_prefixed", set()) & {k}:
                    slot = rescue.setdefault(keys[owner[k]], [])
                    if len(slot) < args.memory_max and all(not torch.equal(ri.cpu(), b) for _, b in slot):
                        slot.append((pi.cpu(), ri.cpu()))
        if args.memory_lambda > 0 and arm in FIXED_ARMS and keys is not None:
            # Verified, stop-terminated, unique successes join the archive after the update.
            for j, (c, n) in enumerate(groups):
                phat[keys[j]] = (c, n)  # counts at the question's latest attempt
            for k, ((pi, ri), rew) in enumerate(zip(answers, rewards)):
                if rew == 1 and len(ri) and int(ri[-1]) in runner.stop_ids:
                    key = keys[owner[k]]
                    if key in base_trace:  # the first own success retires the base-model trace
                        archive[key] = []
                        base_trace.discard(key)
                    slot = archive.setdefault(key, [])
                    if len(slot) < args.memory_max and all(not torch.equal(ri.cpu(), b) for _, b in slot):
                        slot.append((pi.cpu(), ri.cpu()))
            extra["archive_questions"] = sum(1 for v in archive.values() if v)
            extra["archive_base_traces"] = len(base_trace)
        mixed = sum(0 < k < n for k, n in groups)
        texts_out = [runner.tok.decode(r) for _, r in answers]
        boxed_rate = sum(last_boxed(t) is not None for t in texts_out) / len(texts_out)
        truncated_rate = sum(1 for _, r in answers if len(r) >= args.max_new and int(r[-1]) not in runner.stop_ids) / len(answers)
        if step == 0 and sft_stats is not None:
            extra["sft"] = sft_stats
        rec = {"step": step, "mean_reward": sum(rewards) / len(rewards),
               "boxed_rate": boxed_rate, "truncated_rate": truncated_rate,
               "first_block_reward": sum(first_block) / len(first_block), "mixed_groups": mixed,
               "groups": len(groups), "grad_norm": grad_norm, "answers": len(answers),
               "tokens": sum(len(r) for _, r in answers),
               "mean_length": sum(len(r) for _, r in answers) / len(answers),
               "entropy": sum(entropy) / live_tokens if (live_tokens and entropy) else None,
               "group_counts": groups,
               "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if runner.device == "cuda" else None,
               "seconds": time.time() - start, **extra}
        log.append(rec)
        print(f"[{arm}] step {step} reward {rec['mean_reward']:.3f} mixed {mixed}/{len(groups)} answers {len(answers)} "
              f"grad {grad_norm:.3g} len {rec['mean_length']:.0f} {rec['seconds']:.0f}s", flush=True)
        (out_dir / f"{arm}_train.json").write_text(json.dumps(log, indent=1))
    del runner, opt
    if args.device == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train", default=str(HERE / "data" / "train.json"))
    parser.add_argument("--eval", default=str(HERE / "data" / "eval_math500_numeric.json"))
    parser.add_argument("--arms", default="baseline,scaled")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--train-batch", type=int, default=4)
    parser.add_argument("--gen-batch", type=int, default=64)
    parser.add_argument("--prefill-tokens", type=int, default=256,
                        help="shrink a generation batch so batch x padded prompt length stays under gen-batch x this")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-steps", default=None, help="comma-separated greedy evaluation steps (overrides --eval-every)")
    parser.add_argument("--sample-eval-steps", default="", help="steps at which to measure sampled accuracy")
    parser.add_argument("--sample-k", type=int, default=4, help="answers per problem for sampled accuracy")
    parser.add_argument("--final-eval", default=None, help="larger held-out set, evaluated greedily after the last step")
    parser.add_argument("--final-limit", type=int, default=None)
    parser.add_argument("--pos-weight", type=float, default=0.1, help="weight of correct answers in wreinforce")
    parser.add_argument("--train-k", type=int, default=4, help="k of the pass@k objective for the passk arm (k < group)")
    parser.add_argument("--shift", type=float, default=0.0,
                        help="constant baseline b for the shift arm: advantage r - b (0 is psr, 1 is nsr)")
    parser.add_argument("--reuse", type=int, default=1, help="updates per rollout batch (stale samples after the first)")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio for the surrogate loss")
    parser.add_argument("--length-norm", action="store_true", help="average each answer's loss over its tokens")
    parser.add_argument("--norm-selected", action="store_true",
                        help="normalize the loss by the number of answers with nonzero advantage in the update")
    parser.add_argument("--entropy-every", type=int, default=0, help="probe entropy on first-block answers every k steps")
    parser.add_argument("--passk-steps", default="", help="steps at which to measure pass@k")
    parser.add_argument("--passk-n", type=int, default=64, help="samples per problem for pass@k")
    parser.add_argument("--passk-limit", type=int, default=None, help="evaluate pass@k on the first problems only")
    parser.add_argument("--save-passk-texts", action="store_true", help="save every pass@k answer text (for CoT judging)")
    parser.add_argument("--save-model", default=None, help="directory for the final model ({arm} is replaced)")
    parser.add_argument("--save-steps", default="", help="steps at which to save a bf16 checkpoint under the output directory")
    parser.add_argument("--diagnostics", action="store_true", help="split-half gradient statistics every step")
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="psrnoise: multiple of the sign-noise size that matches NSR's predicted second moment")
    parser.add_argument("--entropy-coef", type=float, default=0.0, help="entropy bonus coefficient (start value if adaptive)")
    parser.add_argument("--entropy-target-frac", type=float, default=0.0,
                        help="adaptive entropy control: target = this fraction of the step-0 batch entropy (0: fixed coefficient)")
    parser.add_argument("--entropy-lr", type=float, default=0.0, help="step size of the adaptive entropy coefficient")
    parser.add_argument("--floor-rho", type=float, default=0.0, help="entropy floor on the frozen bank: rho x starting entropy (0: off)")
    parser.add_argument("--floor-bank", default="frozen_prefixes_qwen.json")
    parser.add_argument("--floor-bank-size", type=int, default=64)
    parser.add_argument("--floor-tol", type=float, default=0.01)
    parser.add_argument("--early-floor-rho", type=float, default=0.0,
                        help="entropy floor on the first --early-floor-tokens tokens of each step's own answers: rho x the step-0 value")
    parser.add_argument("--early-floor-tokens", type=int, default=32)
    parser.add_argument("--early-floor-warmup", type=int, default=5)
    parser.add_argument("--memory-weighting", default="uniform", choices=("uniform", "coverage"))
    parser.add_argument("--coverage-k", type=int, default=16)
    parser.add_argument("--memory-lambda", type=float, default=0.0, help="log-mass replay of archived successes (0: off)")
    parser.add_argument("--seed-archive", default=None, help="discovery file: seed the replay archive and its priorities")
    parser.add_argument("--sft-archive", default=None, help="discovery file: supervised fine-tuning on its successes before RL")
    parser.add_argument("--tar-archive", default=None, help="discovery file: TAR's gated replay of base successes (Li et al. 2026)")
    parser.add_argument("--tar-lambda", type=float, default=1.0)
    parser.add_argument("--tar-batch", type=int, default=16, help="stored trajectories replayed per step")
    parser.add_argument("--rescue-archive", default=None, help="discovery file: stored successes for group completion / prefix rollouts")
    parser.add_argument("--complete-groups", action="store_true", help="append a stored success to zero-success groups")
    parser.add_argument("--prefix-rollouts", type=int, default=0, help="rollouts per rare question that continue a stored success's prefix")
    parser.add_argument("--prefix-frac", type=float, default=0.5)
    parser.add_argument("--prefix-anneal-steps", type=int, default=100)
    parser.add_argument("--prefix-max-c", type=int, default=4, help="rare = at most this many base successes of 32")
    parser.add_argument("--prefix-before-box", action="store_true", help="cut every prefix before the stored answer's first box")
    parser.add_argument("--prefix-fixed", action="store_true",
                        help="prefixes only from the archive's stored solution (not from the policy's own later successes)")
    parser.add_argument("--prefix-match-archive", default=None,
                        help="discovery file: prefix only questions that also have a retained success there")
    parser.add_argument("--alloc-first", type=int, default=0, help="two-stage allocation: rollouts per problem in stage 1 (0: off)")
    parser.add_argument("--alloc-cap", type=int, default=32, help="maximum rollouts per problem under allocation")
    parser.add_argument("--alloc-value", default="coverage", choices=("coverage", "law", "variance", "hora"),
                        help="law: frontier value per rollout under the frontier law; variance: posterior mean of p (1-p); "
                             "hora: HORA's greedy hit utility; coverage: posterior mean of k p (1-p)^k")
    parser.add_argument("--alloc-lambda", type=float, default=0.37, help="matched-pass@1 frontier slope used by --alloc-value law")
    parser.add_argument("--alloc-norm", default="global", choices=("global", "group"),
                        help="global: Dr. GRPO's fixed normalizer; group: each problem's terms scaled by group / n_i (HORA)")
    parser.add_argument("--memory-anneal-steps", type=int, default=0, help="replay strength falls linearly to 0 over these steps (0: constant)")
    parser.add_argument("--bbg-k", type=int, default=0, help="Bayesian boundary gating reference k (Yuan et al. 2026; 0: off)")
    parser.add_argument("--bbg-tau", type=float, default=0.01)
    parser.add_argument("--sft-lr", type=float, default=1e-5)
    parser.add_argument("--sft-tokens", type=float, default=2.46e6)
    parser.add_argument("--sft-batch", type=int, default=8)
    parser.add_argument("--memory-questions", type=int, default=8)
    parser.add_argument("--memory-max", type=int, default=4)
    parser.add_argument("--unlikely-beta", type=float, default=0.0, help="unlikeliness reward (He et al. 2025) for Dr. GRPO")
    parser.add_argument("--shadow-arm", default="", help="with --diagnostics: split statistics of this fixed-group "
                        "estimator on the same answers (e.g. nsr), for steps below --shadow-until")
    parser.add_argument("--shadow-until", type=int, default=50)
    parser.add_argument("--remeasure-steps", default="",
                        help="steps at which to re-measure F0 and F1 on the calibration questions")
    parser.add_argument("--ref-block", type=int, default=4)
    parser.add_argument("--within-ref", action="store_true",
                        help="second independent reference within prompts (measures per-prompt signal)")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--calib-prompts", type=int, default=16)
    parser.add_argument("--budget", type=int, default=None, help="answers per step; default prompts x group")
    parser.add_argument("--block", type=int, default=4)
    parser.add_argument("--cap", type=int, default=16)
    parser.add_argument("--table-r", type=float, default=130.0)
    parser.add_argument("--resolve-every", type=int, default=10)
    parser.add_argument("--w", type=float, default=None, help="skip calibration and use this w")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--out", default=str(HERE / "results"))
    args, unknown = parser.parse_known_args()
    if unknown:
        print("ignoring unknown arguments:", unknown, flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.budget is None:
        args.budget = args.prompts_per_step * args.group
    train_path = Path(args.train) if Path(args.train).exists() else HERE / "data" / args.train
    problems = json.loads(train_path.read_text())
    eval_path = Path(args.eval) if Path(args.eval).exists() else HERE / "data" / args.eval
    eval_set = json.loads(eval_path.read_text())[: args.eval_limit]
    final_set = None
    if args.final_eval:
        path = Path(args.final_eval) if Path(args.final_eval).exists() else HERE / "data" / args.final_eval
        final_set = json.loads(path.read_text())[: args.final_limit]
    random.Random(args.seed).shuffle(problems)

    calib = None
    if args.w is None:
        runner = Runner(args)
        calib = runner.measure_ratio(problems[-args.calib_prompts:], args.calib_prompts, args.group, args.seed + 7)
        del runner
        if calib["n_correct"] < 20 or calib["n_wrong"] < 20:
            print(f"WARNING: calibration has only {calib['n_correct']} correct and {calib['n_wrong']} wrong answers; "
                  f"F1 / F0 is imprecise (raise --calib-prompts)", flush=True)
            calib["low_count"] = True
        w = min(1.0, max(0.1, calib["F1"] / calib["F0"])) if calib["F0"] > 0 and calib["F1"] > 0 else 1.0
        calib["w"] = w
        print(f"calibration: F1 {calib['F1']:.4g}, F0 {calib['F0']:.4g} over {calib['n_correct']} correct and "
              f"{calib['n_wrong']} wrong answers; w = {w:.3f}", flush=True)
        (out_dir / "calibration.json").write_text(json.dumps(calib, indent=1))
    else:
        w = args.w
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "w_used": w}, indent=1))
    for arm in args.arms.split(","):
        train_arm(args, arm, w, problems[: -args.calib_prompts], eval_set, out_dir, calib, final_set,
                  problems[-args.calib_prompts:])
    print("done", flush=True)


if __name__ == "__main__":
    main()
